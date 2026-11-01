"""Control-plane isolation and actual CT adapters; no real GPU or patient data."""

import json
from pathlib import Path

import pytest

from chd_ct_v2.panel_v2 import catalog, identity, resources


def roots(tmp_path):
    return {k: str(tmp_path / k) for k in ("runtime", "results", "prepared")}


def test_read_only_query_never_creates_roots(tmp_path):
    values = roots(tmp_path)
    assert catalog.list_jobs(values) == []
    assert not Path(values["runtime"]).exists()


def test_foreign_runtime_and_overlapping_roots_are_rejected(tmp_path):
    values = roots(tmp_path)
    Path(values["runtime"]).mkdir()
    (Path(values["runtime"]) / "last-job.json").write_text("{}")
    with pytest.raises(ValueError, match="V2"):
        identity.claim(values)
    other = roots(tmp_path / "other")
    other["results"] = other["runtime"]
    with pytest.raises(ValueError):
        identity.claim(other)


def test_claim_is_stable_and_mixing_instances_is_rejected(tmp_path):
    a, b = roots(tmp_path / "a"), roots(tmp_path / "b")
    identity.claim(a)
    identity.claim(b)
    assert identity.claim(a) == identity.claim(a)
    with pytest.raises(ValueError):
        identity.claim(dict(a, results=b["results"]))


def test_only_registered_jobs_are_discovered_and_foreign_plan_rejected(tmp_path):
    values = roots(tmp_path)
    identity.claim(values)
    folder = Path(values["results"]) / "unregistered"
    folder.mkdir()
    (folder / "plan.json").write_text("{}")
    assert catalog.list_jobs(values) == []
    with pytest.raises(ValueError):
        catalog.get(values, str(folder))


def test_zero_cpu_blocks_and_eighty_percent_is_applied():
    facts = dict(cpu_available=0, cpu_permitted=8, memory_available=16 * 2**30, missing_evidence=[])
    with pytest.raises(ValueError):
        resources.case_budget(facts, 10, 2**30)
    facts["cpu_available"] = 10
    facts["cpu_permitted"] = 12
    selected = resources.case_budget(facts, 30, 2**30)
    assert selected["cpu_budget"] == 8
    assert selected["processes"] * selected["threads"] <= 8
    assert selected["memory_budget_bytes"] == int(16 * 2**30 * 0.8)


def test_new_package_rejects_v1_cache_and_models_before_loading(tmp_path):
    from chd_ct_v2.imagechd.checkpoints import read_collection
    from chd_ct_v2.imagechd.common import validate_prepared

    with pytest.raises(ValueError):
        validate_prepared(tmp_path, {"format": "imagechd7-v2", "status": "prepared"})
    (tmp_path / "models.json").write_text(json.dumps({"format": "imagechd7-chd-paper-v5"}))
    with pytest.raises(ValueError):
        read_collection(tmp_path)


def profile(tmp_path):
    from chd_ct_v2.server.launcher import SERVER_PROFILE

    p = json.loads(SERVER_PROFILE.read_text(encoding="utf-8"))
    p.update(roots(tmp_path))
    p["dataset"] = str(tmp_path / "raw")
    Path(p["dataset"]).mkdir(exist_ok=True)
    return p


def register(values, plan, status="passed"):
    from chd_ct_v2.recovery import write_json

    folder = Path(plan["directory"])
    folder.mkdir(parents=True)
    write_json(folder / "plan.json", plan)
    write_json(
        folder / "job_state.json",
        {
            "status": status,
            "steps": [{"status": status, "name": s["name"], "output": s["output"]} for s in plan["steps"]],
        },
    )
    with catalog.transaction(values) as index:
        index["jobs"][plan["v2_id"]] = {
            "id": plan["v2_id"],
            "name": plan["name"],
            "directory": str(folder),
            "stage": plan["v2_stage"],
            "sources": plan["sources"],
        }
        index["selected"] = plan["v2_id"]
    return folder


def test_prepare_freezes_independent_code_and_rejects_tampering(tmp_path):
    from chd_ct_v2.panel_v2 import service, snapshot

    p = profile(tmp_path)
    values = roots(tmp_path)
    plan = service.build(values, p, "prepare", "test")
    assert plan["steps"][0]["arguments"]["prepared"].startswith(values["prepared"])
    assert plan["steps"][0]["arguments"]["device"] == "cpu"
    from chd_ct_v2.server.launcher import make_parser

    assert plan["steps"][0]["arguments"]["seed"] == make_parser().parse_args([]).seed
    assert Path(plan["frozen_code"]).parent == Path(values["runtime"]) / "releases"
    snapshot.verify(plan)
    (Path(plan["frozen_code"]) / "start_server.py").write_text("# changed")
    with pytest.raises(ValueError, match="冻结代码"):
        service.validate(plan)


def test_source_selection_requires_registered_completed_correct_stage(tmp_path):
    from chd_ct_v2.panel_v2 import service

    p = profile(tmp_path)
    values = roots(tmp_path)
    with pytest.raises(ValueError):
        service.build(values, p, "train", "train", "old-v1")
    with pytest.raises(ValueError, match="显式选择"):
        service.build(values, p, "train", "train")
    old = service.build(values, p, "environment", "environment")
    register(values, old)
    with pytest.raises(ValueError, match="来源阶段"):
        service.build(values, p, "train", "train", old["v2_id"])
    with pytest.raises(ValueError, match="外部"):
        service.build(values, p, "train", "train", options={"models": "/v1/models"})


def test_loader_budget_is_ephemeral_and_accounts_both_pools():
    import torch

    from chd_ct_v2.imagechd.loading import make_loader

    config = {"stages": {"crop64": {"size": 8, "batch": 1}}, "device": "cpu", "threads": 2, "seed": 0}
    rows = {"train": [{"cache_shape": [8, 8, 8]}], "val": [{"cache_shape": [8, 8, 8]}]}
    runtime, chosen = resources.loader_budget(
        config, rows, "crop64", facts={"cpu_budget": 8, "memory_budget_bytes": 8 * 2**30}
    )
    assert config == {"stages": {"crop64": {"size": 8, "batch": 1}}, "device": "cpu", "threads": 2, "seed": 0}
    assert chosen["total_cpu_slots"] <= 8
    assert chosen["estimated_ram_bytes"] <= 8 * 2**30
    assert runtime["stages"] == config["stages"]
    loader = make_loader(torch.utils.data.TensorDataset(torch.arange(4)), runtime, "crop64", "train", 0)
    assert loader.num_workers == chosen["workers"]
    assert loader.prefetch_factor == 1
    with pytest.raises(ValueError):
        resources.loader_budget(
            config, rows, "crop64", facts={"cpu_budget": 0, "memory_budget_bytes": 8 * 2**30}
        )


def test_two_concurrent_ct_jobs_split_cpu_and_ram_budget():
    x = resources.case_budget(
        {"cpu_budget": 16, "memory_budget_bytes": 16 * 2**30}, 100, 2**30, concurrency=2
    )
    assert x["cpu_budget"] == 8 and x["memory_budget_bytes"] == 8 * 2**30
    assert x["processes"] * x["threads"] <= 8


def test_query_history_is_read_only_and_cli_rejects_unknown_input(tmp_path, capsys):
    from chd_ct_v2.panel_v2 import panel

    p = profile(tmp_path)
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(p))
    assert panel.main(["--server-config", str(config), "--query", "history"]) == 0
    assert not Path(p["runtime"]).exists()
    assert panel.main(["--server-config", str(config), "--query", "status", "--job", "v1"]) == 1
    assert not Path(p["runtime"]).exists()


def test_submission_launches_frozen_entry_and_foreign_plan_fails(tmp_path, monkeypatch):
    from chd_ct_v2.panel_v2 import service
    from chd_ct_v2.server import jobs

    p = profile(tmp_path)
    values = roots(tmp_path)
    plan = service.build(values, p, "environment", "environment")
    calls = []

    class Process:
        pid = 321987

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(jobs.subprocess, "Popen", launch)
    with (tmp_path / "lock").open("w") as lock:
        assert jobs.submit(plan, lock) == 0
        with pytest.raises((KeyError, ValueError)):
            jobs.submit({"runtime": p["runtime"]}, lock)
    command, kwargs = calls[0]
    assert command[3] == str(Path(plan["frozen_code"]) / "start_server.py")
    assert kwargs["cwd"] == Path(plan["frozen_code"])
    assert kwargs["env"]["PYTHONPATH"] == str(Path(plan["frozen_code"]) / "src")


def test_environment_retry_only_uses_v2_proxy(tmp_path, monkeypatch):
    from chd_ct_v2.panel_v2 import environment
    from chd_ct_v2.server import bootstrap, network

    network.save(tmp_path, proxy="http://127.0.0.1:17891")
    attempts = []

    def run(command, env, log):
        attempts.append(env)
        return 1 if len(attempts) == 1 else 0

    monkeypatch.setattr(bootstrap, "run_logged", run)
    original = {"CHD_V2_RUNTIME": str(tmp_path), "HTTPS_PROXY": "http://old:1234"}
    environment.checked(["pip"], original, tmp_path / "install.log")
    assert "HTTPS_PROXY" not in attempts[0]
    assert attempts[1]["HTTPS_PROXY"] == "http://127.0.0.1:17891"
    assert original["HTTPS_PROXY"] == "http://old:1234"


def test_diagnostic_export_uses_unique_relative_paths_and_redacts(tmp_path):
    import zipfile

    from chd_ct_v2.panel_v2 import service

    p = profile(tmp_path)
    values = roots(tmp_path)
    plan = service.build(values, p, "environment", "environment")
    folder = register(values, plan)
    (folder / "launcher.log").write_text("token=ghp_1234567890\ntraceback here", encoding="utf-8")
    bundle = service.diagnostics(values, plan["v2_id"])
    with zipfile.ZipFile(bundle) as archive:
        text = archive.read("launcher.log").decode()
        assert "1234567890" not in text and "traceback here" in text
        assert archive.read("manifest.json")


def test_source_graph_uses_completed_prepared_data_and_detects_corruption(tmp_path, monkeypatch):
    import test_chd_main

    from chd_ct_v2.imagechd.preprocess import prepare
    from chd_ct_v2.panel_v2 import service
    from chd_ct_v2.recovery import write_json
    from chd_ct_v2.server.resume import snapshot as output_snapshot

    p = profile(tmp_path)
    values = roots(tmp_path)
    (tmp_path / "fixture").mkdir()
    p["dataset"] = str(test_chd_main.raw_chd.__wrapped__(tmp_path / "fixture"))
    plan = service.build(values, p, "prepare", "prepared", options={"size": 16, "no_resample": True})
    step = plan["steps"][0]
    prepare(p["dataset"], step["arguments"]["prepared"], size=16, no_resample=True)
    folder = register(values, plan)
    state = {"status": "passed", "steps": [dict(step, status="passed", snapshot=output_snapshot(step))]}
    write_json(folder / "job_state.json", state)
    train = service.build(values, p, "train", "train", source=plan["v2_id"])
    assert train["sources"] == [plan["v2_id"]]
    assert train["steps"][0]["arguments"]["prepared"] == step["arguments"]["prepared"]
    assert [s["name"] for s in train["steps"]] == ["train"]
    cache = next(Path(step["arguments"]["prepared"]).glob("*.npz"))
    cache.write_bytes(b"damaged")
    with pytest.raises(ValueError):
        service.validate(train)


def test_install_build_copy_and_dependency_versions_protect_snapshot(tmp_path):
    from chd_ct_v2.panel_v2 import environment, service, snapshot
    from chd_ct_v2.recovery import write_json

    p = profile(tmp_path)
    values = roots(tmp_path)
    plan = service.build(values, p, "environment", "env")
    root = Path(plan["frozen_code"])
    build = environment.install_source(root, Path(values["runtime"]))
    (build / "src" / "test.egg-info").mkdir()
    snapshot.verify(plan)
    target = tmp_path / "environment"
    target.mkdir()
    log = tmp_path / "probe" / "server.log"
    log.parent.mkdir()
    report = {"packages": {"numpy": {"version": "1.26.0"}}, "python": "3.11.0"}
    write_json(log.parent / "dependency-probe.json", report)
    environment.verify_versions(target, log)
    report["packages"]["numpy"]["version"] = "2.0.0"
    write_json(log.parent / "dependency-probe.json", report)
    with pytest.raises(ValueError, match="版本已改变"):
        environment.verify_versions(target, log)


def test_recycle_dependency_guard_and_restore_original_path(tmp_path, monkeypatch):
    from chd_ct_v2.panel_v2 import service
    from chd_ct_v2.recovery import output_lock
    from chd_ct_v2.server import jobs

    monkeypatch.setattr(jobs, "acquire_lock", output_lock)
    values = roots(tmp_path)
    p = profile(tmp_path)
    first = service.build(values, p, "environment", "source")
    folder = register(values, first)
    second = service.build(values, p, "environment", "downstream")
    second["sources"] = [first["v2_id"]]
    register(values, second)
    with pytest.raises(ValueError, match="下游"):
        catalog.recycle(values, first["v2_id"])
    before = (folder / "plan.json").read_bytes()
    catalog.recycle(values, second["v2_id"])
    catalog.recycle(values, first["v2_id"])
    assert not folder.exists()
    catalog.recycle(values, first["v2_id"], restore=True)
    assert (folder / "plan.json").read_bytes() == before
    assert catalog.get(values, first["v2_id"])[0]["id"] == first["v2_id"]


def test_v1_reader_rejects_v2_artifact_identity(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2] / "chd-ct-reproduction" / "src"
    if not root.is_dir():
        pytest.skip("V1 sibling is absent in standalone V2 copy")
    monkeypatch.syspath_prepend(str(root))
    from chd_ct.imagechd.common import validate_prepared as old_validator

    from chd_ct_v2.imagechd.common import FORMAT

    with pytest.raises(ValueError):
        old_validator(tmp_path, {"format": FORMAT, "status": "prepared"})


def test_frozen_plan_rejects_parameter_edits(tmp_path):
    from chd_ct_v2.panel_v2 import service

    p = profile(tmp_path)
    values = roots(tmp_path)
    plan = service.build(values, p, "prepare", "prepare")
    plan["steps"][0]["arguments"]["size"] = 64
    with pytest.raises(ValueError, match="参数已改变"):
        service.validate(plan)


def test_submit_failure_before_plan_write_rolls_back_registry(tmp_path, monkeypatch):
    from chd_ct_v2.panel_v2 import service
    from chd_ct_v2.recovery import output_lock
    from chd_ct_v2.server import jobs

    values = roots(tmp_path)
    p = profile(tmp_path)
    monkeypatch.setattr(service.sys, "platform", "linux")
    monkeypatch.setattr(jobs, "acquire_lock", output_lock)

    def failure(*a):
        raise ValueError("injected submission failure")

    monkeypatch.setattr(jobs, "submit", failure)
    with pytest.raises(ValueError, match="injected"):
        service.submit(values, p, "environment", "failed")
    assert catalog.list_jobs(values) == []
    assert catalog.read(values)["selected"] is None
    assert list((Path(values["runtime"]) / "failed-submissions").glob("*.json"))


def test_resume_uses_same_frozen_code_and_original_parameters(tmp_path, monkeypatch):
    from chd_ct_v2.panel_v2 import service
    from chd_ct_v2.recovery import output_lock
    from chd_ct_v2.server import jobs

    values = roots(tmp_path)
    p = profile(tmp_path)
    original = service.build(values, p, "environment", "retry")
    register(values, original, status="interrupted")
    monkeypatch.setattr(jobs, "acquire_lock", output_lock)
    seen = []

    def capture(plan, lock):
        service.validate(plan)
        seen.append(plan)
        return 0

    monkeypatch.setattr(jobs, "submit", capture)
    assert service.resume(values, original["v2_id"]) == 0
    assert seen[0]["frozen_code"] == original["frozen_code"]
    assert seen[0]["frozen_plan_sha256"] == original["frozen_plan_sha256"]
    assert seen[0]["resume_existing"] is True


@pytest.mark.parametrize(
    "script,stage",
    [
        ("preprocess", "prepare"),
        ("train", "train"),
        ("predict", "predict-all"),
        ("test", "predict"),
        ("evaluate", "evaluate"),
        ("smoke", "assessment"),
        ("diagnosis", "diagnosis"),
    ],
)
def test_shell_wrapper_uses_supported_v2_stage(script, stage):
    import shlex

    from chd_ct_v2.panel_v2.panel import parser

    root = Path(__file__).resolve().parents[1]
    line = (root / (script + "_server.sh")).read_text().splitlines()[-1]
    options = shlex.split(line)[3:-1]
    assert parser().parse_args(options).stage == stage
