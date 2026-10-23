"""New input links and explicit reruns never overwrite a completed experiment."""

import copy
import json
from pathlib import Path

import pytest

from chd_ct.server import jobs, launcher, panel
from chd_ct.server.lineage import verify_lineage
from chd_ct.server.rerun import build_rerun_plan
from chd_ct.server.resume import snapshot
from chd_ct.server.workflow import build_plan


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    from test_server_jobs import arguments

    args, profile = arguments(tmp_path, "--tasks", "preprocess,train,test,evaluate", "--name", "旧任务")
    Path(profile["dataset"]).mkdir()
    plan = build_plan(args, profile, False)
    job = Path(plan["directory"])
    cache = Path(profile["prepared"])
    save(cache / "dataset.json", dict(status="prepared", cases=[dict(case_id="c1", cache="c1.npz")]))
    (cache / "c1.npz").write_bytes(b"synthetic cache")
    save(cache / "preprocess-contract.json", dict(options={"size": None}))
    for step in plan["steps"][1:]:
        save(Path(step["output"]) / step["arguments"]["task"] / "report.json", {"synthetic": True})
    save(Path(plan["steps"][1]["output"]) / "train/models.json", {"stages": {}})
    save(Path(plan["steps"][2]["output"]) / "test/predictions/report.json", {})
    state = jobs.initial_state(plan)
    state["status"] = "passed"
    for step, record in zip(plan["steps"], state["steps"]):
        record.update(status="passed", snapshot=snapshot(step))
    save(job / "plan.json", plan)
    save(job / "job_state.json", state)
    save(Path(plan["runtime"]) / "last-job.json", {"directory": str(job)})
    return job, plan, profile


def args(*options):
    return launcher.make_parser().parse_args(["--device", "cpu", *map(str, options)])


def bytes_and_times(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_new_task_reuses_selected_final_cache_without_preprocessing(source):
    job, old, profile = source
    before = bytes_and_times(job.parent.parent)
    plan = build_plan(args("--tasks", "preprocess,train", "--input-job", job), profile, False)
    assert [s["name"] for s in plan["steps"]] == ["train"]
    assert plan["steps"][0]["arguments"]["prepared"] == old["steps"][0]["arguments"]["prepared"]
    assert plan["input_job"]["directory"] == str(job)
    assert before == bytes_and_times(job.parent.parent)
    verify_lineage(plan)


def test_no_input_job_can_force_fresh_preprocessing(source):
    _, old, profile = source
    plan = build_plan(args("--task", "preprocess", "--input-job", "none"), profile, False)
    assert plan["steps"][0]["arguments"]["prepared"] != old["steps"][0]["arguments"]["prepared"]
    assert not plan.get("input_job")


def test_no_input_requires_preprocessing_for_downstream(source):
    _, _, profile = source
    with pytest.raises(ValueError, match="preprocess"):
        build_plan(args("--task", "train", "--input-job", "none"), profile, False)


def test_no_input_cannot_reuse_cache(source):
    _, _, profile = source
    with pytest.raises(ValueError, match="矛盾"):
        build_plan(
            args("--task", "preprocess", "--input-job", "none", "--preprocess-cache", "reuse"), profile, False
        )


@pytest.mark.parametrize("failure", ["missing_cache", "partial", "escape", "active", "ambiguous", "changed"])
def test_unusable_input_job_rejected(source, monkeypatch, failure):
    from chd_ct.server import inspection

    job, old, profile = source
    cache = Path(profile["prepared"])
    if failure == "missing_cache":
        (cache / "c1.npz").unlink()
    elif failure == "partial":
        (cache / "dataset.json").unlink()
    elif failure == "escape":
        save(cache / "dataset.json", dict(status="prepared", cases=[dict(cache="../outside.npz")]))
    elif failure == "active":
        monkeypatch.setattr(inspection, "job_active", lambda _: True)
    elif failure == "ambiguous":
        old["steps"][1]["arguments"]["prepared"] = str(cache.parent / "other")
        save(job / "plan.json", old)
    else:
        save(cache / "dataset.json", dict(status="prepared", cases=[dict(case_id="changed", cache="c1.npz")]))
    with pytest.raises(ValueError):
        build_plan(args("--task", "train", "--input-job", job), profile, False)


def test_input_job_rejects_conflicting_explicit_path(source):
    job, _, profile = source
    with pytest.raises(ValueError, match="prepared|输入"):
        build_plan(args("--task", "train", "--input-job", job, "--prepared", "wrong"), profile, False)


def test_source_change_between_form_and_submit_rejected(source):
    job, _, profile = source
    plan = build_plan(args("--task", "train", "--input-job", job), profile, False)
    save(job / "receipt.json", {"pid": 123456})
    with pytest.raises(ValueError, match="改变"):
        verify_lineage(plan)
    assert not Path(plan["directory"]).exists()


def test_rerun_selected_steps_rebinds_new_outputs_and_preserves_old(source):
    job, old, profile = source
    before = bytes_and_times(job.parent.parent)
    plan = build_rerun_plan(args("--rerun-job", job, "--rerun-tasks", "train,test,evaluate"), profile, False)
    train, test, evaluate = plan["steps"]
    assert train["arguments"]["prepared"] == profile["prepared"]
    assert test["arguments"]["models"] == str(Path(train["output"]) / "train")
    assert evaluate["arguments"]["predictions"] == str(Path(test["output"]) / "test/predictions")
    assert plan["directory"] != str(job)
    assert plan["rerun_of"]["directory"] == str(job)
    assert all(s["status"] == "pending" for s in jobs.initial_state(plan)["steps"])
    assert not any(s.get("reuse") or s["arguments"].get("resume") for s in plan["steps"])
    assert before == bytes_and_times(job.parent.parent)
    verify_lineage(plan)


def test_single_test_rerun_uses_original_models(source):
    job, old, profile = source
    plan = build_rerun_plan(args("--rerun-job", job, "--rerun-tasks", "test"), profile, False)
    assert plan["steps"][0]["arguments"]["models"] == old["steps"][2]["arguments"]["models"]


def test_rerun_rebinds_explicit_model_manifest_file(source):
    job, old, profile = source
    old["steps"][2]["arguments"]["models"] += "/models.json"
    save(job / "plan.json", old)
    plan = build_rerun_plan(args("--rerun-job", job, "--rerun-tasks", "train,test"), profile, False)
    assert plan["steps"][1]["arguments"]["models"] == str(
        Path(plan["steps"][0]["output"]) / "train/models.json"
    )


def test_demo_can_rerun_without_external_dependencies(source):
    job, old, profile = source
    demo = dict(name="demo", output=str(job / "steps/01_demo"), arguments=dict(task="demo", device="cpu"))
    old["steps"] = [demo]
    save(job / "plan.json", old)
    save(
        job / "job_state.json",
        dict(status="passed", steps=[dict(name="demo", output=demo["output"], status="passed")]),
    )
    plan = build_rerun_plan(args("--rerun-job", job), profile, False)
    assert [s["name"] for s in plan["steps"]] == ["demo"]


def test_preprocess_rerun_new_cache_propagates(source):
    job, _, profile = source
    plan = build_rerun_plan(args("--rerun-job", job, "--rerun-tasks", "preprocess,train"), profile, False)
    pre, train = plan["steps"]
    assert pre["arguments"]["prepared"] == train["arguments"]["prepared"] != profile["prepared"]
    assert pre["arguments"]["preprocess_cache"] == "new"
    assert not Path(pre["arguments"]["prepared"]).exists()


def test_preprocess_cannot_rebind_old_models(source):
    job, _, profile = source
    with pytest.raises(ValueError, match="模型|依赖"):
        build_rerun_plan(args("--rerun-job", job, "--rerun-tasks", "preprocess,test"), profile, False)


def test_rerun_freezes_old_configuration(source):
    job, old, profile = source
    changed = copy.deepcopy(old)
    changed["frozen_files"] = {old["steps"][1]["arguments"]["config"]: "bad-digest"}
    save(job / "plan.json", changed)
    with pytest.raises(ValueError):
        build_rerun_plan(args("--rerun-job", job, "--rerun-tasks", "train"), profile, False)


def test_rerun_does_not_silently_ignore_method_changes(source):
    job, _, profile = source
    with pytest.raises(ValueError, match="seed"):
        build_rerun_plan(args("--rerun-job", job, "--seed", "99"), profile, False)


def test_panel_selects_completed_task_for_rerun(source, monkeypatch):
    job, _, profile = source
    answers = iter(["k", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    options = args("--runtime", profile["runtime"])
    assert panel.run(options, profile)
    assert options.rerun_job == str(job)


@pytest.mark.parametrize("bad_state", ["{broken", "null", "[]"])
def test_bad_neighbor_does_not_block_selecting_good_input(source, monkeypatch, bad_state):
    from chd_ct.server import inspection

    job, old, profile = source
    bad = job.parent / "bad_workflow"
    save(bad / "plan.json", dict(runtime=old["runtime"], directory=str(bad)))
    (bad / "job_state.json").write_text(bad_state, encoding="utf-8")
    monkeypatch.setattr(inspection, "history", lambda *_: [bad, job])
    monkeypatch.setattr("builtins.input", lambda prompt: "2" if prompt.startswith("输入任务 [") else "")
    plan = build_plan(args("--task", "train"), profile, True)
    assert plan["input_job"]["directory"] == str(job)


@pytest.mark.parametrize(
    "options",
    [
        ["--query", "history", "--input-job", "none"],
        ["--status", "--rerun-job", "old"],
        ["--resume-job", "old", "--rerun-job", "old"],
        ["--rerun-tasks", "train"],
    ],
)
def test_incompatible_cli_flags_rejected(options):
    assert launcher.main(options) != 0
