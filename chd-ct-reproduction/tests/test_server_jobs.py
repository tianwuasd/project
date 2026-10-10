"""Background workflow, hand-off, and task-specific configuration contracts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chd_ct.server import launcher
from chd_ct.server.settings import resolve_settings


def arguments(tmp_path, *options):
    args = launcher.make_parser().parse_args(
        ["--python", sys.executable, "--device", "cpu", "--non-interactive", *options]
    )
    profile = dict(
        runtime=str(tmp_path / "runtime"),
        results=str(tmp_path / "results"),
        dataset=str(tmp_path / "raw"),
        prepared=str(tmp_path / "prepared"),
        config=str(Path(launcher.ROOT) / "configs/chd.yaml"),
        source_updated="2026-09-26",
    )
    return args, profile


def test_multiple_selection_normalizes_order_and_rejects_unknown():
    from chd_ct.server.workflow import parse_steps

    assert parse_steps("test，train, preprocess environment train") == [
        "environment",
        "preprocess",
        "train",
        "test",
    ]
    assert parse_steps("0,1,2,4,5") == ["environment", "preprocess", "smoke", "train", "test"]
    with pytest.raises(ValueError):
        parse_steps("train,no-such-task")


@pytest.mark.parametrize("task", ["preprocess", "evaluate", "diagnosis-labels", "diagnosis-demo"])
def test_cpu_settings_never_ask_or_save_gpu_and_model(task, tmp_path, monkeypatch):
    args, profile = arguments(tmp_path, "--task", task, "--predictions", str(tmp_path / "pred"))
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    _, _, _, saved = resolve_settings(args, profile, True)
    assert not any("GPU" in p or "模型配置" in p or "训练模式" in p for p in prompts)
    assert "gpu" not in saved and "gpu_count" not in saved and "config" not in saved
    if task in {"diagnosis-labels", "diagnosis-demo"}:
        assert "prepared" not in saved


def test_plan_chains_current_outputs_without_running_tasks(tmp_path):
    from chd_ct.server.workflow import build_plan

    args, profile = arguments(tmp_path, "--tasks", "test,train,preprocess,smoke")
    Path(profile["dataset"]).mkdir()
    plan = build_plan(args, profile, False)
    assert [s["name"] for s in plan["steps"]] == ["preprocess", "smoke", "train", "test"]
    steps = {s["name"]: s for s in plan["steps"]}
    assert steps["test"]["arguments"]["models"] == str(Path(steps["train"]["output"]) / "train")
    assert steps["test"]["arguments"]["models"] != str(Path(steps["smoke"]["output"]) / "train")
    assert "gpu" not in steps["preprocess"]["arguments"]
    assert not Path(plan["directory"]).exists()


def test_diagnosis_pipeline_binds_full_prediction_not_test_output(tmp_path):
    from chd_ct.server.workflow import build_plan

    args, profile = arguments(
        tmp_path,
        "--tasks",
        "preprocess,train,test,predict,diagnosis-labels,diagnosis-features,diagnosis-train,diagnose,diagnosis-evaluate",
    )
    raw = Path(profile["dataset"])
    raw.mkdir()
    (raw / "imageCHD_dataset_info.xlsx").touch()
    plan = build_plan(args, profile, False)
    steps = {s["name"]: s for s in plan["steps"]}
    assert steps["diagnosis-features"]["arguments"]["predictions"] == str(
        Path(steps["predict"]["output"]) / "predict"
    )
    assert steps["diagnosis-train"]["arguments"]["features"] == str(
        Path(steps["diagnosis-features"]["output"]) / "diagnosis-features"
    )
    assert steps["diagnose"]["arguments"]["classifier"] == str(
        Path(steps["diagnosis-train"]["output"]) / "diagnosis-train"
    )
    args.tasks = "preprocess,train,test,diagnosis-labels,diagnosis-features,diagnosis-train"
    with pytest.raises(ValueError, match="全病例"):
        build_plan(args, profile, False)


def test_multi_cpu_plan_has_no_gpu_prompts(tmp_path, monkeypatch):
    from chd_ct.server.workflow import build_plan

    args, profile = arguments(tmp_path, "--tasks", "diagnosis-labels,diagnosis-demo")
    raw = Path(profile["dataset"])
    raw.mkdir()
    (raw / "imageCHD_dataset_info.xlsx").touch()
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    plan = build_plan(args, profile, True)
    assert all("gpu" not in s["arguments"] for s in plan["steps"])
    assert not any("GPU" in p or "预处理" in p for p in prompts)


def test_fail_fast_state_does_not_run_later_steps(tmp_path, monkeypatch):
    from chd_ct.server import imagechd, jobs

    args, profile = arguments(tmp_path, "--tasks", "diagnosis-demo,demo")
    from chd_ct.server.workflow import build_plan

    plan = build_plan(args, profile, False)
    folder = Path(plan["directory"])
    folder.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(imagechd, "run", lambda args, profile, **kw: calls.append(args.task) or 7)
    assert jobs.execute(plan) == 7
    assert len(calls) == 1
    state = json.loads((folder / "job_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["steps"][0]["exit_code"] == 7
    assert state["steps"][1]["status"] == "pending"


def test_background_spawn_is_detached_and_inherits_lock(tmp_path, monkeypatch):
    from chd_ct.server import jobs

    args, profile = arguments(tmp_path, "--tasks", "diagnosis-demo")
    from chd_ct.server.workflow import build_plan

    plan = build_plan(args, profile, False)
    Path(plan["runtime"]).mkdir(parents=True)
    seen = {}

    def spawn(cmd, **kwargs):
        seen.update(kwargs)
        seen["cmd"] = cmd
        return SimpleNamespace(pid=9123)

    monkeypatch.setattr(jobs.subprocess, "Popen", spawn)
    with (Path(plan["runtime"]) / ".job.lock").open("a") as lock:
        assert jobs.submit(plan, lock, foreground=False) == 0
        assert seen["start_new_session"] is True
        assert seen["stdin"] is jobs.subprocess.DEVNULL
        assert seen["stderr"] is jobs.subprocess.STDOUT
        assert seen["pass_fds"] == (lock.fileno(),)
        assert "--worker" in seen["cmd"]
    assert Path(plan["directory"], "plan.json").is_file()


def test_status_does_not_start_runtime_or_require_data(tmp_path, monkeypatch, capsys):
    args, profile = arguments(tmp_path, "--status")
    config = tmp_path / "server.json"
    config.write_text(json.dumps(profile))
    assert launcher.main(["--status", "--server-config", str(config)]) == 0
    assert "尚未" in capsys.readouterr().out
    assert not Path(profile["runtime"]).exists()


def test_explicit_train_mode_is_not_silently_promoted_to_formal(tmp_path):
    from chd_ct.server.workflow import build_plan

    args, profile = arguments(tmp_path, "--tasks", "preprocess,train", "--mode", "smoke")
    Path(profile["dataset"]).mkdir()
    plan = build_plan(args, profile, False)
    assert plan["steps"][-1]["arguments"]["mode"] == "smoke"


def test_generated_inputs_are_not_prompted_again(tmp_path, monkeypatch):
    from chd_ct.server.workflow import build_plan

    args, profile = arguments(tmp_path, "--tasks", "preprocess,train,test")
    Path(profile["dataset"]).mkdir()
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    build_plan(args, profile, True)
    assert not any("完整六阶段模型" in p for p in prompts)


def test_plan_freezes_paths_from_callers_directory(tmp_path, monkeypatch):
    from chd_ct.server.workflow import build_plan

    monkeypatch.chdir(tmp_path)
    args, profile = arguments(
        tmp_path,
        "--tasks",
        "preprocess,diagnose",
        "--python",
        "env/bin/python",
        "--split-file",
        "split.json",
        "--evidence",
        "evidence.json",
        "--rules",
        "rules.yaml",
        "--features",
        "features",
        "--method",
        "rules",
    )
    Path(profile["dataset"]).mkdir()
    (tmp_path / "features").mkdir()
    plan = build_plan(args, profile, False)
    for step, fields in zip(plan["steps"], [("python", "split_file"), ("python", "evidence", "rules")]):
        for field in fields:
            path = Path(step["arguments"][field])
            assert path.is_absolute() and tmp_path in path.parents


def test_worker_reset_removes_old_preferences_and_keeps_new_step_outputs(tmp_path, monkeypatch):
    import signal

    from chd_ct.server import imagechd, jobs
    from chd_ct.server.workflow import build_plan

    args, profile = arguments(tmp_path, "--tasks", "environment,diagnosis-demo", "--reset-settings")
    runtime = Path(profile["runtime"])
    runtime.mkdir()
    settings_file = runtime / ".server-settings.json"
    settings_file.write_text(
        json.dumps(dict(gpu="GPU-OLD", gpu_count=4, config="old-config", last_models="old"))
    )
    plan = build_plan(args, profile, False)
    Path(plan["directory"]).mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(imagechd, "query_gpus", lambda: pytest.fail("CPU task queried GPU"))
    monkeypatch.setattr(imagechd, "run_logged", lambda *a: 0)
    assert jobs.execute(plan) == 0
    saved = json.loads(settings_file.read_text())
    assert not {"gpu", "gpu_count", "config", "last_models"}.intersection(saved)
    assert saved["results"] == profile["results"]


def test_status_detects_worker_that_exited_before_identity_capture(tmp_path, monkeypatch, capsys):
    from chd_ct.server import jobs

    runtime, folder = tmp_path / "runtime", tmp_path / "job"
    runtime.mkdir()
    folder.mkdir()
    launcher.write_json(runtime / "last-job.json", {"directory": str(folder)})
    launcher.write_json(folder / "job_state.json", {"status": "queued", "log": "log", "steps": []})
    launcher.write_json(folder / "receipt.json", {"pid": 12345, "identity": None})
    monkeypatch.setattr(jobs, "identity", lambda pid: None)
    assert jobs.show_status(runtime) == 0
    assert "stale" in capsys.readouterr().out


def test_workflow_handoff_only_waits_for_its_own_idle_cards(monkeypatch):
    from chd_ct.imagechd import stage_queue
    from chd_ct.server import imagechd

    row = dict(index="0", uuid="GPU-0", name="3090", free=23000, util=60, busy=False)
    samples = iter([[row], [dict(row, util=0)]])
    monkeypatch.setattr(imagechd, "query_gpus", lambda: next(samples))
    waited = []
    monkeypatch.setattr(stage_queue, "check_gpu_handoff", lambda gpu, used: waited.append((gpu, used)))
    assert imagechd.select_with_handoff(1, "0", "0", 6000, {"GPU-0"})[0]["uuid"] == "GPU-0"
    assert waited == [("GPU-0", True)]
    for busy, recent in [(True, {"GPU-0"}), (False, set())]:
        monkeypatch.setattr(imagechd, "query_gpus", lambda: [dict(row, busy=busy)])
        with pytest.raises(ValueError):
            imagechd.select_with_handoff(1, "0", "0", 6000, recent)
    assert waited == [("GPU-0", True)]
