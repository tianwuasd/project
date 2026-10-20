"""Read-only inspection and the running/history/new-job panel contract."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chd_ct.server import inspection, jobs, launcher, panel


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def session(tmp_path):
    runtime, results = tmp_path / "runtime", tmp_path / "results"
    job = results / "20261013_120000_new_workflow"
    step = job / "steps/02_preflight"
    state = dict(
        name="心脏 首次试跑",
        status="interrupted",
        log=str(job / "launcher.log"),
        steps=[dict(name="preflight", status="interrupted", output=str(step))],
    )
    plan = dict(name=state["name"], runtime=str(runtime), directory=str(job), steps=state["steps"])
    save(job / "job_state.json", state)
    save(job / "plan.json", plan)
    save(runtime / "last-job.json", dict(directory=str(job)))
    save(step / "status.json", dict(phase="preflight", phase_started_at=1))
    save(step / "train/effective-config.json", dict(stages={"blood2d": {"epochs": 1}}))
    save(
        step / "train/blood2d.history.json", [dict(epoch=1, learning_rate=0.002, loss={"train": 2, "val": 3})]
    )
    save(step / "train/queue.json", dict(sampled_at=2, stages={"blood2d": {"status": "complete"}}))
    (step / "train/stage_logs").mkdir()
    (step / "train/stage_logs/blood2d.log").write_text("MODEL epoch=1 loss=3\n")
    (step / "server.log").write_text("SCHEDULER\n")
    (job / "launcher.log").write_text("WORKFLOW\n")
    profile = dict(runtime=str(runtime), results=str(results))
    return runtime, job, step, profile


def test_readonly_cli_models_and_logs_need_no_training_environment(session, tmp_path):
    runtime, job, step, profile = session
    config = tmp_path / "profile.json"
    save(config, profile)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    command = [sys.executable, "-S", str(launcher.ROOT / "start_server.py"), "--server-config", str(config)]
    for query, expected in [("models", "1/1"), ("logs", "MODEL"), ("history", "心脏 首次试跑")]:
        result = subprocess.run([*command, "--query", query], capture_output=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_queries_do_not_create_missing_runtime_or_acquire_lock(tmp_path, monkeypatch, capsys):
    profile = tmp_path / "profile.json"
    save(profile, dict(runtime=str(tmp_path / "absent"), results=str(tmp_path / "results")))
    monkeypatch.setattr(jobs, "acquire_lock", lambda *a: pytest.fail("query acquired training lock"))
    assert launcher.main(["--query", "logs", "--server-config", str(profile)]) == 0
    assert not (tmp_path / "absent").exists()
    assert "没有任务" in capsys.readouterr().out


def test_preflight_and_formal_model_paths_and_single_model(session):
    _, job, step, _ = session
    assert inspection.log_paths(job, "auto", "blood2d") == [step / "train/stage_logs/blood2d.log"]
    save(step / "preflight/queue.json", {})
    assert inspection.run_directory(step) == step / "preflight"
    save(step / "status.json", dict(phase="train"))
    assert inspection.run_directory(step) == step / "train"


def test_follow_discovers_new_model_file_and_handles_truncation(session, monkeypatch, capsys):
    _, job, step, _ = session
    target = step / "train/stage_logs/blood_lstm.log"
    ticks = []

    def advance(_):
        ticks.append(1)
        if len(ticks) == 1:
            target.write_text("first long model line\n")
        elif len(ticks) == 2:
            target.write_text("next\n")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(inspection.time, "sleep", advance)
    inspection.tail_logs(lambda: inspection.log_paths(job, "auto", "blood_lstm"), lines=10, follow=True)
    output = capsys.readouterr().out
    assert "first long model line" in output and "next" in output and "退出查看" in output


def test_model_summary_marks_old_queue_and_dependency(session, capsys):
    _, job, step, _ = session
    save(step / "status.json", dict(phase="train", phase_started_at=3))
    inspection.show_models(job)
    assert "旧记录" in capsys.readouterr().out
    save(
        step / "train/queue.json",
        dict(sampled_at=4, stages={"blood2d": {"status": "running"}, "blood_lstm": {"status": "pending"}}),
    )
    inspection.show_models(job)
    assert "等待 blood2d" in capsys.readouterr().out


def test_history_only_includes_same_runtime_and_keeps_completed_records(session):
    runtime, job, _, profile = session
    other = job.parent / "20261014_other_workflow"
    save(other / "plan.json", dict(runtime=str(runtime.parent / "other")))
    save(other / "job_state.json", dict(status="passed", steps=[]))
    assert [p for p in inspection.history(runtime, profile["results"])] == [job]


def test_panel_queries_while_running_and_stop_requires_target_confirmation(session, monkeypatch):
    runtime, job, _, profile = session
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    monkeypatch.setattr(inspection, "job_active", lambda p: True)
    answers = iter(["g", "n", "x", "n", "x", "y", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    seen = []
    monkeypatch.setattr(inspection, "show_gpu", lambda: seen.append("gpu"))

    def leave_gpu_view(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(inspection.time, "sleep", leave_gpu_view)
    monkeypatch.setattr(jobs, "stop_job", lambda rt, directory: seen.append(("stop", rt, directory)) or 0)
    assert panel.run(args, profile) is False
    assert seen == ["gpu", ("stop", runtime, job)]


def test_gpu_follow_refreshes_without_taking_training_lock(session, monkeypatch, capsys):
    runtime, _, _, profile = session
    args = launcher.make_parser().parse_args(["--query", "gpu", "--follow", "--runtime", str(runtime)])
    seen = []
    monkeypatch.setattr(inspection, "show_gpu", lambda: seen.append("sample"))
    monkeypatch.setattr(jobs, "acquire_lock", lambda *a: pytest.fail("GPU view acquired training lock"))
    monkeypatch.setattr(jobs, "stop_job", lambda *a, **kw: pytest.fail("GPU view stopped training"))

    def advance(seconds):
        seen.append(seconds)
        if len(seen) >= 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(inspection.time, "sleep", advance)
    assert inspection.query(args, profile) == 0
    assert seen == ["sample", 1, "sample", 1]
    assert "退出显卡查看" in capsys.readouterr().out
    seen.clear()
    args.follow = False
    assert inspection.query(args, profile) == 0
    assert seen == ["sample"]


def test_history_resume_selects_exact_old_job_without_changing_pointer(session, monkeypatch):
    runtime, job, _, profile = session
    older = job.parent / "20261012_old_workflow"
    save(older / "plan.json", dict(runtime=str(runtime), directory=str(older), steps=[]))
    save(older / "job_state.json", dict(name="第二个实验", status="failed", steps=[]))
    pointer = (runtime / "last-job.json").read_bytes()
    answers = iter(["r", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(inspection, "job_active", lambda p: False)
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    assert panel.run(args, profile) is True
    assert Path(args.resume_job) == older
    assert pointer == (runtime / "last-job.json").read_bytes()


def test_named_new_runs_remain_unique_and_state_preserves_name(tmp_path):
    from test_server_jobs import arguments

    from chd_ct.server.workflow import build_plan

    args, profile = arguments(tmp_path, "--task", "diagnosis-demo", "--name", "我的实验 / 比较组")
    a, b = [build_plan(args, profile, False) for _ in range(2)]
    assert a["name"] == "我的实验 / 比较组"
    assert a["directory"] != b["directory"]
    assert jobs.initial_state(a)["name"] == a["name"]


def test_query_rejects_accidental_run_flags(session, tmp_path):
    _, _, _, profile = session
    config = tmp_path / "profile.json"
    save(config, profile)
    assert launcher.main(["--query", "models", "--tasks", "train", "--server-config", str(config)]) != 0


def test_history_remembers_noncurrent_results_root(session):
    runtime, job, _, profile = session
    earlier = runtime.parent / "other_results/20261011_old_workflow"
    save(earlier / "plan.json", dict(runtime=str(runtime)))
    save(earlier / "job_state.json", dict(status="passed", steps=[]))
    save(runtime / "job-history.json", [str(earlier)])
    assert inspection.history(runtime, profile["results"]) == [job, earlier]


def test_models_display_actual_gpu_uuid(session, capsys):
    _, job, step, _ = session
    save(
        step / "train/queue.json",
        dict(sampled_at=4, stages={"blood2d": {"status": "running", "gpu_uuid": "GPU-example"}}),
    )
    inspection.show_models(job)
    assert "GPU-example" in capsys.readouterr().out


def test_completed_history_can_be_selected_for_extension(session, monkeypatch):
    runtime, job, _, profile = session
    save(job / "job_state.json", dict(status="passed", steps=[]))
    answers = iter(["r", "1", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    assert panel.run(args, profile) is True
    assert args.resume_job == str(job)


def test_no_model_steps_does_not_invent_model_output(session, capsys):
    _, job, _, _ = session
    save(
        job / "job_state.json",
        dict(
            status="passed",
            steps=[dict(name="environment", status="passed", output=str(job / "steps/01_environment"))],
        ),
    )
    inspection.show_models(job)
    output = capsys.readouterr().out
    assert "尚无模型训练记录" in output and "crop64" not in output


def test_gpu_timeout_returns_to_panel(session, monkeypatch, capsys):
    runtime, _, _, profile = session

    def timeout():
        raise subprocess.TimeoutExpired("nvidia-smi", 15)

    monkeypatch.setattr(inspection, "show_gpu", timeout)
    answers = iter(["g", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    assert panel.run(args, profile) is False
    assert "查询/操作未完成" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["q", "退出"])
def test_quit_from_nested_log_input_does_not_stop_task(session, monkeypatch, answer):
    runtime, _, _, profile = session
    answers = iter(["l", answer])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(jobs, "stop_job", lambda *a, **kw: pytest.fail("quit stopped a task"))
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    assert panel.run(args, profile) is False


def test_back_from_nested_input_and_zero_is_still_valid(monkeypatch):
    from chd_ct.server.prompts import PanelBack

    answers = iter(["b", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    with pytest.raises(PanelBack):
        launcher.choose(None, None, "", "GPU", True)
    assert launcher.choose(None, None, "", "GPU", True) == "0"


def test_update_stops_exact_job_holds_lock_and_exits_old_panel(session, monkeypatch):
    from contextlib import contextmanager

    from chd_ct.server import updater

    runtime, job, _, profile = session
    answers = iter(["u", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(inspection, "job_active", lambda _: True)
    monkeypatch.setattr(updater, "inspect_repository", lambda: {"root": "repo", "head": "abc"})
    seen = []
    monkeypatch.setattr(jobs, "stop_job", lambda rt, directory: seen.append(("stop", rt, directory)))

    @contextmanager
    def idle(rt, target):
        seen.append(("locked", rt, target))
        yield
        seen.append("released")

    monkeypatch.setattr(updater, "idle_lock", idle)
    monkeypatch.setattr(updater, "update", lambda repo, rt: seen.append("updated") or True)
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    assert panel.run(args, profile) is False
    assert seen == [("stop", runtime, job), ("locked", runtime, job), "updated", "released"]


def test_failed_form_returns_to_panel_without_submitting(session, monkeypatch, tmp_path):
    runtime, _, _, profile = session
    config = tmp_path / "profile.json"
    save(config, profile)
    answers = iter(["n", "my run", "invalid-step", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(jobs, "submit", lambda *a, **kw: pytest.fail("invalid form submitted"))
    assert launcher.main(["--runtime", str(runtime), "--server-config", str(config)]) == 0
