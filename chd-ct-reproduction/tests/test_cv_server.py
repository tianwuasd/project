"""Three-fold server inspection and surviving-worker recovery remain read-only."""

import json
import subprocess
import sys

import pytest

from chd_ct.server import inspection, jobs, launcher, resume


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def threefold(tmp_path):
    job = tmp_path / "results/threefold_workflow"
    step = job / "steps/02_train"
    run = step / "train"
    runtime = tmp_path / "runtime"
    steps = [dict(name="train", status="interrupted", output=str(step))]
    save(job / "job_state.json", dict(name="三折实验", status="interrupted", steps=steps))
    save(job / "plan.json", dict(runtime=str(runtime), directory=str(job), steps=steps))
    save(runtime / "last-job.json", dict(directory=str(job)))
    save(step / "status.json", dict(phase="train", phase_started_at=10))
    save(run / "effective-config.json", dict(stages={"blood2d": {"epochs": 20}}))
    save(
        run / "report.json",
        dict(
            task="chd_threefold",
            status="interrupted",
            current_fold="fold_1",
            folds={
                "fold_0": dict(status="complete", train_cases=40, val_cases=20, test_cases=20),
                "fold_1": dict(status="running", train_cases=40, val_cases=20, test_cases=20),
                "fold_2": dict(status="pending", train_cases=40, val_cases=20, test_cases=20),
            },
        ),
    )
    save(run / "fold_0/effective-config.json", dict(stages={"blood2d": {"epochs": 3}}))
    save(run / "fold_0/queue.json", dict(sampled_at=2, stages={"blood2d": {"status": "complete"}}))
    save(run / "fold_0/blood2d.history.json", [dict(epoch=3, loss={"train": 0.2, "val": 0.3})])
    save(
        run / "fold_1/queue.json",
        dict(
            sampled_at=11,
            stages={
                "blood2d": dict(status="running", gpu_uuid="GPU-current"),
                "blood_lstm": dict(status="pending"),
            },
        ),
    )
    save(run / "fold_1/blood2d.history.json", [dict(epoch=4, loss={"train": 0.4, "val": 0.5})])
    (run / "fold_0/stage_logs").mkdir()
    (run / "fold_0/stage_logs/blood2d.log").write_text("FOLD ZERO epoch=3\n", encoding="utf-8")
    (step / "server.log").write_text("SCHEDULER\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    save(profile, dict(runtime=str(runtime), results=str(job.parent)))
    return job, step, run, profile


def test_threefold_queries_use_only_stdlib_and_do_not_modify_files(threefold, tmp_path):
    _, _, _, profile = threefold
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    command = [sys.executable, "-S", str(launcher.ROOT / "start_server.py"), "--server-config", str(profile)]
    for query, expected in [("models", "已完成 1/3"), ("logs", "FOLD ZERO")]:
        result = subprocess.run([*command, "--query", query], capture_output=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_threefold_summary_shows_each_fold_progress_and_config_fallback(threefold, capsys):
    job, _, _, _ = threefold
    inspection.show_models(job)
    output = capsys.readouterr().out
    assert "已完成 1/3" in output and "当前折：fold_1" in output
    assert all(f"{fold} |" in output for fold in ("fold_0", "fold_1", "fold_2"))
    assert "3/3" in output and "4/20" in output
    assert "GPU-current" in output and "等待 blood2d" in output
    assert "train=40" in output and "val=20" in output and "test=20" in output
    # A previously completed fold is historical, but its completed queue is still valid.
    assert "旧记录" not in output


def test_threefold_current_queue_is_stale_after_resume(threefold, capsys):
    job, step, _, _ = threefold
    save(step / "status.json", dict(phase="train", phase_started_at=12))
    inspection.show_models(job)
    output = capsys.readouterr().out
    assert "旧记录" in output
    assert "GPU-current" not in output


@pytest.mark.parametrize("model", ["all", "blood2d"])
def test_threefold_logs_follow_future_fold_files(threefold, monkeypatch, capsys, model):
    job, _, run, _ = threefold
    target = run / "fold_1/stage_logs/blood2d.log"
    ticks = []

    def advance(_):
        ticks.append(1)
        if len(ticks) == 1:
            target.parent.mkdir()
            target.write_text("FOLD ONE first model line\n", encoding="utf-8")
        elif len(ticks) == 2:
            target.write_text("NEW\n", encoding="utf-8")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(inspection.time, "sleep", advance)
    inspection.tail_logs(lambda: inspection.log_paths(job, "auto", model), lines=10, follow=True)
    output = capsys.readouterr().out
    assert "FOLD ZERO" in output and "FOLD ONE" in output and "NEW" in output
    assert "退出查看" in output


def test_threefold_missing_logs_fall_back_to_server(threefold):
    job, step, run, _ = threefold
    (run / "fold_0/stage_logs/blood2d.log").unlink()
    assert inspection.log_paths(job) == [step / "server.log"]
    assert inspection.log_paths(job, model="blood2d") == [step / "server.log"]


def test_fold_discovery_ignores_metadata_paths_and_unrecognized_folds(threefold, tmp_path):
    job, _, run, _ = threefold
    foreign = tmp_path / "foreign"
    save(foreign / "models.json", {})
    save(run / "fold_3/queue.json", {})
    save(
        run / "models.json",
        dict(format="imagechd7-threefold-v1", folds={"fold_0": {"file": str(foreign / "models.json")}}),
    )
    assert inspection.model_runs(run) == [run / f"fold_{index}" for index in range(3)]
    assert inspection.log_paths(job) == [run / "fold_0/stage_logs/blood2d.log"]


def test_model_index_identifies_threefold_without_root_report(threefold):
    _, _, run, _ = threefold
    (run / "report.json").unlink()
    save(run / "models.json", dict(format="imagechd7-threefold-v1", folds={}))
    assert inspection.model_runs(run) == [run / f"fold_{index}" for index in range(3)]


def test_current_fold_cannot_select_an_external_metadata_path(threefold):
    _, _, run, _ = threefold
    save(run / "report.json", dict(task="chd_threefold", current_fold="../../foreign"))
    assert inspection.active_model_run(run) is None


def test_status_uses_current_fold_resource_counts(threefold, capsys):
    job, step, run, profile = threefold
    state = json.loads((job / "job_state.json").read_text(encoding="utf-8"))
    state["status"] = state["steps"][0]["status"] = "running"
    save(job / "job_state.json", state)
    save(
        run / "fold_1/queue.json",
        dict(sampled_at=11, status="running", active_gpus=1, gpu_limit=2, pending_stages=3),
    )
    assert jobs.show_status(profile.parent / "runtime", job) == 0
    output = capsys.readouterr().out
    assert "当前折：fold_1" in output and "1/2" in output and "3 个阶段" in output
    save(step / "status.json", dict(phase="train", phase_started_at=12))
    jobs.show_status(profile.parent / "runtime", job)
    assert "1/2" not in capsys.readouterr().out


def test_no_current_fold_does_not_reuse_an_old_queue(threefold, capsys):
    job, _, run, profile = threefold
    state = json.loads((job / "job_state.json").read_text(encoding="utf-8"))
    state["status"] = state["steps"][0]["status"] = "running"
    save(job / "job_state.json", state)
    save(run / "report.json", dict(task="chd_threefold", status="passed", current_fold=None))
    save(run / "queue.json", dict(sampled_at=11, status="running", active_gpus=5, gpu_limit=5))
    assert inspection.active_model_run(run) is None
    jobs.show_status(profile.parent / "runtime", job)
    assert "5/5" not in capsys.readouterr().out


def test_cross_fold_queries_show_total_resources_and_all_active_folds(threefold, capsys):
    job, step, run, profile = threefold
    state = json.loads((job / "job_state.json").read_text(encoding="utf-8"))
    state["status"] = state["steps"][0]["status"] = "running"
    save(job / "job_state.json", state)
    summary = json.loads((run / "report.json").read_text(encoding="utf-8"))
    summary.update(status="running", scheduling="cross_fold", current_folds=["fold_1", "fold_2"])
    save(run / "report.json", summary)
    save(
        run / "queue.json",
        dict(sampled_at=11, status="running", active_gpus=3, gpu_limit=3, pending_stages=7),
    )
    assert inspection.active_model_run(run) == run
    jobs.show_status(profile.parent / "runtime", job)
    inspection.show_models(job)
    output = capsys.readouterr().out
    assert "3/3 张卡" in output and "7 个阶段" in output
    assert "当前运行折：fold_1, fold_2" in output and "跨折共享队列" in output
    save(step / "status.json", dict(phase="train", phase_started_at=12))
    jobs.show_status(profile.parent / "runtime", job)
    assert "3/3 张卡" not in capsys.readouterr().out
    summary.update(status="passed", current_fold=None, current_folds=[])
    save(run / "report.json", summary)
    assert inspection.active_model_run(run) is None


def test_nested_workers_require_matching_identity_and_keep_legacy_queue(threefold, monkeypatch):
    job, step, run, _ = threefold
    valid = dict(boot="test-boot", start="123")
    mismatch = dict(boot="older-boot", start="123")
    save(step / "active-command.json", dict(running=True, pid=101, identity=valid))
    save(run / "queue.json", dict(stages={"crop64": dict(pid=102, identity=valid)}))
    save(run / "fold_0/queue.json", dict(stages={"crop128": dict(pid=103, identity=valid)}))
    save(
        run / "fold_1/queue.json",
        dict(
            stages={
                "crop64": dict(pid=104, identity=mismatch),
                "crop128": dict(pid=105),
                "all64": dict(pid=106, identity=valid),
            }
        ),
    )
    save(run / "fold_3/queue.json", dict(stages={"crop64": dict(pid=107, identity=valid)}))
    save(step / "preflight/fold_2/queue.json", dict(stages={"crop64": dict(pid=108, identity=valid)}))
    seen = []

    def identity(pid):
        seen.append(pid)
        return None if pid == 106 else valid

    monkeypatch.setattr(jobs, "identity", identity)
    plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
    live = {record["pid"]: record for record in resume.live_commands(plan)}
    assert set(live) == {101, 102, 103, 108}
    assert live[101]["process_group"] is True
    assert all(live[pid]["process_group"] is False for pid in (102, 103, 108))
    assert 105 not in seen and 107 not in seen
    assert inspection.job_active(job)
    with pytest.raises(ValueError, match="原任务子进程仍存活"):
        resume.reject_live_commands(plan)
