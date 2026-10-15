"""Elastic resource scheduling uses synthetic workers, never a physical GPU."""

import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_server_workflow import arguments, rows

from chd_ct.imagechd import stage_queue
from chd_ct.server import jobs, launcher, resources


def test_pool_separates_authorization_from_occupancy_and_limit():
    devices = rows()
    devices[1]["busy"] = True
    devices[3]["free"] = 1000
    pool = launcher.gpu_pool(devices, visible="1,3")
    assert [r["uuid"] for r in pool] == ["GPU-1", "GPU-3"]
    assert not launcher.idle_gpus(devices, [r["uuid"] for r in pool])
    assert len(launcher.gpu_pool(devices, "1", "1,3")) == 1
    for requested, visible in [("0", "1,3"), ("auto", ""), ("99", None), ("", None)]:
        with pytest.raises(ValueError):
            launcher.gpu_pool(devices, requested, visible)
    with pytest.raises(ValueError, match="重复"):
        launcher.gpu_pool(devices, "1,GPU-1")


def test_wait_starts_on_one_card_without_filling_the_target(monkeypatch):
    busy = [dict(r, busy=True) for r in rows()]
    partial = [dict(r, busy=r["index"] != "3") for r in rows()]
    samples = iter([busy, busy, partial])
    updates, sleeps = [], []
    monkeypatch.setattr(resources.time, "sleep", sleeps.append)
    selected = resources.wait_for_gpus(
        ["GPU-1", "GPU-3"], 4, lambda: next(samples), lambda status, detail: updates.append((status, detail))
    )
    assert [r["uuid"] for r in selected] == ["GPU-3"]
    assert [s for s, _ in updates] == ["waiting_gpu", "waiting_gpu", "running"]
    assert all(d["requested_gpus"] == 4 for _, d in updates)
    assert sleeps == [5, 5]
    with pytest.raises(ValueError, match="不可见"):
        resources.wait_for_gpus(["GPU-missing"], 4, rows, lambda *a: None)


def make_worker(tmp_path):
    script = tmp_path / "synthetic_worker.py"
    script.write_text(
        """import json, os, sys, time
from pathlib import Path
root, stage = Path(sys.argv[1]), sys.argv[2]
(root / (stage + '.started')).write_text(os.environ['CUDA_VISIBLE_DEVICES'])
if stage == 'blood_lstm':
    assert (root / 'blood2d.result.json').is_file()
if stage == 'hold':
    deadline = time.monotonic() + 10
    while not (root / 'later.started').exists():
        if time.monotonic() > deadline: raise RuntimeError('new GPU never joined')
        time.sleep(0.01)
time.sleep(0.1)
(root / (stage + '.result.json')).write_text(json.dumps({'file':stage+'.pt'}))
""",
        encoding="utf-8",
    )
    return lambda stage: [sys.executable, script, tmp_path, stage]


def test_one_gpu_can_finish_all_stages_with_larger_limit(tmp_path):
    stages = ["blood_lstm", "crop64", "crop128", "all64", "all128", "blood2d"]
    result = stage_queue.run_queue(
        stages,
        ["GPU-a", "GPU-b"],
        tmp_path,
        make_worker(tmp_path),
        os.environ.copy(),
        available_gpus=lambda: ["GPU-a"],
        gpu_limit=4,
        poll_interval=0.01,
    )
    assert set(result) == set(stages)
    assert {(tmp_path / (s + ".started")).read_text() for s in stages} == {"GPU-a"}
    queue = json.loads((tmp_path / "queue.json").read_text())
    assert queue["status"] == "complete" and queue["gpu_limit"] == 4


def test_new_gpu_joins_while_another_stage_is_running(tmp_path):
    snapshots = []

    def available():
        path = tmp_path / "queue.json"
        if path.exists():
            snapshots.append(json.loads(path.read_text()))
        if (tmp_path / "hold.started").exists():
            return ["GPU-a", "GPU-b", "GPU-c"]
        return ["GPU-a"]

    result = stage_queue.run_queue(
        ["hold", "later", "last"],
        ["GPU-a", "GPU-b", "GPU-c"],
        tmp_path,
        make_worker(tmp_path),
        os.environ.copy(),
        available_gpus=available,
        gpu_limit=2,
        poll_interval=0.01,
    )
    assert set(result) == {"hold", "later", "last"}
    assert (tmp_path / "later.started").read_text() == "GPU-b"
    assert all(sum(s["status"] == "running" for s in q["stages"].values()) <= 2 for q in snapshots)


def test_waiting_and_handoff_race_do_not_fail_queue(tmp_path):
    snapshots, checks = [], []

    def available():
        path = tmp_path / "queue.json"
        if path.exists():
            snapshots.append(json.loads(path.read_text()))
        return [] if not snapshots else ["GPU-a"]

    def check(gpu, used):
        checks.append(gpu)
        return len(checks) > 1  # Occupied after first snapshot, free on a later attempt.

    result = stage_queue.run_queue(
        ["only"],
        ["GPU-a"],
        tmp_path,
        make_worker(tmp_path),
        os.environ.copy(),
        before_start=check,
        available_gpus=available,
        gpu_limit=4,
        poll_interval=0.01,
    )
    assert "only" in result and len(checks) == 2
    assert any(s["status"] == "waiting_gpu" for s in snapshots)


def test_wait_can_be_interrupted_and_queue_restarted(tmp_path, monkeypatch):
    command = make_worker(tmp_path)

    def interrupt(_):
        raise KeyboardInterrupt("stop while waiting")

    with monkeypatch.context() as patch:
        patch.setattr(stage_queue.time, "sleep", interrupt)
        with pytest.raises(KeyboardInterrupt):
            stage_queue.run_queue(
                ["only"],
                ["GPU-a"],
                tmp_path,
                command,
                os.environ.copy(),
                available_gpus=lambda: [],
                gpu_limit=3,
            )
    queue = json.loads((tmp_path / "queue.json").read_text())
    assert queue["status"] == "interrupted"
    assert queue["stages"]["only"]["status"] == "not_started"
    result = stage_queue.run_queue(
        ["only"],
        ["GPU-a"],
        tmp_path,
        command,
        os.environ.copy(),
        available_gpus=lambda: ["GPU-a"],
        gpu_limit=3,
    )
    assert "only" in result


def test_elastic_pool_never_accepts_outside_gpu(tmp_path):
    with pytest.raises(ValueError, match="候选池"):
        stage_queue.run_queue(
            ["only"],
            ["GPU-a"],
            tmp_path,
            make_worker(tmp_path),
            os.environ.copy(),
            available_gpus=lambda: ["GPU-outside"],
            gpu_limit=2,
        )
    assert not (tmp_path / "only.started").exists()


def test_server_dispatches_full_pool_even_when_only_one_is_idle(tmp_path, monkeypatch):
    from chd_ct.server import imagechd

    args, profile = arguments(tmp_path, "--task", "train", "--gpu-count", "4", "--device", "cuda")
    Path(profile["prepared"]).mkdir()
    (Path(profile["prepared"]) / "dataset.json").write_text("{}")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(imagechd, "query_gpus", lambda: [dict(r, busy=r["index"] != "3") for r in rows()])
    calls = []
    monkeypatch.setattr(
        imagechd, "run_logged", lambda cmd, env, log: calls.append((list(map(str, cmd)), env.copy())) or 0
    )
    assert imagechd.run(args, profile) == 0
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "GPU-3"
    command, env = calls[1]
    assert command[command.index("--gpu-ids") + 1] == "GPU-1,GPU-3"
    assert command[command.index("--gpu-limit") + 1] == "4"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-1,GPU-3"


@pytest.mark.parametrize("cuda", [True, False])
def test_stage_worker_probes_in_process_and_rejects_unusable_device(tmp_path, monkeypatch, cuda):
    from chd_ct.imagechd import stage_worker
    from chd_ct.quickstart import environment

    launcher.write_json(tmp_path / "effective-config.json", {})
    monkeypatch.setattr(environment, "probe", lambda: dict(ok=True, cuda=cuda))
    monkeypatch.setattr(environment, "check_environment", lambda *a: pytest.fail("nested probe process"))
    monkeypatch.setattr(stage_worker, "read_prepared", lambda *a: (tmp_path, {}))
    monkeypatch.setattr(stage_worker, "training_rows", lambda *a: {})
    monkeypatch.setattr(stage_worker, "file_hash", lambda *a: "digest")
    trained = []
    monkeypatch.setattr(stage_worker, "train_stage", lambda *a: trained.append(True) or {"file": "crop64.pt"})
    args = [
        "--prepared",
        str(tmp_path),
        "--output",
        str(tmp_path),
        "--stage",
        "crop64",
        "--mode",
        "smoke",
        "--check-environment",
    ]
    if cuda:
        assert stage_worker.main(args) == 0 and trained
    else:
        with pytest.raises(RuntimeError, match="环境检查"):
            stage_worker.main(args)
        assert not trained
    assert json.loads((tmp_path / "crop64.environment.json").read_text())["cuda"] is cuda


@pytest.mark.parametrize("in_queue", [False, True])
def test_status_shows_waiting_and_running_card_count(tmp_path, capsys, in_queue):
    step = tmp_path / "step"
    step.mkdir()
    launcher.write_json(tmp_path / "last-job.json", {"directory": str(tmp_path)})
    launcher.write_json(
        tmp_path / "job_state.json",
        dict(status="running", log="log", steps=[dict(name="train", status="running", output=str(step))]),
    )
    launcher.write_json(
        step / "status.json",
        dict(
            status="running" if in_queue else "waiting_gpu",
            phase="train",
            gpu_resources=dict(requested_gpus=4, waited_seconds=45, reason="有计算进程"),
        ),
    )
    if in_queue:
        (step / "train").mkdir()
        launcher.write_json(
            step / "train/queue.json",
            dict(status="waiting_gpu", active_gpus=0, gpu_limit=4, pending_stages=6),
        )
    assert jobs.show_status(tmp_path) == 0
    text = capsys.readouterr().out
    assert "waiting_gpu" in text and "0/4" in text
    if in_queue:
        launcher.write_json(
            step / "train/queue.json", dict(status="running", active_gpus=1, gpu_limit=4, pending_stages=5)
        )
        jobs.show_status(tmp_path)
        assert "1/4" in capsys.readouterr().out


@pytest.mark.parametrize("phase", [None, "train"])
def test_resumed_status_ignores_previous_attempt_queue(tmp_path, capsys, phase):
    step = tmp_path / "step"
    (step / "train").mkdir(parents=True)
    launcher.write_json(tmp_path / "last-job.json", {"directory": str(tmp_path)})
    launcher.write_json(
        tmp_path / "job_state.json",
        dict(status="running", log="log", steps=[dict(name="train", status="running", output=str(step))]),
    )
    launcher.write_json(
        step / "train/queue.json",
        dict(status="running", active_gpus=2, gpu_limit=2, pending_stages=4, sampled_at=10),
    )
    report = dict(status="waiting_gpu", gpu_resources=dict(requested_gpus=4, waited_seconds=5))
    if phase:
        report.update(phase=phase, phase_started_at=20)
    launcher.write_json(step / "status.json", report)
    jobs.show_status(tmp_path)
    text = capsys.readouterr().out
    assert "waiting_gpu" in text and "0/4" in text and "2/2" not in text
    # Also while the new command is reading/hashing its inputs before queue creation.
    report["status"] = "running"
    launcher.write_json(step / "status.json", report)
    jobs.show_status(tmp_path)
    assert "2/2" not in capsys.readouterr().out
