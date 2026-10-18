"""Cross-fold scheduling with synthetic workers; no physical GPU is used."""

import json
import os
import sys

import pytest

from chd_ct.imagechd.stage_queue import run_queue


def test_next_fold_runs_before_previous_encoder_finishes(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import json, os, sys, time
from pathlib import Path
root, task = Path(sys.argv[1]), sys.argv[2]
fold, stage = task.split('/')
child = root / fold
(child / (stage + '.started')).write_text(os.environ['CUDA_VISIBLE_DEVICES'])
if task == 'fold_0/blood2d':
    deadline = time.monotonic() + 8
    while not (root / 'fold_1/crop64.started').exists():
        if time.monotonic() > deadline: raise RuntimeError('later fold blocked')
        time.sleep(0.01)
    time.sleep(0.3)
if stage == 'blood_lstm':
    assert (child / 'blood2d.result.json').exists()
(child / (stage + '.result.json')).write_text(json.dumps({'task':task}))
""",
        encoding="utf-8",
    )
    for fold in ("fold_0", "fold_1"):
        (tmp_path / fold / "stage_logs").mkdir(parents=True)
    tasks = ["fold_0/blood2d", "fold_0/blood_lstm", "fold_1/crop64"]
    snapshots = []
    result = run_queue(
        tasks,
        ["GPU-a", "GPU-b", "GPU-c"],
        tmp_path,
        lambda task: [sys.executable, worker, tmp_path, task],
        os.environ.copy(),
        gpu_limit=2,
        dependencies={"fold_0/blood_lstm": ["fold_0/blood2d"]},
        result_path_for=lambda task: tmp_path / (task + ".result.json"),
        log_path_for=lambda task: tmp_path
        / task.split("/")[0]
        / "stage_logs"
        / (task.split("/")[1] + ".log"),
        on_state=lambda state: snapshots.append(json.loads(json.dumps(state))),
    )
    assert set(result) == set(tasks)
    assert all(s["active_gpus"] <= 2 for s in snapshots)
    assert any(
        s["stages"]["fold_0/blood2d"]["status"] == "running"
        and s["stages"]["fold_1/crop64"]["status"] == "running"
        for s in snapshots
    )
    assert all(
        s["stages"]["fold_0/blood_lstm"]["status"] != "running"
        or s["stages"]["fold_0/blood2d"]["status"] == "complete"
        for s in snapshots
    )
    assert (tmp_path / "fold_1/stage_logs/crop64.log").exists()


def test_cross_fold_dependencies_do_not_accept_another_folds_encoder(tmp_path):
    with pytest.raises(ValueError, match="依赖"):
        run_queue(
            ["fold_1/blood_lstm", "fold_0/blood2d"],
            ["GPU-a"],
            tmp_path,
            lambda _: pytest.fail("Invalid dependency launched a worker"),
            os.environ.copy(),
            dependencies={"fold_1/blood_lstm": ["fold_1/blood2d"]},
        )


@pytest.mark.parametrize("failure", ["worker", "interrupt"])
def test_cross_fold_failure_reaps_workers_and_preserves_dependencies(tmp_path, monkeypatch, failure):
    import subprocess

    from chd_ct.imagechd import stage_queue

    worker = tmp_path / "failure_worker.py"
    worker.write_text(
        """import sys, time
from pathlib import Path
root, task = Path(sys.argv[1]), sys.argv[2]
(root / (task.replace('/', '-') + '.started')).write_text('started')
if task == 'fold_1/fail':
    deadline = time.monotonic() + 8
    while not (root / 'fold_0-blood2d.started').exists():
        if time.monotonic() > deadline: sys.exit(9)
        time.sleep(0.01)
    sys.exit(7)
time.sleep(30)
""",
        encoding="utf-8",
    )
    processes = []
    actual_popen = subprocess.Popen

    def track(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stage_queue.subprocess, "Popen", track)

    def on_state(state):
        if failure == "interrupt" and state["active_gpus"] == 2:
            raise KeyboardInterrupt("stop all folds")

    tasks = ["fold_0/blood2d", "fold_0/blood_lstm", "fold_1/fail"]
    error = RuntimeError if failure == "worker" else KeyboardInterrupt
    with pytest.raises(error):
        run_queue(
            tasks,
            ["GPU-a", "GPU-b"],
            tmp_path,
            lambda task: [sys.executable, worker, tmp_path, task],
            os.environ.copy(),
            dependencies={"fold_0/blood_lstm": ["fold_0/blood2d"]},
            on_state=on_state,
        )
    assert len(processes) == 2 and all(p.poll() is not None for p in processes)
    assert not (tmp_path / "fold_0-blood_lstm.started").exists()
    state = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert state["status"] == ("failed" if failure == "worker" else "interrupted")
    assert state["active_gpus"] == 0
    assert state["stages"]["fold_0/blood_lstm"]["status"] == "not_started"
