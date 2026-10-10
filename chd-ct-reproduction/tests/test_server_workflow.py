"""Exercise server preferences, resource limits and real child-process scheduling."""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from chd_ct.quickstart.commands import add_task_arguments, build_command
from chd_ct.server.launcher import select_gpus
from chd_ct.server.settings import resolve_settings


def rows():
    return [
        dict(index=str(i), uuid=f"GPU-{i}", name="3090", free=24000 - i, util=0, busy=False) for i in range(4)
    ]


def test_multigpu_selection_never_exceeds_allocation():
    assert [r["index"] for r in select_gpus(rows(), 2, visible="1,3")] == ["1", "3"]
    with pytest.raises(ValueError):
        select_gpus(rows(), 3, visible="1,3")
    with pytest.raises(ValueError):
        select_gpus(rows(), 2, "0,1", visible="1,3")
    with pytest.raises(ValueError, match="重复"):
        select_gpus(rows(), 2, "0,GPU-0")
    for value in (0, -1, 6, True, 1.5):
        with pytest.raises(ValueError):
            select_gpus(rows(), value)
    devices = rows()
    devices[0]["busy"] = True
    assert [r["index"] for r in select_gpus(devices, 2)] == ["1", "2"]


def arguments(tmp_path, *options):
    parser = argparse.ArgumentParser()
    add_task_arguments(parser)
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--gpu")
    parser.add_argument("--runtime")
    parser.add_argument("--results")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(["--non-interactive", *options])
    profile = dict(
        runtime=str(tmp_path / "runtime"),
        results=str(tmp_path / "results"),
        prepared=str(tmp_path / "prepared"),
        dataset=str(tmp_path / "raw"),
        source_updated="2026-09-26",
        gpu_count=1,
        gpu="auto",
    )
    return args, profile


def test_settings_default_saved_and_explicit_precedence(tmp_path):
    args, profile = arguments(tmp_path, "--task", "train")
    runtime, _, settings_file, settings = resolve_settings(args, profile, False)
    assert args.dataset is None and args.gpu_count == 1  # Training only consumes prepared data.
    runtime.mkdir()
    settings.update(dataset=str(tmp_path / "saved_raw"), gpu_count=3, last_models=str(tmp_path / "models"))
    settings_file.write_text(json.dumps(settings))
    args, _ = arguments(tmp_path, "--task", "train", "--gpu-count", "2")
    resolve_settings(args, profile, False)
    assert args.dataset is None and args.gpu_count == 2
    args, _ = arguments(tmp_path, "--task", "test")
    resolve_settings(args, profile, False)
    assert args.models.endswith("models")
    command = list(map(str, build_command(args, tmp_path, sys.executable, tmp_path / "out", "cpu")))
    assert command[2] == "chd_ct.imagechd.test"
    assert "--mode" not in command and "--input" not in command


def test_server_checks_every_gpu_and_dispatches_only_train(tmp_path, monkeypatch):
    import signal
    from types import SimpleNamespace

    from chd_ct.server import imagechd

    args, profile = arguments(tmp_path, "--task", "train", "--gpu-count", "2", "--device", "cuda")
    Path(profile["prepared"]).mkdir()
    (Path(profile["prepared"]) / "dataset.json").write_text("{}")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(imagechd, "query_gpus", rows)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    calls = []
    monkeypatch.setattr(
        imagechd, "run_logged", lambda cmd, env, log: calls.append((list(map(str, cmd)), env.copy())) or 0
    )
    assert imagechd.run(args, profile) == 0
    assert len(calls) == 3
    assert [e["CUDA_VISIBLE_DEVICES"] for _, e in calls[:2]] == ["GPU-1", "GPU-3"]
    command, env = calls[2]
    assert command[2] == "chd_ct.imagechd.train"
    assert command[command.index("--gpu-ids") + 1] == "GPU-1,GPU-3"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-1,GPU-3"


def test_queue_obeys_dependency_and_failure_cleanup(tmp_path):
    from chd_ct.imagechd.stage_queue import run_queue

    worker = tmp_path / "worker.py"
    worker.write_text("""import json, os, sys, time
from pathlib import Path
root, stage, fail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
(root / (stage+'.started')).write_text(os.environ['CUDA_VISIBLE_DEVICES'])
if stage=='blood_lstm':
    assert (root/'blood2d.result.json').exists()
if fail=='yes':
    time.sleep(0.3)
    if stage=='blood2d': sys.exit(9)
    time.sleep(60)
else:
    time.sleep(0.1)
(root / (stage+'.result.json')).write_text(json.dumps({'file':stage+'.pt'}))
""")
    output = tmp_path / "success"
    output.mkdir()
    stages = ["blood_lstm", "blood2d", "all64"]

    def command(stage):
        return [sys.executable, worker, output, stage, "no"]

    result = run_queue(stages, ["GPU-a", "GPU-b"], output, command, os.environ.copy())
    assert set(result) == set(stages)
    assert (output / "blood2d.started").read_text() == "GPU-a"
    assert (output / "all64.started").read_text() == "GPU-b"
    queue = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    assert queue["status"] == "complete"
    output = tmp_path / "failure"
    output.mkdir()

    def command(stage):
        return [sys.executable, worker, output, stage, "yes"]

    with pytest.raises(RuntimeError, match="blood2d"):
        run_queue(stages, ["GPU-a", "GPU-b"], output, command, os.environ.copy())
    queue = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    assert queue["status"] == "failed"
    assert queue["stages"]["all64"]["status"] == "interrupted"
    assert not (output / "blood_lstm.started").exists()
    assert not (output / "all64.result.json").exists()


def test_handoff_waits_for_stale_utilization_but_never_for_external_process(monkeypatch):
    from chd_ct.imagechd import stage_queue
    from chd_ct.server import launcher

    samples = iter(
        [
            [dict(index="0", uuid="GPU-0", name="3090", free=23000, util=60, busy=False)],
            [dict(index="0", uuid="GPU-0", name="3090", free=23000, util=0, busy=False)],
        ]
    )
    monkeypatch.setattr(launcher, "query_gpus", lambda: next(samples))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0")
    sleeps = []
    monkeypatch.setattr(stage_queue.time, "sleep", sleeps.append)
    stage_queue.check_gpu_handoff("GPU-0", used=True)
    assert sleeps == [1]
    busy = dict(index="0", uuid="GPU-0", name="3090", free=23000, util=0, busy=True)
    monkeypatch.setattr(launcher, "query_gpus", lambda: [busy])
    with pytest.raises(ValueError, match="不再空闲"):
        stage_queue.check_gpu_handoff("GPU-0", used=True)
    assert sleeps == [1]


def test_reset_settings_uses_profile_without_removing_cache(tmp_path):
    args, profile = arguments(tmp_path, "--task", "train")
    runtime, _, settings_file, settings = resolve_settings(args, profile, False)
    runtime.mkdir()
    settings["gpu_count"] = 4
    settings_file.write_text(json.dumps(settings))
    args, _ = arguments(tmp_path, "--task", "train")
    args.reset_settings = True
    resolve_settings(args, profile, False)
    assert args.gpu_count == 1 and settings_file.is_file()


@pytest.mark.parametrize(
    "task",
    [
        "diagnosis-labels",
        "diagnosis-features",
        "diagnosis-train",
        "diagnose",
        "diagnosis-evaluate",
        "diagnosis-demo",
    ],
)
def test_diagnosis_dispatch_is_independent_and_cpu_only(task, tmp_path, monkeypatch):
    import signal
    from types import SimpleNamespace

    from chd_ct.quickstart.diagnosis import OUTPUT_KEYS
    from chd_ct.server import imagechd

    args, profile = arguments(
        tmp_path,
        "--task",
        task,
        "--features",
        str(tmp_path / "features"),
        "--diagnosis-labels",
        str(tmp_path / "labels"),
        "--classifier",
        str(tmp_path / "classifier"),
        "--predictions",
        str(tmp_path / "segmentations"),
        "--diagnoses",
        str(tmp_path / "diagnoses"),
    )
    # No prepared files or available GPUs; launcher must not call segmentation or GPU selection.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(imagechd, "query_gpus", lambda: pytest.fail("诊断不应检查或占用显卡"))
    calls = []
    monkeypatch.setattr(
        imagechd, "run_logged", lambda cmd, env, log: calls.append((list(map(str, cmd)), env)) or 0
    )
    assert imagechd.run(args, profile) == 0
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("--device") + 1] == "cpu"
    command, env = calls[1]
    assert command[2] == "chd_ct.diagnosis.cli"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    if task == "diagnosis-labels":
        assert command[command.index("--blank-policy") + 1] == "negative"
        assert command[command.index("--input") + 1].endswith("imageCHD_dataset_info.xlsx")
    settings = json.loads((Path(profile["runtime"]) / ".server-settings.json").read_text())
    if task in OUTPUT_KEYS:
        assert Path(settings[OUTPUT_KEYS[task]]).name == task
    assert "last_models" not in settings


def test_explicit_diagnosis_dataset_overrides_saved_source(tmp_path):
    args, profile = arguments(tmp_path, "--task", "diagnosis-labels")
    runtime = Path(profile["runtime"])
    runtime.mkdir()
    (runtime / ".server-settings.json").write_text(
        json.dumps({"diagnosis_source": str(tmp_path / "old.xlsx"), "gpu_count": 3, "gpu": "0,1,2"})
    )
    args.dataset = str(tmp_path / "new-dataset")
    _, _, _, saved = resolve_settings(args, profile, False)
    assert args.diagnosis_source == str(tmp_path / "new-dataset/imageCHD_dataset_info.xlsx")
    assert saved["gpu_count"] == 3 and saved["gpu"] == "0,1,2"
