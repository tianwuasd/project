"""Exercise the cross-fold coordinator with real CPU stage training and fake GPU assignment."""

from pathlib import Path

import pytest
import test_chd_main

from chd_ct.imagechd import cross_fold_queue, stage_worker
from chd_ct.imagechd import train as trainer
from chd_ct.imagechd.config import STAGES
from chd_ct.imagechd.preprocess import prepare
from chd_ct.recovery import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def install_cpu_dispatch(monkeypatch, *, fail_after=None):
    seen = []
    monkeypatch.setattr(
        cross_fold_queue, "options", lambda ids, limit, *args: dict(gpu_ids=ids, gpu_limit=limit)
    )

    def dispatch(tasks, **kwargs):
        assert kwargs["gpu_limit"] == 2
        assert len(kwargs["lock_fds"]) == 4  # CV root and all three child locks
        ordered = sorted(tasks, key=lambda t: (STAGES.index(t.split("/")[1]), t))
        state = dict(
            status="running",
            sampled_at=1,
            gpu_limit=2,
            active_gpus=0,
            stages={t: {"status": "pending"} for t in tasks},
        )
        for task in ordered:
            for dependency in kwargs["dependencies"].get(task, []):
                assert state["stages"][dependency]["status"] == "complete"
            command = list(map(str, kwargs["command_for"](task)))
            key, stage = task.split("/")
            child = Path(command[command.index("--output") + 1])
            assert child.name == key
            frozen = read_json(child / "effective-config.json")
            assert frozen["fold"]["index"] == int(key[-1])
            seen.append((task, "--resume" in command))
            if fail_after is not None and len(seen) == fail_after:
                state["status"] = "interrupted"
                kwargs["on_state"](state)
                raise KeyboardInterrupt("synthetic cross-fold interruption")
            command.remove("--check-environment")
            state["stages"][task].update(status="running", gpu_uuid="GPU-test")
            kwargs["on_state"](state)
            assert stage_worker.main(command[3:]) == 0
            result = read_json(child / (stage + ".result.json"))
            state["stages"][task]["status"] = "complete"
            kwargs["on_complete"](task, result)
            kwargs["on_state"](state)
        state["status"] = "complete"
        kwargs["on_state"](state)

    monkeypatch.setattr(cross_fold_queue, "run_queue", dispatch)
    return seen


def test_cross_fold_coordinator_resumes_partial_folds_and_keeps_frozen_settings(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "models"
    prepare(test_chd_main.raw_chd.__wrapped__(tmp_path), cache, size=16)
    seen = install_cpu_dispatch(monkeypatch, fail_after=5)
    args = dict(mode="smoke", device="cpu", gpu_ids=["GPU-a", "GPU-b"], gpu_limit=2)
    config = ROOT / "configs/chd-cv-smoke.yaml"
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cache, output, config, **args)
    assert {task.split("/")[0] for task, _ in seen} == {"fold_0", "fold_1", "fold_2"}
    assert read_json(output / "report.json")["status"] == "interrupted"
    checkpoint = output / "fold_0/crop64.last.pt"
    saved = file_hash(checkpoint)
    seen = install_cpu_dispatch(monkeypatch)
    report = trainer.train(cache, output, config, resume=True, **args)
    assert report["status"] == "passed" and report["completed_folds"] == 3
    assert report["scheduling"] == "cross_fold" and report["current_folds"] == []
    assert len(seen) == 18 and all(resumed for _, resumed in seen)
    assert file_hash(checkpoint) == saved
    from chd_ct.imagechd.checkpoints import read_collection

    _, collection = read_collection(output, allow_smoke=True)
    assert len(collection["_members"]) == 3
    manifests = {p: p.read_bytes() for p in output.glob("fold_*/models.json")}
    seen = install_cpu_dispatch(monkeypatch)
    trainer.train(cache, output, config, resume=True, **args)
    assert not seen
    assert all(p.read_bytes() == value for p, value in manifests.items())


def test_legacy_serial_interruption_can_resume_with_global_queue(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "models"
    prepare(test_chd_main.raw_chd.__wrapped__(tmp_path), cache, size=16)
    original = trainer.train_stage

    def interrupt(directory, rows, stage, config, *args, **kwargs):
        if config["fold"]["index"] == 1 and stage == "all64":
            raise KeyboardInterrupt("legacy serial partial fold")
        return original(directory, rows, stage, config, *args, **kwargs)

    monkeypatch.setattr(trainer, "train_stage", interrupt)
    config = ROOT / "configs/chd-cv-smoke.yaml"
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cache, output, config, mode="smoke")
    preserved = (output / "fold_0/models.json").read_bytes()
    seen = install_cpu_dispatch(monkeypatch)
    result = trainer.train(
        cache,
        output,
        config,
        mode="smoke",
        device="cpu",
        gpu_ids=["GPU-a", "GPU-b"],
        gpu_limit=2,
        resume=True,
    )
    assert result["completed_folds"] == 3
    assert len(seen) == 12 and all(not t.startswith("fold_0/") for t, _ in seen)
    assert all(resumed == t.startswith("fold_1/") for t, resumed in seen)
    assert (output / "fold_0/models.json").read_bytes() == preserved
