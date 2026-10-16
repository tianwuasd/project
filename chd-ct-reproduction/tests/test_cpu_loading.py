"""Equivalent scientific inputs, reproducible training, bounded automatic tuning."""
# ruff: noqa: F811

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from test_chd_main import raw_chd  # noqa: F401

from chd_ct.imagechd import autotune, cpu_cache, cpu_resources
from chd_ct.imagechd import train as training
from chd_ct.imagechd.config import STAGES, load_config
from chd_ct.imagechd.cpu_resources import GIB, worker_limit
from chd_ct.imagechd.dataset import StageDataset
from chd_ct.imagechd.loading import settings
from chd_ct.imagechd.preprocess import prepare
from chd_ct.recovery import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    config = load_config(ROOT / "configs/chd-smoke.yaml")
    config["device"] = "cpu"
    config["data_loading"] = settings({}) | {"cache": "mmap"}
    return config


@pytest.mark.parametrize("stage", STAGES)
def test_mmap_inputs_match_original_and_do_not_repeat_volume_work(raw_chd, tmp_path, monkeypatch, stage):
    directory = tmp_path / "prepared"
    data = prepare(raw_chd, directory, size=0)
    config = configuration()
    rows = data["cases"][:2]
    original = StageDataset(directory, rows, stage, config)
    cached = StageDataset(directory, rows, stage, config, tmp_path / "inputs")
    assert original.index == cached.index
    order = [0, len(original) - 1, 1, len(original) // 2, 0]
    expected = [original[i] for i in order]

    def forbidden(*args):
        raise AssertionError("unexpected whole-volume load/morphology")

    monkeypatch.setattr(cpu_cache, "prepare_input", forbidden)
    reused = StageDataset(directory, rows, stage, config, tmp_path / "inputs")
    for ds in (cached, reused):
        for i, pair in zip(order, expected):
            assert all(torch.equal(a, b) for a, b in zip(pair, ds[i]))
    assert not cached.__getstate__()["maps"]


def test_cache_identity_integrity_and_budget(raw_chd, tmp_path):
    directory = tmp_path / "prepared"
    row = prepare(raw_chd, directory, size=16)["cases"][0]
    config = configuration()
    inputs = tmp_path / "inputs"
    record = cpu_cache.cached_input(inputs, directory, row, "crop64", config)
    altered = copy.deepcopy(config)
    altered["stages"]["crop64"]["size"] = 8
    other = cpu_cache.cached_input(inputs, directory, row, "crop64", altered)
    assert record["paths"] != other["paths"]
    record["paths"]["image"].write_bytes(b"damaged")
    with pytest.raises(ValueError, match="校验失败"):
        cpu_cache.cached_input(inputs, directory, row, "crop64", config)
    before = file_hash(directory / row["cache"])
    with pytest.raises(ValueError, match="cache_max_gib"):
        cpu_cache.cached_input(tmp_path / "tiny", directory, row, "crop64", config, max_gib=1e-9)
    assert file_hash(directory / row["cache"]) == before


@pytest.mark.parametrize("workers", [0, 2])
def test_parallel_training_and_resume_preserve_optimizer_rng(raw_chd, tmp_path, monkeypatch, workers):
    directory = tmp_path / "prepared"
    data = prepare(raw_chd, directory, size=16)
    config = configuration()
    config.update(augment=True)
    config["stages"]["crop64"]["epochs"] = 2
    config["data_loading"].update(workers=workers, cache_dir=str(tmp_path / "inputs"))
    rows = training.training_rows(data)
    baseline = copy.deepcopy(config)
    baseline.pop("data_loading")
    paths = [tmp_path / name for name in ("legacy", "optimized", "resume")]
    for path in paths:
        path.mkdir()
    digest = file_hash(directory / "dataset.json")
    for path, cfg in zip(paths[:2], [baseline, config]):
        training.train_stage(directory, rows, "crop64", cfg, path, "train", digest)
    original_write = training.write_json

    def stop(path, value):
        original_write(path, value)
        if str(path).endswith("history.json") and len(value) == 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(training, "write_json", stop)
    with pytest.raises(KeyboardInterrupt):
        training.train_stage(directory, rows, "crop64", config, paths[2], "train", digest)
    monkeypatch.setattr(training, "write_json", original_write)
    training.train_stage(directory, rows, "crop64", config, paths[2], "train", digest, resume=True)
    expected = torch.load(paths[0] / "crop64.last.pt", weights_only=True)
    for path in paths[1:]:
        actual = torch.load(path / "crop64.last.pt", weights_only=True)
        assert expected["history"] == actual["history"]
        assert torch.equal(expected["torch_rng"], actual["torch_rng"])
        assert np.array_equal(expected["numpy_rng"][1], actual["numpy_rng"][1])
        assert all(torch.equal(v, actual["state_dict"][k]) for k, v in expected["state_dict"].items())
        for key, value in expected["optimizer"]["state"].items():
            assert all(torch.equal(v, actual["optimizer"]["state"][key][k]) for k, v in value.items())


def test_worker_budget_respects_concurrency_and_shared_memory():
    config = configuration()
    hardware = dict(cpus=16, memory_available=32 * GIB, shared_available=16 * GIB, concurrency=2)
    assert worker_limit(config, hardware, "blood2d") == 3
    assert worker_limit(config, {**hardware, "cpus": 2}, "blood2d") == 0
    assert worker_limit(config, {**hardware, "shared_available": 1}, "blood2d") == 0


def test_cgroup_combined_mount_and_parent_quota(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    combined = root / "cpu,cpuacct"
    child = combined / "user/job"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("2:cpu,cpuacct:/user/job\n")
    (combined / "cpu.cfs_quota_us").write_text("300000")
    (combined / "cpu.cfs_period_us").write_text("100000")
    folders = cpu_resources.cgroup_dirs(root, membership)
    assert combined in folders and child in folders
    monkeypatch.setattr(cpu_resources, "cgroup_dirs", lambda: folders)
    monkeypatch.setattr(cpu_resources.os, "sched_getaffinity", lambda _: set(range(256)), raising=False)
    assert cpu_resources.resources(2)["cpus"] == 3


def test_cgroup_v2_parent_memory_and_cpu_limits(tmp_path, monkeypatch):
    (tmp_path / "cpu.max").write_text("200000 100000")
    (tmp_path / "memory.max").write_text(str(GIB))
    (tmp_path / "memory.current").write_text(str(GIB // 2))
    monkeypatch.setattr(cpu_resources, "cgroup_dirs", lambda: [tmp_path])
    monkeypatch.setattr(cpu_resources.os, "sched_getaffinity", lambda _: set(range(256)), raising=False)
    limits = cpu_resources.resources(1)
    assert limits["cpus"] == 2
    assert limits["memory_available"] == GIB // 2


def test_automatic_tuning_reuses_real_sizes_and_freezes_recommendation(raw_chd, tmp_path, monkeypatch):
    directory = tmp_path / "prepared"
    data = prepare(raw_chd, directory, size=16)
    config = configuration()
    config["stages"]["crop128"]["size"] = 32
    config["data_loading"].update(auto_tune=True, workers=4)
    rows = training.training_rows(data)["train"]
    hardware = dict(
        cpus=32, memory_available=32 * GIB, shared_available=16 * GIB, concurrency=2, host="fixture"
    )
    monkeypatch.setattr(autotune, "resources", lambda count: hardware)
    calls = []

    def trial(request, folder, seconds):
        calls.append(request)
        workers = request["config"]["data_loading"]["stage_workers"][request["stage"]]
        return dict(status="passed", seconds_per_sample={0: 3, 4: 1, 7: 0.98, 8: 0.98}.get(workers, 1))

    monkeypatch.setattr(autotune, "trial", trial)
    outputs = [tmp_path / s for s in ("smoke", "preflight", "train")]
    for output in outputs:
        output.mkdir()
        selected = autotune.resolve(config, directory, rows, output, 2)
        assert selected["data_loading"]["auto_tune"] is False
        assert selected["data_loading"]["stage_workers"]["crop128"] == 4
        report = read_json(output / "cpu-tuning.json")
        assert report["groups"]["crop128"]["input_size"] == 32
        assert report["training_cases"] == [r["case_id"] for r in rows[:2]]
    assert len(calls) == 9
    assert config["data_loading"]["auto_tune"] is True
    assert all(r["split"] == "train" for request in calls for r in request["rows"])
    monkeypatch.setattr(autotune, "resources", lambda _: pytest.fail("resume must not retune"))
    assert autotune.resolve(config, directory, rows, outputs[-1], 2, True) == selected
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="来源改变"):
        autotune.resolve(changed, directory, rows, outputs[-1], 2, True)


def test_failed_candidates_have_safe_fallback():
    assert autotune.choose([dict(status="failed", workers=4)], 4)[0] == 0


@pytest.mark.parametrize("cancel", [True, False])
def test_trial_cancellation_during_launch_and_timeout_clean_up(tmp_path, monkeypatch, cancel):
    import signal
    import subprocess

    previous = signal.getsignal(signal.SIGTERM)
    stopped = []

    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("fixture", timeout)

    process = Process()

    def launch(*args, **kwargs):
        if cancel:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return process

    monkeypatch.setattr(autotune.subprocess, "Popen", launch)
    monkeypatch.setattr(autotune, "terminate", stopped.append)
    if cancel:
        with pytest.raises(KeyboardInterrupt, match="取消"):
            autotune.trial({}, tmp_path / "trial", 1)
    else:
        assert autotune.trial({}, tmp_path / "trial", 0.01) == {"status": "timeout"}
    assert stopped == [process]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_actual_spawned_calibration(raw_chd, tmp_path):
    directory = tmp_path / "prepared"
    data = prepare(raw_chd, directory, size=16)
    config = configuration()
    config["data_loading"]["workers"] = 2
    request = dict(
        config=config,
        stage="blood_lstm",
        directory=str(directory),
        rows=training.training_rows(data)["train"][:2],
        cache_root=str(tmp_path / "inputs"),
    )
    result = autotune.trial(request, tmp_path / "trial", 45)
    assert result["status"] == "passed"
    assert result["samples"] == 12
    assert result["seconds_per_sample"] > 0


@pytest.mark.parametrize("folds", [1, 3])
def test_smoke_automatically_calibrates_then_freezes_across_folds_and_resume(
    raw_chd, tmp_path, monkeypatch, folds
):
    import json

    directory = tmp_path / "prepared"
    prepare(raw_chd, directory, size=16)
    config = configuration()
    config["cross_validation"]["folds"] = folds
    config["stages"]["crop128"]["size"] = 32
    config["data_loading"].update(auto_tune=True, workers=4)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    hardware = dict(cpus=2, memory_available=2 * GIB, shared_available=GIB, concurrency=1, host="fixture")
    monkeypatch.setattr(autotune, "resources", lambda count: hardware)
    output = tmp_path / "models"
    result = training.train(directory, output, source, "smoke")
    assert result["status"] == "passed"
    recommendation = read_json(output / "cpu-tuning.json")
    assert recommendation["groups"]["crop128"]["input_size"] == 32
    assert all(g["trials"][0]["status"] == "passed" for g in recommendation["groups"].values())
    frozen = read_json(output / "effective-config.json")
    assert frozen["data_loading"] == recommendation["selected"]
    assert frozen["stages"]["crop128"]["size"] == 16
    if folds == 3:
        for i in range(3):
            assert (
                read_json(output / f"fold_{i}" / "effective-config.json")["data_loading"]
                == frozen["data_loading"]
            )
    monkeypatch.setattr(autotune, "trial", lambda *a: pytest.fail("resume must reuse frozen recommendation"))
    assert training.train(directory, output, source, "smoke", resume=True)["status"] == "passed"
