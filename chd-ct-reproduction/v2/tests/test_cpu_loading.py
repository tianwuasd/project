"""Equivalent scientific inputs, reproducible training, bounded automatic tuning."""
# ruff: noqa: F811

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from test_chd_main import raw_chd  # noqa: F401

from chd_ct_v2.imagechd import autotune, cpu_cache, cpu_resources
from chd_ct_v2.imagechd import train as training
from chd_ct_v2.imagechd.config import STAGES, load_config
from chd_ct_v2.imagechd.cpu_resources import GIB, worker_limit
from chd_ct_v2.imagechd.dataset import StageDataset
from chd_ct_v2.imagechd.loading import settings
from chd_ct_v2.imagechd.preprocess import prepare
from chd_ct_v2.recovery import file_hash

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
    rows = [
        {**r, "predicted_roi": [[1, int(n) - 1] for n in r["cache_shape"]], "roi_sha256": "test-predicted"}
        for r in data["cases"][:2]
    ]
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


def test_v2_direct_configuration_never_runs_candidate_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(autotune, "trial", lambda *a: pytest.fail("V2 must not benchmark candidates"))
    config = configuration()
    before = copy.deepcopy(config)
    selected = autotune.resolve(config, tmp_path, [], tmp_path)
    assert config == before
    assert selected["stages"] == config["stages"]
    assert selected["data_loading"]["auto_tune"] is False
    assert not (tmp_path / "cpu-tuning.json").exists()


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


def test_v2_zero_and_unknown_capacity_are_distinct():
    from chd_ct_v2.panel_v2.resources import loader_budget

    config = configuration()
    rows = {"train": [{"cache_shape": [16] * 3}], "val": [{"cache_shape": [16] * 3}]}
    _, report = loader_budget(config, rows, "crop64", facts={"cpu_budget": None, "memory_budget_bytes": None})
    assert report["workers"] == 0 and report["main_threads"] == 1
    with pytest.raises(ValueError):
        loader_budget(config, rows, "crop64", facts={"cpu_budget": 0, "memory_budget_bytes": 10**12})


def test_v2_resume_resources_are_live_but_scientific_configuration_stays_fixed(tmp_path):
    from chd_ct_v2.panel_v2.resources import loader_budget

    config = configuration()
    rows = {"train": [{"cache_shape": [16] * 3}], "val": [{"cache_shape": [16] * 3}]}
    first = autotune.resolve(config, tmp_path, [], tmp_path)
    resumed = autotune.resolve(config, tmp_path, [], tmp_path, resume=True)
    assert first == resumed
    fast, a = loader_budget(
        first, rows, "crop64", facts={"cpu_budget": 16, "memory_budget_bytes": 32 * 2**30}
    )
    slow, b = loader_budget(
        resumed, rows, "crop64", facts={"cpu_budget": 2, "memory_budget_bytes": 4 * 2**30}
    )
    assert a["workers"] > b["workers"]
    assert fast["stages"] == slow["stages"] == config["stages"]
