"""Disk capacity changes must not invalidate cached tensors or frozen configs."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from chd_ct.imagechd import cpu_cache
from chd_ct.recovery import read_json

GIB = 1024**3


@pytest.fixture
def cache_case(tmp_path, monkeypatch):
    monkeypatch.delenv("CHD_CT_CACHE_CAPACITY_GIB", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "case.npz").write_bytes(b"unchanged-source")
    row = {"cache": "case.npz", "cache_shape": [2, 2, 2]}
    config = {"roi_margin": 0, "stages": {"crop64": {"size": 2}},
              "data_loading": {"cache_max_gib": 32}}
    monkeypatch.setattr(cpu_cache, "prepare_input", lambda *args: (
        np.zeros((2, 2, 2), np.float32), np.ones((2, 2, 2), np.uint8)))
    root = tmp_path / "inputs"
    root.mkdir()
    return root, source, row, config


def test_old_32_gib_is_no_longer_a_cap_and_config_is_unchanged(cache_case, monkeypatch):
    root, source, row, config = cache_case
    before = copy.deepcopy(config)
    monkeypatch.setattr(cpu_cache, "cache_used_bytes", lambda root: 40 * GIB)
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=60 * GIB))
    record = cpu_cache.cached_input(root, source, row, "crop64", config, max_gib=32)
    assert not record["reused"]
    assert config == before
    budget = read_json(root / "cache-budget.json")
    assert budget["limit_bytes"] == 80 * GIB
    assert budget["legacy_cache_max_gib"] == 32


def test_capacity_does_not_grow_as_the_cache_fills(tmp_path, monkeypatch):
    monkeypatch.delenv("CHD_CT_CACHE_CAPACITY_GIB", raising=False)
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=60 * GIB))
    a = cpu_cache.disk_budget(tmp_path, 40 * GIB, 2)
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=50 * GIB))
    b = cpu_cache.disk_budget(tmp_path, 50 * GIB, 2)
    assert a["limit_bytes"] == b["limit_bytes"] == 80 * GIB
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=20 * GIB))
    assert cpu_cache.disk_budget(tmp_path, 50 * GIB, 2)["limit_bytes"] == 56 * GIB


@pytest.mark.parametrize("free", [0, GIB, 2 * GIB])
def test_low_disk_stops_new_writes_but_verified_cache_can_be_reused(cache_case, monkeypatch, free):
    root, source, row, config = cache_case
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * GIB))
    first = cpu_cache.cached_input(root, source, row, "crop64", config)
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=free))
    assert cpu_cache.cached_input(root, source, row, "crop64", config)["reused"]
    changed = copy.deepcopy(config)
    changed["stages"]["crop64"]["size"] = 3
    with pytest.raises(ValueError, match="磁盘|80%"):
        cpu_cache.cached_input(root, source, row, "crop64", changed)
    assert np.load(first["paths"]["target"]).sum() == 8
    assert len(list(root.glob("*.npy"))) == 2


def test_known_allocation_caps_shared_disk_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("CHD_CT_CACHE_CAPACITY_GIB", "100")
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=20000 * GIB))
    budget = cpu_cache.disk_budget(tmp_path, 40 * GIB, 2)
    assert budget["limit_bytes"] == 80 * GIB
    assert budget["allocation_bytes"] == 100 * GIB


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "0", "-1"])
def test_invalid_allocation_does_not_silently_use_whole_shared_disk(tmp_path, monkeypatch, value):
    monkeypatch.setenv("CHD_CT_CACHE_CAPACITY_GIB", value)
    with pytest.raises(ValueError, match="CHD_CT_CACHE_CAPACITY_GIB"):
        cpu_cache.disk_budget(tmp_path, 0, 2)


def test_eighty_percent_rejection_preserves_existing_cache(cache_case, monkeypatch):
    root, source, row, config = cache_case
    monkeypatch.setattr(cpu_cache, "cache_used_bytes", lambda root: 80 * GIB)
    monkeypatch.setattr(cpu_cache.shutil, "disk_usage", lambda root: SimpleNamespace(free=20 * GIB))
    with pytest.raises(ValueError, match="80%"):
        cpu_cache.cached_input(root, source, row, "crop64", config)
    assert not list(root.glob("*.npy"))
