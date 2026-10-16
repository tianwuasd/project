"""CPU-only spawned workers, separate RNG streams and bounded prefetch."""

import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

DEFAULTS = {
    "workers": 0,
    "worker_threads": 1,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "pin_memory": True,
    "cache": "none",
    "cache_dir": None,
    "cache_max_gib": 32,
    "cache_reserve_gib": 2,
    "progress_seconds": 30,
    "auto_tune": False,
    "tune_seconds": 180,
    "stage_workers": {},
}


def settings(config):
    return {**DEFAULTS, **config.get("data_loading", {})}


def validate_settings(config):
    value = config.get("data_loading", {})
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("data_loading 配置或字段无效")
    value = settings(config)
    for name, lower, upper in (("workers", 0, 64), ("worker_threads", 1, 16), ("prefetch_factor", 1, 8)):
        if type(value[name]) is not int or not lower <= value[name] <= upper:
            raise ValueError("data_loading." + name + " 超出允许范围")
    for name in ("persistent_workers", "pin_memory", "auto_tune"):
        if type(value[name]) is not bool:
            raise ValueError("data_loading." + name + " 必须为布尔值")
    for name in ("cache_max_gib", "cache_reserve_gib", "progress_seconds", "tune_seconds"):
        if type(value[name]) not in (int, float) or not np.isfinite(value[name]) or value[name] <= 0:
            raise ValueError("data_loading." + name + " 需要有限正数")
    if value["cache"] not in {"none", "mmap"}:
        raise ValueError("data_loading.cache 需要 none/mmap")
    if value["cache_dir"] is not None and (not isinstance(value["cache_dir"], str) or not value["cache_dir"]):
        raise ValueError("data_loading.cache_dir 需要目录路径或 null")
    overrides = value["stage_workers"]
    if not isinstance(overrides, dict) or any(
        k not in config["stages"] or type(v) is not int or not 0 <= v <= 64 for k, v in overrides.items()
    ):
        raise ValueError("data_loading.stage_workers 需要有效阶段及 0—64 的进程数")


def worker_init(worker_id, threads):
    torch.set_num_threads(threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, config, stage, split, seed):
    options = settings(config)
    workers = options["stage_workers"].get(stage, options["workers"])
    # Match legacy DataLoader's use of the main RNG for its base seed and
    # RandomSampler, but isolate worker lifetime/prefetch from augmentation RNG.
    if "data_loading" not in config:
        return DataLoader(dataset, batch_size=config["stages"][stage]["batch"], shuffle=split == "train")
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset) if split == "train" else SequentialSampler(dataset)
    kwargs = {}
    if workers:
        kwargs.update(
            multiprocessing_context="spawn",
            worker_init_fn=partial(worker_init, threads=options["worker_threads"]),
            persistent_workers=options["persistent_workers"],
            prefetch_factor=options["prefetch_factor"],
            timeout=120,
        )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config["stages"][stage]["batch"],
        num_workers=workers,
        generator=generator,
        pin_memory=options["pin_memory"] and config.get("device") == "cuda",
        **kwargs,
    )


def start_epoch(loader, optimized):
    if optimized:
        # Preserve the legacy single-process base-seed RNG draw once per split,
        # including persistent-worker epochs and epoch-boundary resume.
        torch.empty((), dtype=torch.int64).random_()
    iterator = iter(loader)
    loader._chd_iterator = iterator
    return iterator


def close_loader(loader):
    iterator = getattr(loader, "_chd_iterator", None) or getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    loader._iterator = None
    loader._chd_iterator = None
