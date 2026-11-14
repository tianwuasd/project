from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from time import perf_counter

import numpy as np
import torch

from .dataset import CaseDataset, sample_patch
from ..training.resources import register_member

_dataset = None
_gate = None


def initialize(prepared, config, resources, gate, owner):
    global _dataset, _gate
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    register_member(config, owner)
    _dataset, _gate = CaseDataset(prepared, resources), gate


def patch_task(request):
    names, settings, phase, attempt, slot = request
    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], phase, attempt, slot]))
    name = names[int(rng.integers(len(names)))]
    began = perf_counter()
    with _gate:
        entry = _dataset.get(name)
        pair = sample_patch(entry, settings["patch"], rng, settings["foreground_fraction"] if phase == 0 else 0.)
    return pair, entry["unknown_classes"], name, perf_counter() - began


class BatchSupply:
    def __init__(self, prepared, config, slots, workers, cache_bytes, gate, owner):
        self.prepared, self.config, self.slots = prepared, config, slots
        self.workers, self.cache_bytes, self.gate = workers, cache_bytes, gate
        self.owner = owner
        self.pool = None
        self.local = None
        self.queue = deque()

    def __enter__(self):
        resources = {**self.config["resources"], "cache_bytes": self.cache_bytes}
        if self.workers:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                initializer=initialize, initargs=(self.prepared, self.config, resources, self.gate, self.owner))
        else:
            self.local = CaseDataset(self.prepared, resources)
        return self

    def submit(self, names, phase, attempt):
        requests = [(names, self.config["training"], phase, attempt, slot) for slot in self.slots]
        if self.pool:
            self.queue.append([self.pool.submit(patch_task, request) for request in requests])
        else:
            self.queue.append(requests)

    def take(self, device):
        requests = self.queue.popleft()
        if self.pool:
            results = [future.result() for future in requests]
        else:
            results = []
            for names, settings, phase, attempt, slot in requests:
                rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], phase, attempt, slot]))
                name = names[int(rng.integers(len(names)))]
                began = perf_counter()
                with self.gate:
                    entry = self.local.get(name)
                    pair = sample_patch(entry, settings["patch"], rng, settings["foreground_fraction"] if phase == 0 else 0.)
                results.append((pair, entry["unknown_classes"], name, perf_counter() - began))
        tensors = [torch.from_numpy(np.stack(values)) for values in zip(*(r[0] for r in results))]
        if device.type == "cuda":
            tensors = [value.pin_memory().to(device, non_blocking=True) for value in tensors]
        return {"image": tensors[0], "label": tensors[1], "unknown_classes": [r[1] for r in results],
                "case_ids": [r[2] for r in results], "supply_seconds": sum(r[3] for r in results)}

    def __exit__(self, *_):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.local:
            self.local.clear()
        self.queue.clear()
