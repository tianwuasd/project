from collections import OrderedDict
from time import perf_counter

from monai.data import Dataset
from monai.transforms import Compose
import numpy as np

from ..preprocessing.offline_store import OfflineTransform, validate_storage
from ..preprocessing.inputs import resolve_preprocessed
from ..preprocessing.arrays import CompressedArray


def array_bytes(value):
    if isinstance(value, CompressedArray):
        return 64 * 1024
    if isinstance(value, np.memmap):
        # 映射文件的逻辑尺寸独立于驻留页；实际物理占用由MemoryGuard持续监控。
        return min(value.nbytes, 64 * 1024)
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(array_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(array_bytes(item) for item in value)
    return 0


class CaseDataset:
    def __init__(self, prepared, resources, with_label=True, log_dir=None, logging=None):
        self.rows = prepared["manifest"]["cases"]
        self.by_id = {row["case_id"]: index for index, row in enumerate(self.rows)}
        self.source = resolve_preprocessed(prepared['storage']['root'], dataset=prepared['dataset'],
            case_ids=list(self.by_id),storage=prepared['storage'])
        self.with_label = with_label
        self.cache, self.cached_bytes = OrderedDict(), 0
        self.budget = resources["cache_bytes"]
        self.case_limit = resources.get("cache_entries", 16)
        self.hits, self.misses, self.wait_seconds = 0, 0, 0.

    def get(self, name):
        started = perf_counter()
        if name in self.cache:
            result, size = self.cache.pop(name)
            self.cache[name] = (result, size)
            self.hits += 1
        else:
            result = self.source.open(name, with_label=self.with_label)
            self.misses += 1
            size = array_bytes(result)
            while self.cache and (self.cached_bytes + size > self.budget or len(self.cache) >= self.case_limit):
                _, (_, removed) = self.cache.popitem(last=False)
                self.cached_bytes -= removed
            if size <= self.budget:
                self.cache[name] = (result, size)
                self.cached_bytes += size
        self.wait_seconds += perf_counter() - started
        return result

    def clear(self):
        self.cache.clear()
        self.cached_bytes = 0


def sample_patch(entry, patch, rng, foreground_fraction):
    shape, patch = np.array(entry["image"].shape[1:]), np.asarray(patch)
    extent = np.minimum(shape, patch)
    start = rng.integers(0, shape - extent + 1)
    locations = entry["properties"]["class_locations"]
    available = [values for key, values in locations.items() if isinstance(key, int) and key > 0 and len(values)]
    if available and rng.random() < foreground_fraction:
        candidates = available[rng.integers(len(available))]
        point = candidates[rng.integers(len(candidates))][1:]
        start = np.clip(point - extent // 2, 0, shape - extent)
    region = tuple(slice(int(begin), int(end)) for begin, end in zip(start, start + extent))
    padding = tuple((0, int(size)) for size in patch - extent)
    image = np.pad(entry["image"][(slice(None), *region)], ((0, 0), *padding))
    label = np.pad(entry["label"][region], padding, constant_values=-1)
    return np.array(image, copy=True), np.array(label, dtype=np.int64, copy=True)
