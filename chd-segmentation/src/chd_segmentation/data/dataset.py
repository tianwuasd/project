from collections import OrderedDict
from time import perf_counter

from monai.data import Dataset
from monai.transforms import Compose
import numpy as np

from ..preprocessing.transforms import CaseTransform


class CaseDataset:
    def __init__(self, prepared, resources, with_label=True):
        self.rows = prepared["manifest"]["cases"]
        self.by_id = {row["case_id"]: index for index, row in enumerate(self.rows)}
        self.transform = CaseTransform(prepared, resources["reserve_ram_bytes"], with_label)
        self.dataset = Dataset(self.rows, transform=Compose([self.transform]))
        self.cache, self.cached_bytes = OrderedDict(), 0
        self.budget = resources["cache_bytes"]
        self.hits, self.misses, self.wait_seconds = 0, 0, 0.

    def get(self, name):
        started = perf_counter()
        if name in self.cache:
            self.transform.check_sources(self.rows[self.by_id[name]])
            result, size = self.cache.pop(name)
            self.cache[name] = (result, size)
            self.hits += 1
        else:
            result = self.dataset[self.by_id[name]]
            self.misses += 1
            size = sum(value.nbytes for value in result.values() if isinstance(value, np.ndarray))
            while self.cache and self.cached_bytes + size > self.budget:
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
