"""Stage-specific tensors; optional mmap cache removes repeated volume work."""

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import load_case
from .geometry import blood_target, bounding_box, resize
from .loading import settings


class StageDataset(Dataset):
    def __init__(self, directory, rows, stage, config, cache_root=None):
        self.directory, self.rows, self.stage, self.config = directory, rows, stage, config
        self.spec = config["stages"][stage]
        self.key, self.cached = None, None
        self.records, self.maps = [], OrderedDict()
        self.index = []
        options = settings(config)
        for i, row in enumerate(rows):
            if cache_root is not None and options["cache"] == "mmap":
                from .cpu_cache import cached_input

                record = cached_input(
                    cache_root,
                    directory,
                    row,
                    stage,
                    config,
                    options["cache_max_gib"],
                    options["cache_reserve_gib"],
                )
                self.records.append(record)
                depth = record["shape"][0] if stage.startswith("blood") else None
                if depth is not None:
                    self.index.extend((i, z) for z in range(depth))
                else:
                    self.index.append((i, None))
                print(
                    f"[input-cache] {stage} {i + 1}/{len(rows)} {'reused' if record['reused'] else 'built'}",
                    flush=True,
                )
            elif stage.startswith("blood"):
                image, _ = self._load(i)
                self.index.extend((i, z) for z in range(image.shape[2]))
            else:
                self.index.append((i, None))

    def __getstate__(self):
        # Spawn only paths/index metadata, never copies of volumes or mappings.
        state = dict(self.__dict__)
        state.update(maps=OrderedDict(), key=None, cached=None)
        return state

    def _mapped(self, i):
        if i not in self.maps:
            paths = self.records[i]["paths"]
            self.maps[i] = tuple(
                np.load(paths[k], mmap_mode="r", allow_pickle=False) for k in ("image", "target")
            )
            if len(self.maps) > 8:
                self.maps.popitem(last=False)
        self.maps.move_to_end(i)
        return self.maps[i]

    def _load(self, i):
        if self.key != i:
            image, target = load_case(self.directory, self.rows[i])
            if not self.stage.startswith("crop"):
                roi = bounding_box((target > 0) & (target < 8), self.config["roi_margin"])
                image, target = image[roi], target[roi]
            if self.stage.startswith("blood"):
                target = blood_target(target)
            self.cached = (image, target)
            self.key = i
        return self.cached

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, z = self.index[index]
        if self.records:
            image, target = self._mapped(i)
            size = self.spec["size"]
            if z is None:
                x, y = image[None], target
            else:
                y = resize(target[z].copy(), (size, size), labels=True)
                if self.stage == "blood_lstm":
                    radius = self.spec["sequence"] // 2
                    indices = np.clip(np.arange(z - radius, z + radius + 1), 0, image.shape[0] - 1)
                    x = np.stack([resize(image[j].copy(), (size, size))[None] for j in indices])
                else:
                    x = resize(image[z].copy(), (size, size))[None]
            return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).long()
        image, target = self._load(i)
        size = self.spec["size"]
        if z is None:
            x = resize(image, (size,) * 3)[None]
            y = resize(target, (size,) * 3, labels=True)
        else:
            y = resize(target[:, :, z], (size, size), labels=True)
            if self.stage == "blood_lstm":
                radius = self.spec["sequence"] // 2
                indices = np.clip(np.arange(z - radius, z + radius + 1), 0, image.shape[2] - 1)
                x = np.stack([resize(image[:, :, j], (size, size))[None] for j in indices])
            else:
                x = resize(image[:, :, z], (size, size))[None]
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).long()
