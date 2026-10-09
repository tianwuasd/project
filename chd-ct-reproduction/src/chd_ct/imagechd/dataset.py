"""Stage-specific tensors from prepared caches; only training uses true ROIs."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import load_case
from .geometry import blood_target, bounding_box, resize


class StageDataset(Dataset):
    def __init__(self, directory, rows, stage, config):
        self.directory, self.rows, self.stage, self.config = directory, rows, stage, config
        self.spec = config["stages"][stage]
        self.key, self.cached = None, None
        self.index = []
        for i, row in enumerate(rows):
            if stage.startswith("blood"):
                image, _ = self._load(i)
                self.index.extend((i, z) for z in range(image.shape[2]))
            else:
                self.index.append((i, None))

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
