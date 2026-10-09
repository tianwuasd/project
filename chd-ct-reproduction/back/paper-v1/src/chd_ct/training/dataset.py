import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import aligned_label, blood_target, bounding_box, load_volume, normalize_ct, resize


class StageDataset(Dataset):
    """One-case cache avoids loading a clinical cohort into memory."""

    def __init__(self, rows, stage, config):
        self.rows, self.stage, self.config = rows, stage, config
        self.spec = config["stages"][stage]
        self.cache_key, self.cache = None, None
        self.index = []
        for i, row in enumerate(rows):
            if stage.startswith("init") and not row.get("initial_label"):
                raise ValueError(f"initial_label is required for {stage}")
            if stage.startswith("blood"):
                # Only shapes are retained, never an entire cohort of tensors.
                _, labels, _ = self._load(i)
                self.index.extend((i, z) for z in range(labels.shape[2]))
            else:
                self.index.append((i, None))

    def _load(self, i):
        if self.cache_key != i:
            row = self.rows[i]
            v = load_volume(row["image"])
            labels = aligned_label(row["label"], v)
            initial = aligned_label(row["initial_label"], v, 7) if self.stage.startswith("init") else None
            x = normalize_ct(v.data, self.config["window"])
            roi = (
                tuple(slice(0, n) for n in x.shape)
                if self.stage.startswith("crop")
                else bounding_box(labels, self.config["roi_margin"])
            )
            x, labels = x[roi], labels[roi]
            target = initial[roi] if initial is not None else labels
            if self.stage.startswith("blood"):
                target = blood_target(labels)
            self.cache = (x, target, roi)
            self.cache_key = i
        return self.cache

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, z = self.index[index]
        image, target, _ = self._load(i)
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
