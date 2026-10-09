"""Voxel-grid operations reused from the paper workflow, with native seven labels."""

import numpy as np
import torch
from scipy import ndimage as ndi

BLOOD_IDS = (1, 2, 3, 4, 6, 7)


def resize(array, shape, labels=False):
    x = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)[None, None]
    mode = "nearest" if labels else ("trilinear" if array.ndim == 3 else "bilinear")
    result = torch.nn.functional.interpolate(
        x, size=tuple(shape), mode=mode, **({} if labels else {"align_corners": False})
    )[0, 0].numpy()
    return result.astype(np.int64) if labels else result


def bounding_box(mask, margin=3):
    points = np.where(mask)
    if not points[0].size:
        return tuple(slice(0, n) for n in mask.shape)
    return tuple(
        slice(max(0, int(p.min()) - margin), min(n, int(p.max()) + margin + 1))
        for p, n in zip(points, mask.shape)
    )


def blood_target(target):
    pool = np.isin(target, BLOOD_IDS)
    structure = ndi.generate_binary_structure(3, 1)
    interior = ndi.binary_erosion(pool, structure=structure)
    result = np.where(pool, np.where(interior, 1, 2), 0).astype(np.uint8)
    # Unknown structures cannot provide reliable blood/background boundary supervision.
    unknown = ndi.binary_dilation(target == 255, structure=structure)
    result[unknown] = 255
    return result
