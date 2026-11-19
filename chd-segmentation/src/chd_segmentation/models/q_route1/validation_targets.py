import math

import numpy as np
import torch
import torch.nn.functional as F

from ..stunet.streaming import tile_extent, tiles
from .targets import build_heatmap_targets


def localizer_statistics(logits, entry, dataset, workspace):
    from .adapter import allowed

    stats = np.zeros((3, 6), dtype=np.float64)
    spacing = np.asarray(entry['processed_spacing'], dtype=float)
    halo = (np.ceil(8 / spacing / 4).astype(int) + 1) * 4
    voxels = max(1, workspace.block_bytes // (64 * 512))
    total = math.prod(math.ceil(n / b) for n, b in zip(logits.shape[1:], tile_extent(logits.shape[1:], voxels)))
    for index, section in enumerate(tiles(logits.shape[1:], voxels), 1):
        with workspace.boundary():
            starts = [t.start*4-int(h) for t, h in zip(section, halo)]
            ends = [t.stop*4+int(h) for t, h in zip(section, halo)]
            axes = [np.arange(a, b) for a, b in zip(starts, ends)]
            indices = np.ix_(*(a.clip(0, n-1) for a, n in zip(axes, entry['label'].shape)))
            raw = np.array(entry['label'][indices], copy=True)
            for axis, (positions, n) in enumerate(zip(axes, entry['label'].shape)):
                selection = [slice(None)] * 3
                selection[axis] = (positions < 0) | (positions >= n)
                raw[tuple(selection)] = -1
            heat, mask = build_heatmap_targets(allowed(raw, entry['unknown_classes'], dataset), raw >= 0, spacing)
            crop = tuple(slice(int(h)//4, int(h)//4+t.stop-t.start) for h, t in zip(halo, section))
            heat, mask = heat[(0, slice(None), *crop)], mask[(0, slice(None), *crop)]
            values = torch.from_numpy(np.array(logits[(slice(None), *section)], copy=True))
            probability = values.sigmoid()
            for task in range(3):
                h, g, m, p = values[task], heat[task], mask[task], probability[task]
                stats[task] += [float((g*m).double().sum()), float(((1-g)*m).double().sum()),
                                float((F.softplus(-h)*g*m).double().sum()),
                                float((F.softplus(h)*(1-g)*m).double().sum()),
                                float((p*g*m).double().sum()), float((p*m).double().sum())]
            del raw, heat, mask, values, probability
        workspace.release(entry['label'])
        workspace.progress('q_localizer_loss', index, total)
    terms = []
    for positive, negative, bce_positive, bce_negative, intersection, predicted in stats:
        if positive+negative == 0:
            continue
        bce = ([bce_positive/positive] if positive > 0 else []) + ([bce_negative/negative] if negative > 0 else [])
        dice = 1-(2*intersection+1e-6)/(predicted+positive+1e-6) if positive > 0 else 0.
        terms.append(float(np.mean(bce))+dice)
    return float(np.mean(terms)) if terms else 0.
