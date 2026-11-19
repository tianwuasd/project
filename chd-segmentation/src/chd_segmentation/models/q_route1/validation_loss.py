import math

import numpy as np
import torch

from ..stunet.streaming import tile_extent, tiles


def segmentation_statistics(probability, label, missing, dataset, workspace):
    from .adapter import allowed

    ce, count = 0., 0
    intersect, predicted, reference = (np.zeros(11, dtype=np.float64) for _ in range(3))
    voxels = max(1, workspace.block_bytes // 320)
    total = math.prod(math.ceil(n / b) for n, b in zip(label.shape, tile_extent(label.shape, voxels)))
    for index, section in enumerate(tiles(label.shape, voxels), 1):
        with workspace.boundary():
            raw = np.array(label[section], copy=True)
            support = torch.from_numpy(allowed(raw, missing, dataset))[None]
            valid = torch.from_numpy(raw >= 0)[None]
            values = torch.from_numpy(np.array(probability[(slice(None), *section)], copy=True))
            logits = values.clamp_min(1e-8).log()[None]
            exact_count = support.sum(1)
            selected = valid & (exact_count < 11)
            if selected.any():
                chosen = logits.movedim(1, -1)[selected]
                permitted = support.movedim(1, -1)[selected]
                losses = -chosen.log_softmax(-1).masked_fill(~permitted, -torch.inf).logsumexp(-1)
                ce += float(losses.double().sum())
                count += int(selected.sum())
            normalized = logits.softmax(1)
            exact = exact_count == 1
            for c in range(1, 11):
                positive = exact & support[:, c] & valid
                known = valid & (exact | ~support[:, c])
                p = normalized[:, c]
                intersect[c] += float((p * positive).double().sum())
                predicted[c] += float((p * known).double().sum())
                reference[c] += int(positive.sum())
            del values, logits, normalized, support, valid, raw
        workspace.release(probability, label)
        workspace.progress('q_validation_loss', index, total)
    active = reference[1:] > 0
    dice = 1 - (2 * intersect[1:][active] + 1e-6) / (predicted[1:][active] + reference[1:][active] + 1e-6)
    result = (ce / count if count else 0.) + (float(dice.mean()) if active.any() else 0.)
    if not np.isfinite(result):
        raise FloatingPointError('Q分块验证损失非有限')
    return float(result)
