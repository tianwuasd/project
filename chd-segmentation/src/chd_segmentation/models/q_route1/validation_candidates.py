from itertools import product
import math

import numpy as np

from ..stunet.streaming import tiles
from .candidates import _candidate, _grid_box, _iou


def bounded_candidates(heatmap, shape, workspace, quotas=(4, 4, 8), threshold=.2, dedup_iou=.8):
    grid_shape = heatmap.shape[1:]
    selected = []
    for fractions in product((.25, .75), repeat=3):
        candidate = _candidate(tuple(f*n for f, n in zip(fractions, shape)), 'fixed')
        if all(_iou(candidate, c, shape) < dedup_iou for c in selected):
            selected.append(candidate)
    remaining = workspace.create('layer', heatmap.shape, np.float32)
    voxels = max(1, workspace.block_bytes // 256)
    for section in tiles(grid_shape, voxels):
        with workspace.boundary():
            value = np.array(heatmap[(slice(None), *section)], copy=True)
            if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                raise ValueError('定位概率范围或数值无效')
            coords = np.meshgrid(*(np.arange(s.start, s.stop)*4+1.5 for s in section), indexing='ij')
            valid = np.ones(value.shape[1:], dtype=bool)
            for coord, n in zip(coords, shape):
                valid &= coord < n
            remaining[(slice(None), *section)] = value*(value >= threshold)*valid[None]
        workspace.release()

    def clear(candidate):
        low, high = _grid_box(tuple(s+24 for s in candidate.start), 80, grid_shape)
        remaining[(slice(None), *(slice(a, b) for a, b in zip(low, high)))] = 0

    for candidate in selected:
        clear(candidate)
    integral = workspace.create('layer', tuple(n+1 for n in grid_shape), np.float64)
    for task, quota in enumerate(quotas):
        for _ in range(quota):
            for section in tiles(grid_shape, voxels):
                dst = tuple(slice(t.start+1, t.stop+1) for t in section)
                integral[dst] = remaining[(task, *section)]
                workspace.release()
            for axis in range(3):
                for section in tiles(integral.shape, voxels, full_axis=axis):
                    with workspace.boundary():
                        np.cumsum(integral[section], axis=axis, out=integral[section])
                    workspace.release()
            chosen, best = None, None
            for section in tiles(grid_shape, voxels):
                with workspace.boundary():
                    indices = np.stack(np.meshgrid(*(np.arange(t.start, t.stop) for t in section), indexing='ij'), -1).reshape(-1, 3)
                    centers = indices*4+1.5
                    valid = (centers < np.asarray(shape)).all(1)
                    indices, centers = indices[valid], centers[valid]
                    starts = np.floor(centers-63).astype(np.int64)
                    lows = np.clip(np.ceil((starts+16-1.5)/4).astype(np.int64), 0, grid_shape)
                    highs = np.clip(np.ceil((starts+112-1.5)/4).astype(np.int64), 0, grid_shape)
                    scores = np.zeros(len(centers), dtype=np.float64)
                    for bits in product((0, 1), repeat=3):
                        points = np.where(np.asarray(bits)[None], highs, lows)
                        scores += (-1)**(3-sum(bits))*integral[tuple(points.T)]
                    scores /= 24**3
                    order = np.argsort(-scores, kind='stable')
                    for i in order:
                        if scores[i] <= 0:
                            break
                        candidate = _candidate(centers[i], 'learned', task, scores[i])
                        if all(_iou(candidate, c, shape) < dedup_iou for c in selected):
                            key = (-scores[i], int(np.ravel_multi_index(tuple(indices[i]), grid_shape)))
                            if best is None or key < best:
                                best, chosen = key, candidate
                            break
                workspace.release()
            if chosen is None:
                break
            selected.append(chosen)
            clear(chosen)
            workspace.progress('q_candidates', len(selected), 24)
    workspace.drop(remaining)
    workspace.drop(integral)
    return selected
