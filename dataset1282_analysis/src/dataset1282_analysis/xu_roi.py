"""Xu dual-scale localization. Fit the box from predictions; audit labels separately."""
from math import prod

import numpy as np

from .xu_geometry import _box_in, _prediction, _scale_bounds, prediction_box


RULE = 'dual-hard-foreground-128-margin-v2'
PADDING_PERCENT_OPTIONS = (0, 10, 20)
DEFAULT_PADDING_PERCENT = 20
# Project acceptance limits, not values reported in the paper.
MAX_BOX_FRACTION = 0.95
MIN_CLASS_P05 = 0.95


def _bounds(mask, cache_shape):
    if not mask.any():
        return None
    result = []
    for axis, length in enumerate(cache_shape):
        other = tuple(i for i in range(3) if i != axis)
        occupied = np.flatnonzero(mask.any(axis=other))
        result.append(_scale_bounds(int(occupied[0]), int(occupied[-1]) + 1,
                                    mask.shape[axis], length))
    return result


def box_fraction(box, shape):
    bounds = _box_in(box, shape)
    return prod(high - low for low, high in bounds) / prod(shape)


def native_support(box, geometry):
    """Native XYZ intervals whose voxel centres actually sample inside the ROI (restore_label)."""
    result = [None] * 3
    for axis, ((lo, hi), (start, stop), count) in enumerate(zip(
            _box_in(box, geometry.cache_shape), geometry.bbox_transposed, geometry.cache_shape)):
        length = stop - start
        # A centre j belongs iff 2*length*lo <= (2*j+1)*count < 2*length*hi.
        def first(edge):
            return start + max(0, min(length, -((count-2*length*edge) // (2*count))))
        result[2-geometry.transpose_forward[axis]] = (first(lo), first(hi))
    return tuple(result)


def measure_box(box, geometry):
    if box is None:
        return dict(accepted=False, reason='empty_vote', cache_fraction=None, native_fraction=None)
    cache_fraction = box_fraction(box, geometry.cache_shape)
    native = geometry.cache_box_to_native(box)
    native_fraction = box_fraction(native, geometry.native_shape_xyz)
    support = native_support(box, geometry)
    largest = max(cache_fraction, native_fraction)
    if any(lo >= hi for lo, hi in support):
        reason = 'no_native_support'
    elif largest == 1:
        reason = 'whole_volume'
    elif largest > MAX_BOX_FRACTION:
        reason = 'almost_whole_volume'
    else:
        reason = None
    return dict(accepted=reason is None, reason=reason, cache_fraction=cache_fraction,
                native_fraction=native_fraction, native_box=native, native_support=support)


def expand_box(box, shape, percent):
    """Increase total side length, splitting added voxels between its two faces, then clip."""
    if type(percent) is not int or percent not in PADDING_PERCENT_OPTIONS:
        raise ValueError(f'Expected a total side expansion from {PADDING_PERCENT_OPTIONS} percent')
    if box is None:
        return None
    expanded = []
    for (low, high), length in zip(_box_in(box, shape), shape):
        extra = ((high-low)*percent + 99) // 100
        expanded.append((max(0, low-extra//2), min(length, high+extra-extra//2)))
    return expanded


def vote_box(pred64, pred128, geometry, padding_percent=DEFAULT_PADDING_PERCENT):
    """Hard voting on a shared voxel-centred grid. Conflicting foreground types still support ROI."""
    low = _prediction(pred64)
    high = _prediction(pred128)
    if low.shape != (64,) * 3 or high.shape != (128,) * 3:
        raise ValueError('Require whole-scan 64 and 128 cubes')
    # Exactly 2x nearest-neighbour expansion under the existing voxel-centre convention.
    low = low.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    fg_low, fg_high = low > 0, high > 0
    agree = fg_low & fg_high
    strict = _bounds(agree, geometry.cache_shape)
    box = expand_box(strict, geometry.cache_shape, padding_percent)
    legacy = prediction_box(pred64, high, geometry.cache_shape) if (fg_low.any() or fg_high.any()) else None
    return dict(box=box, strict_box=strict, quality=measure_box(box, geometry), legacy_box=legacy,
                union_box=_bounds(fg_low | fg_high, geometry.cache_shape),
                same_foreground=int(np.count_nonzero(agree & (low == high))),
                different_foreground=int(np.count_nonzero(agree & (low != high))),
                disputed_membership=int(np.count_nonzero(fg_low ^ fg_high)))


def _counts(volume):
    counts = np.zeros(11, dtype=np.int64)
    for i in range(0, volume.shape[2], 8):
        slab = np.asarray(volume[:, :, i:i+8])
        if slab.dtype.kind not in 'ui' or slab.min() < 0 or slab.max() > 10:
            raise ValueError('Invalid native reference label')
        counts += np.bincount(slab.ravel(), minlength=11)
    return counts


def coverage(reference, boxes, geometry, present):
    """Fractions of each known native structure retained, including omitted structures as zero."""
    if reference.shape != geometry.native_shape_xyz:
        raise ValueError('Wrong native reference geometry')
    totals = _counts(reference)
    classes = np.flatnonzero(totals[1:]) + 1
    if set(classes) != set(present):
        raise ValueError('Native reference and supervised classes disagree')
    result = {}
    for key, box in boxes.items():
        retained = np.zeros_like(totals)
        if box is not None:
            native = native_support(box, geometry)
            if all(a < b for a,b in native):
                retained = _counts(reference[tuple(slice(a, b) for a, b in native)])
        result[key] = {int(c): float(retained[c] / totals[c]) for c in classes}
    return result
