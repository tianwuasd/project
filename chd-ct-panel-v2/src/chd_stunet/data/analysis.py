"""Read-only native-grid statistics. No repairs, resampling, or cohort exclusion.

Labels follow the previously audited ImageCHD dictionary. Spreadsheet spacing is
reported separately from native headers; derived physical volumes remain provisional.
"""
from __future__ import annotations
import hashlib
from contextlib import nullcontext
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

LABELS = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "MYO", 6: "AO", 7: "PA"}
VERSION = "imagechd-read-only-analysis-v1"


def sha256_file(path: str | Path) -> str:
    """Hash compressed file bytes without changing the source or retaining them."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: np.ndarray, *, exact_mode: bool = True) -> dict:
    """Exact finite counts/moments; bounded sampled percentiles, documented mode."""
    flat = np.asarray(values).ravel(order="K")
    count, total, square, low, high = 0, 0.0, 0.0, None, None
    for start in range(0, flat.size, 1_000_000):
        block = flat[start:start + 1_000_000]
        block = block[np.isfinite(block)]
        if not block.size:
            continue
        count += int(block.size)
        total += float(block.sum(dtype=np.float64))
        square += float(np.einsum("i,i->", block, block, dtype=np.float64, casting="unsafe"))
        bmin, bmax = float(block.min()), float(block.max())
        low, high = bmin if low is None else min(low, bmin), bmax if high is None else max(high, bmax)
    result = dict(voxels=int(flat.size), finite_voxels=count,
                  nonfinite_voxels=int(flat.size-count), min=low, max=high,
                  mean=None, std=None, percentiles=None, percentile_sample_n=0,
                  mode=None, mode_fraction=None, mode_method=None)
    if not count:
        return result
    mean = total / count
    result.update(mean=mean, std=float(np.sqrt(max(0.0, square/count-mean*mean))))
    if flat.size > 200_000:
        indices = np.sort(np.random.default_rng(0).choice(flat.size, 200_000, replace=False))
        sample = flat[indices]
    else:
        sample = flat
    sample = sample[np.isfinite(sample)]
    result.update(percentiles=dict(zip(("p01", "p05", "p50", "p95", "p99"),
                                      map(float, np.percentile(sample, [1,5,50,95,99])))),
                  percentile_sample_n=int(sample.size),
                  percentiles_method="exact" if flat.size <= 200_000 else "fixed-seed uniform sample")
    # Histogram is exact for bounded integer input and avoids sorting every voxel.
    if exact_mode and np.issubdtype(flat.dtype, np.integer) and high-low <= 131_072:
        bins = np.zeros(int(high-low)+1, dtype=np.int64)
        for start in range(0, flat.size, 1_000_000):
            block = flat[start:start + 1_000_000].astype(np.int64)
            bins += np.bincount(block-int(low), minlength=bins.size)
        maximum = int(bins.argmax())
        result.update(mode=float(maximum+low), mode_fraction=float(bins[maximum]/count),
                      mode_method="exact integer histogram")
    else:
        chosen = flat[np.isfinite(flat)] if exact_mode and flat.size <= 2_000_000 else sample
        unique, counts = np.unique(chosen, return_counts=True)
        index = int(counts.argmax())
        result.update(mode=float(unique[index]), mode_fraction=float(counts[index]/chosen.size),
                      mode_method="exact unique" if chosen.size == count else "sampled unique")
    return result


def bounding_box(mask: np.ndarray) -> list[list[int]] | None:
    """Half-open voxel-index bounds; no physical-coordinate claim."""
    result = []
    for axis in range(3):
        present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        if not present.size:
            return None
        result.append([int(present[0]), int(present[-1])+1])
    return result


def _snapshot(image, labels, destination):
    """Store three small 2D scientific views, avoiding a second volume decode."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    focus = labels == 5
    if not focus.any():
        focus = (labels >= 1) & (labels <= 7)
    payload = {}
    for axis in range(3):
        support = focus.sum(axis=tuple(a for a in range(3) if a != axis))
        index = int(support.argmax()) if support.any() else image.shape[axis]//2
        payload[f"image{axis}"] = np.take(image, index, axis=axis)
        payload[f"label{axis}"] = np.take(labels, index, axis=axis)
        payload[f"index{axis}"] = np.asarray(index)
    np.savez_compressed(destination, **payload)


def analyze_pair(image_path: str, label_path: str, *, case_id: str,
                 expected_hashes: dict | None = None, author_spacing: list | None = None,
                 components: bool = False, selected_components: bool = False,
                 snapshot_path: str | None = None, read_guard=None) -> dict:
    """Analyze a native 3D pair and return JSON-safe evidence.

    Invalid labels and geometry are reported, not coerced. Per-class image
    intensity is computed only when shape and affine match. Connectivity is 6.
    Both compressed source hashes are checked again before returning.
    """
    started = time.perf_counter()
    paths = {"image": Path(image_path), "label": Path(label_path)}
    with read_guard if read_guard is not None else nullcontext():
        before = {name: sha256_file(path) for name, path in paths.items()}
        source, reference = nib.load(paths["image"]), nib.load(paths["label"])
        if source.ndim != 3 or reference.ndim != 3:
            raise ValueError("Only native 3D source pairs are supported")
        image, labels = np.asanyarray(source.dataobj), np.asanyarray(reference.dataobj)
    grid_match = source.shape == reference.shape and bool(np.allclose(
        source.affine, reference.affine, rtol=0, atol=1e-5))
    flags = []
    if not grid_match:
        flags.append("image_label_grid_mismatch")
    if not np.isfinite(source.affine).all() or abs(np.linalg.det(source.affine[:3,:3])) < 1e-8:
        flags.append("invalid_image_affine")
        grid_match = False
    units = list(source.header.get_xyzt_units())
    if units[0] != "mm":
        flags.append("native_spatial_unit_not_mm")
    expected_hashes = expected_hashes or {}
    for kind, actual in before.items():
        if expected_hashes.get(kind) and expected_hashes[kind] != actual:
            flags.append(f"descriptor_{kind}_sha_mismatch")
    unique, counts = np.unique(labels, return_counts=True)
    invalid = ~np.isfinite(unique) | (unique < 0) | (unique > 255) | (unique != np.floor(unique))
    invalid_voxels = int(counts[invalid].sum())
    if invalid_voxels:
        flags.append("invalid_label_values")
    extras = [int(x) for x in unique[~invalid] if x > 7]
    if extras:
        flags.append("additional_source_label_ids")
    intensity = describe(image, exact_mode=False)
    if intensity["nonfinite_voxels"]:
        flags.append("nonfinite_image_voxels")
    if intensity["std"] == 0:
        flags.append("constant_image")
    spacing = None
    if author_spacing is not None:
        spacing = np.asarray(author_spacing, dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Author-reported spacing must be three finite positive values")
    structures = {}
    for label_id, name in LABELS.items():
        mask = labels == label_id
        voxels = int(mask.sum())
        stats = describe(image[mask]) if voxels and grid_match else describe(np.asarray([], dtype=float))
        box = bounding_box(mask) if voxels else None
        stats.update(voxels=voxels, label_id=label_id, bbox=box,
                     author_volume_ml=float(voxels*np.prod(spacing)/1000) if spacing is not None else None,
                     touches_boundary=bool(box and any(lo == 0 or hi == labels.shape[a]
                                                      for a, (lo,hi) in enumerate(box))),
                     component_count=None, largest_component_fraction=None)
        if (stats["mode_fraction"] is not None and stats["mode_fraction"] >= .5
                and voxels >= 1000 and stats["mode_method"].startswith("exact")):
            flags.append("dominant_class_constant_intensity:"+name)
        structures[name] = stats
    missing = [name for name, stats in structures.items() if not stats["voxels"]]
    if missing:
        flags.append("missing_target_labels")
    do_components = components or (selected_components and (
        bool(missing) or any(f.startswith("dominant_class") for f in flags)))
    if do_components:
        for label_id, name in LABELS.items():
            if not structures[name]["voxels"]:
                structures[name].update(component_count=0, largest_component_fraction=None)
                continue
            connected, count = ndimage.label(labels == label_id, structure=ndimage.generate_binary_structure(3,1))
            sizes = np.bincount(connected.ravel())[1:]
            structures[name].update(component_count=int(count),
                largest_component_fraction=float(sizes.max()/structures[name]["voxels"]))
            del connected, sizes
    foreground = (labels >= 1) & (labels <= 7)
    foreground_box = bounding_box(foreground)
    coverage = ([float((hi-lo)/labels.shape[a]) for a,(lo,hi) in enumerate(foreground_box)]
                if foreground_box else None)
    pixel_digest = hashlib.sha256()
    pixel_digest.update((str(image.shape)+str(image.dtype)).encode())
    pixel_digest.update(np.ascontiguousarray(image))
    if snapshot_path and grid_match:
        _snapshot(image, labels, snapshot_path)
    with read_guard if read_guard is not None else nullcontext():
        after = {name: sha256_file(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        flags.append("source_changed_during_analysis")
    return dict(case_id=case_id, analysis_version=VERSION, analysis_success=unchanged,
        image=str(paths["image"]), label=str(paths["label"]), image_sha256=before["image"],
        label_sha256=before["label"], sha256_after=after, source_unchanged=unchanged,
        image_voxel_sha256=pixel_digest.hexdigest(), shape=list(source.shape),
        label_shape=list(reference.shape), image_dtype=str(image.dtype), label_dtype=str(labels.dtype),
        affine=source.affine.tolist(), label_affine=reference.affine.tolist(),
        header_spacing=list(map(float, source.header.get_zooms()[:3])), header_units=units,
        qform_code=int(source.header["qform_code"]), sform_code=int(source.header["sform_code"]),
        axis_codes=list(nib.aff2axcodes(source.affine)), grid_match=grid_match,
        author_spacing_mm=None if spacing is None else spacing.tolist(),
        author_physical_geometry_verified=False,
        label_counts={str(float(value)) if not np.isfinite(value) or value != np.floor(value) else str(int(value)):int(n)
                      for value,n in zip(unique,counts)},
        extra_label_ids=extras, invalid_label_voxels=invalid_voxels,
        missing_structures=missing, structures=structures, intensity=intensity,
        foreground_bbox=foreground_box, foreground_bbox_axis_coverage=coverage,
        components_checked=do_components, connectivity=6 if do_components else None,
        snapshot=str(snapshot_path) if snapshot_path and grid_match else None,
        flags=sorted(set(flags)), elapsed_seconds=time.perf_counter()-started)