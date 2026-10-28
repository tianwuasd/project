"""Measured features plus explicit unknowns for unpublished anatomical localization."""

import numpy as np
from scipy import ndimage as ndi

from ..data import validate_labels
from ..labels import LABELS
from .graph import graph_summary

PAIRS = [
    ("LV", "RV"),
    ("LV", "AO"),
    ("RV", "AO"),
    ("RV", "PA"),
    ("LV", "RA"),
    ("RV", "LA"),
    ("LA", "RA"),
    ("LV", "PA"),
    ("PV", "LA"),
    ("PV", "RA"),
    ("PV", "SVC"),
    ("PV", "IVC"),
    ("AO", "PA"),
]


def connection(a, b, spacing, contact_mm):
    if not a.any() or not b.any():
        return None
    return bool(np.min(ndi.distance_transform_edt(~a, sampling=spacing)[b]) <= contact_mm)


def extract_features(labels, spacing, initial=None, spine=None, affine=None, contact_mm=1.5):
    labels = validate_labels(np.asarray(labels), 11)
    spacing = np.asarray(spacing, dtype=float)
    if (
        spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
        or not np.isfinite(contact_mm)
        or contact_mm < 0
    ):
        raise ValueError("Positive finite 3D spacing and nonnegative contact distance required")
    if affine is not None:
        affine = np.asarray(affine, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("Expected a finite 4x4 affine")
        axes = affine[:3, :3] / spacing
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
            raise ValueError("Distance features require an orthogonal affine matching the voxel spacing")
    masks = {k: labels == v for k, v in LABELS.items() if k != "BG"}
    result = {f"volume_{k}_mm3": float(v.sum() * spacing.prod()) for k, v in masks.items()}
    for a, b in PAIRS:
        result[f"conn_{a}_{b}"] = connection(masks[a], masks[b], spacing, contact_mm)
    result["conn_ao_pa"] = result["conn_AO_PA"]
    for vessel in ("AO", "SVC"):
        _, number = ndi.label(masks[vessel])
        result[f"n_island_{vessel}"] = int(number) if number else None
    islands, number = ndi.label(masks["SVC"])
    for target in ("RA", "LA", "AO", "PA"):
        result[f"n_island_SVC_to_{target}"] = (
            sum(connection(islands == i, masks[target], spacing, contact_mm) for i in range(1, number + 1))
            if number and masks[target].any()
            else None
        )
    result["n_island_init_part"] = None
    result["ratio_PA_to_AO"] = None
    if initial is not None:
        if initial.shape != labels.shape:
            raise ValueError("Initial mask must share the label grid")
        initial = validate_labels(initial, 7) == 6
        _, count = ndi.label(initial)
        result["n_island_init_part"] = int(count) if count else None
        ao, pa = int((initial & masks["AO"]).sum()), int((initial & masks["PA"]).sum())
        # Table 2's symbol says PA/AO but its verbal definition says AO/PA; choose the definition.
        result["ratio_PA_to_AO"] = ao / pa if pa and ao else None
    for vessel in ("AO", "PA"):
        summary = graph_summary(masks[vessel], spacing)
        result.update({f"graph_{vessel}_{k}": v for k, v in summary.items()})
    # These require precise trunk, valve, arch and descending-aorta localization, not supplied in the paper.
    for key in (
        "ratio_PA_to_low_AO",
        "stenosis_AO_arch",
        "stenosis_descending_AO",
        "stenosis_PA_trunk",
        "stenosis_PA_valve_upper",
        "stenosis_PA_valve",
        "stenosis_PA_valve_down",
        "single_ventricle_confirmed",
        "single_atrium_confirmed",
        "ao_override_fraction",
        "pa_atresia_confirmed",
        "common_arterial_trunk_confirmed",
    ):
        result[key] = None
    for level in ("arch", "low", "middle"):
        result[f"posi_AO_to_spine_{level}"] = None
    if spine is not None:
        if spine.shape != labels.shape:
            raise ValueError("Spine mask must share the label grid")
        zs = np.flatnonzero(masks["AO"].any(axis=(0, 1)))
        if zs.size:
            slices = {"arch": int(zs.max()), "low": int(zs.min()), "middle": int((zs.max() + zs.min()) // 2)}
            matrix = np.diag([*spacing, 1.0]) if affine is None else affine
            for level, z in slices.items():
                a = np.argwhere(masks["AO"][:, :, z])
                s = np.argwhere(spine[:, :, z] > 0)
                if a.size and s.size:
                    delta = np.r_[a.mean(0) - s.mean(0), 0.0]
                    result[f"posi_AO_to_spine_{level}"] = float((matrix[:3, :3] @ delta)[0])
    result["_metadata"] = {
        "contact_mm": float(contact_mm),
        "ratio_convention": "AO/PA per Table 2 text",
        "graph_method": "26-neighbor skeleton and widest path; approximate",
        "status": "unvalidated research features",
    }
    return result
