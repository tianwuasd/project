"""NIfTI geometry, patient-safe manifests and explicit target construction."""

import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi

from .labels import BLOOD_IDS


@dataclass
class Volume:
    data: np.ndarray
    affine: np.ndarray
    original_affine: np.ndarray
    original_header: nib.Nifti1Header
    original_orientation: np.ndarray

    @property
    def spacing(self):
        return tuple(float(x) for x in nib.affines.voxel_sizes(self.affine))


def load_volume(path) -> Volume:
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError(f"Expected a 3D volume with every dimension >= 2: {path}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid NIfTI affine")
    canonical = nib.as_closest_canonical(image)
    data = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite image values: {path}")
    return Volume(
        data,
        canonical.affine,
        image.affine,
        image.header.copy(),
        nib.orientations.io_orientation(image.affine),
    )


def save_prediction(labels, reference: Volume, path):
    if labels.shape != reference.data.shape:
        raise ValueError("Prediction shape differs from canonical input shape")
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ras, reference.original_orientation)
    native = nib.orientations.apply_orientation(labels, transform).astype(np.uint8)
    header = reference.original_header.copy()
    header.set_data_dtype(np.uint8)
    out = nib.Nifti1Image(native, reference.original_affine, header)
    out.set_sform(reference.original_affine, code=1)
    # qform cannot represent shear; sform remains authoritative.
    out.set_qform(reference.original_affine, code=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))


def validate_labels(array, classes):
    if array.ndim != 3 or not array.size:
        raise ValueError("Expected a nonempty 3D segmentation")
    if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
        raise ValueError("Segmentation must contain finite integer labels")
    if array.min() < 0 or array.max() >= classes:
        raise ValueError(f"Labels must be in [0, {classes - 1}]")
    return array.astype(np.uint8)


def aligned_label(path, reference, classes=11):
    labels = load_volume(path)
    if labels.data.shape != reference.data.shape or not np.allclose(
        labels.affine, reference.affine, atol=1e-4
    ):
        raise ValueError("Image and label must have matching NIfTI grids and affines")
    return validate_labels(labels.data, classes)


def read_manifest(path, check_files=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "patient_id", "split", "image", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    ids, patients, images = set(), {}, {}
    for row in rows:
        if not all(row.get(k) for k in required):
            raise ValueError("Required manifest values cannot be empty")
        if row["case_id"] in ids:
            raise ValueError("Duplicate case_id")
        ids.add(row["case_id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        old = patients.setdefault(row["patient_id"], row["split"])
        if old != row["split"]:
            raise ValueError("A patient appears in multiple splits")
        for key in ("image", "label", "initial_label"):
            if row.get(key):
                p = (path.parent / row[key]).resolve()
                if check_files and not p.is_file():
                    raise ValueError(f"Missing {key}: {p}")
                row[key] = str(p)
        old_case = images.setdefault(row["image"], row["case_id"])
        if old_case != row["case_id"]:
            raise ValueError("Duplicate image used by different cases")
    return rows


def normalize_ct(array, window=(-200.0, 1000.0)):
    low, high = window
    if not low < high:
        raise ValueError("CT window must be increasing")
    return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)


def bounding_box(mask, margin=3):
    if margin < 0:
        raise ValueError("ROI margin must be nonnegative")
    points = np.where(mask > 0)
    if not points[0].size:
        return tuple(slice(0, n) for n in mask.shape)
    return tuple(
        slice(max(0, int(p.min()) - margin), min(n, int(p.max()) + margin + 1))
        for p, n in zip(points, mask.shape)
    )


def resize(array, shape, labels=False):
    x = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)[None, None]
    mode = "nearest" if labels else ("trilinear" if array.ndim == 3 else "bilinear")
    kwargs = {} if labels else {"align_corners": False}
    result = torch.nn.functional.interpolate(x, size=tuple(shape), mode=mode, **kwargs)[0, 0].numpy()
    return result.astype(np.int64) if labels else result


def blood_target(labels):
    """One-voxel inner boundary is an implementation assumption (paper is silent)."""
    pool = np.isin(labels, BLOOD_IDS)
    interior = ndi.binary_erosion(pool, structure=ndi.generate_binary_structure(3, 1))
    return np.where(pool, np.where(interior, 1, 2), 0).astype(np.uint8)
