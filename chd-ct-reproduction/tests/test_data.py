import csv

import nibabel as nib
import numpy as np
import pytest

from chd_ct.data import blood_target, bounding_box, load_volume, read_manifest, save_prediction


def test_mask_dimensions_are_validated():
    from chd_ct.data import validate_labels

    with pytest.raises(ValueError, match="3D"):
        validate_labels(np.zeros((3, 3)), 11)


def test_orientation_roundtrip(tmp_path):
    a = np.zeros((6, 7, 8), dtype=np.float32)
    a[1:3, 2:4, 3:5] = 1
    affine = np.diag([-0.7, 1.2, -2.0, 1.0])
    p = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(a, affine), p)
    v = load_volume(p)
    assert nib.aff2axcodes(v.affine) == ("R", "A", "S")
    out = tmp_path / "out.nii.gz"
    save_prediction(v.data.astype(np.uint8), v, out)
    result = nib.load(out)
    np.testing.assert_array_equal(result.get_fdata(), a)
    np.testing.assert_allclose(result.affine, affine)


def test_manifest_rejects_patient_leakage(tmp_path):
    p = tmp_path / "manifest.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "patient_id", "split", "image", "label"])
        w.writerow(["a", "same", "train", "a.nii.gz", "a_seg.nii.gz"])
        w.writerow(["b", "same", "val", "b.nii.gz", "b_seg.nii.gz"])
    with pytest.raises(ValueError, match="patient"):
        read_manifest(p, check_files=False)


def test_blood_boundary_and_empty_roi():
    labels = np.zeros((9, 9, 9), dtype=np.uint8)
    labels[2:7, 2:7, 2:7] = 1
    labels[0] = 7
    target = blood_target(labels)
    assert target[4, 4, 4] == 1
    assert target[2, 4, 4] == 2
    assert not target[0].any()
    assert bounding_box(np.zeros((3, 4, 5)), margin=1) == (slice(0, 3), slice(0, 4), slice(0, 5))
