"""用已知答案的小型影像验证统计逻辑，不依赖真实患者数据。"""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from medical import (describe_header, discover, file_hash, geometry_check,
                     inspect_case, label_statistics)
from analyze import prepare_output


class MedicalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data, affine=None):
        image = nib.Nifti1Image(data, affine if affine is not None else np.diag([1., 2., 3., 1.]))
        image.header.set_xyzt_units("mm")
        path = self.root / name
        nib.save(image, path)
        return path

    def test_labels_volume_and_unknown_preserved(self):
        a = np.array([0, 205, 205, 421, 500, 500, 500, 500], dtype=np.int16).reshape(2, 2, 2)
        rows, problems, nonfinite = label_statistics(a, .006)
        by_value = {r["value"]: r for r in rows}
        self.assertAlmostEqual(by_value[205]["volume_ml"], .012)
        self.assertEqual(by_value[421]["voxels"], 1)
        self.assertEqual(by_value[420]["voxels"], 0)
        self.assertAlmostEqual(by_value[500]["target_fraction"], 4 / 6)
        self.assertEqual(sum(r["voxels"] for r in rows), 8)
        self.assertIn("unexpected_label", [p["code"] for p in problems])
        self.assertEqual(nonfinite, 0)

    def test_fractional_and_nan_labels_not_rounded(self):
        a = np.array([0, 420.5, np.nan, 500], dtype=np.float32).reshape(2, 2, 1)
        rows, problems, count = label_statistics(a, None)
        self.assertEqual(count, 1)
        self.assertTrue(any(r["value"] == 420.5 for r in rows))
        self.assertTrue(all(r["volume_ml"] is None for r in rows))
        self.assertIn("fractional_label", [p["code"] for p in problems])

    def test_scaling_pair_and_no_mutation(self):
        path = self.root / "Case1_image.nii.gz"
        image = nib.Nifti1Image(np.arange(8, dtype=np.int16).reshape(2, 2, 2), np.eye(4))
        image.header.set_xyzt_units("mm")
        image.header.set_slope_inter(2, -1024)
        nib.save(image, path)
        label = self.write("Case1_label.nii.gz", np.full((2, 2, 2), 500, dtype=np.int16), np.eye(4))
        before = [file_hash(path), file_hash(label)]
        pairs, _ = discover(self.root)
        r, _ = inspect_case(pairs[0], 100)
        self.assertEqual(r["intensity"]["min"], -1024)
        self.assertEqual(r["intensity"]["max"], -1010)
        self.assertEqual(r["image_meta"]["slope"], 2)
        self.assertEqual(before, [file_hash(path), file_hash(label)])

    def test_geometry_codes_can_differ_without_spatial_mismatch(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.diag([1., 2., 3., 1.]))
        image.header.set_xyzt_units("mm")
        image.set_qform(image.affine, code=1)
        image.set_sform(image.affine, code=1)
        one = describe_header(image)
        image.set_sform(None, code=0)
        two = describe_header(image)
        self.assertEqual(geometry_check(one, two), [])

    def test_singular_sform_is_reported(self):
        affine = np.diag([1., 2., 3., 1.])
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), affine)
        image.header.set_xyzt_units("mm")
        image.set_qform(affine, code=1)
        image.set_sform(np.zeros((4, 4)), code=2)
        meta = describe_header(image)
        self.assertFalse(meta["affine_valid"])
        self.assertTrue(meta["display_fallback"])
        self.assertIn("invalid_affine", [p["code"] for p in geometry_check(meta, meta)])

    def test_physical_origin_mismatch_detected(self):
        p = self.write("a.nii.gz", np.zeros((2, 2, 2), dtype=np.int16))
        one = describe_header(nib.load(p))
        other = dict(one, affine=np.array(one["affine"]).tolist())
        other["affine"][0][3] += 2
        self.assertIn("geometry_mismatch", [p["code"] for p in geometry_check(one, other)])

    def test_meter_unit_converts_to_millimeters(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.diag([.001, .002, .003, 1]))
        image.header.set_xyzt_units("meter")
        meta = describe_header(image)
        np.testing.assert_allclose(meta["spacing_mm"], [1, 2, 3], rtol=1e-6)
        self.assertAlmostEqual(meta["voxel_ml"], .006)

    def test_missing_and_orphan_pair(self):
        self.write("Case1_image.nii.gz", np.zeros((2, 2, 2), dtype=np.int16))
        self.write("Case2_label.nii.gz", np.zeros((2, 2, 2), dtype=np.int16))
        pairs, issues = discover(self.root)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({p["code"] for p in issues}, {"missing_label", "orphan_label"})

    def test_submillimeter_meter_affine_is_valid(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.diag([.0004, .0004, .0005, 1]))
        image.header.set_xyzt_units("meter")
        meta = describe_header(image)
        self.assertTrue(meta["affine_valid"])
        self.assertAlmostEqual(meta["voxel_ml"], .00008)

    def test_affine_spacing_disagreement_and_volume(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4))
        image.header.set_xyzt_units("mm")
        image.set_sform(np.diag([2., 3., 4., 1.]), code=2)
        meta = describe_header(image)
        np.testing.assert_allclose(meta["spacing_mm"], [1, 1, 1])
        self.assertAlmostEqual(meta["voxel_ml"], .024)
        self.assertIn("spacing_affine_mismatch", [p["code"] for p in geometry_check(meta, meta)])

    def test_cannot_output_inside_input(self):
        with self.assertRaises(ValueError):
            prepare_output(self.root, self.root / "output")


if __name__ == "__main__":
    unittest.main()
