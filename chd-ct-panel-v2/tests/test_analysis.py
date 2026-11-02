"""Synthetic behavioral checks; never use patient files as test fixtures."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "outputs" / "tests-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.folder = Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to(scratch.resolve()))
        self.addCleanup(self.temp.cleanup)
        self.image = (np.arange(512).reshape(8, 8, 8) % 100).astype(np.int16)
        self.label = np.zeros((8, 8, 8), dtype=np.uint8)
        for value in range(1, 8):
            self.label[value, 1:3, 1:3] = value
        self.image[self.label == 5] = 2492

    def subject(self):
        self.assertIsNotNone(importlib.util.find_spec("chd_stunet.data.analysis"),
                             "Read-only analysis module must be implemented")
        from chd_stunet.data.analysis import analyze_pair
        return analyze_pair

    def write_pair(self, image=None, label=None, label_affine=None):
        image = self.image if image is None else image
        label = self.label if label is None else label
        ip, lp = self.folder/"synthetic_image.nii.gz", self.folder/"synthetic_label.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), ip)
        nib.save(nib.Nifti1Image(label, np.eye(4) if label_affine is None else label_affine), lp)
        return ip, lp

    def analyze(self, **kwargs):
        ip, lp = self.write_pair()
        return self.subject()(str(ip), str(lp), case_id="synthetic", **kwargs)

    def test_counts_constant_region_and_source_bytes(self):
        ip, lp = self.write_pair()
        before = [p.read_bytes() for p in (ip, lp)]
        result = self.subject()(str(ip), str(lp), case_id="synthetic")
        self.assertEqual(result["missing_structures"], [])
        self.assertEqual(result["structures"]["MYO"]["voxels"], 4)
        self.assertEqual(result["structures"]["MYO"]["mode"], 2492)
        self.assertEqual(result["structures"]["MYO"]["mode_fraction"], 1.0)
        self.assertIsNone(result["structures"]["MYO"]["author_volume_ml"])
        self.assertTrue(result["source_unchanged"])
        self.assertEqual([p.read_bytes() for p in (ip, lp)], before)

    def test_missing_annotation_is_reported_without_edit(self):
        self.label[self.label == 5] = 0
        result = self.analyze()
        self.assertEqual(result["missing_structures"], ["MYO"])
        self.assertIn("missing_target_labels", result["flags"])
        self.assertEqual(result["structures"]["MYO"]["voxels"], 0)
        self.assertIsNone(result["structures"]["MYO"]["mean"])

    def test_nonfinite_image_is_counted_and_json_safe(self):
        self.image = self.image.astype(np.float32)
        self.image[0, 0, 0], self.image[0, 0, 1] = np.nan, np.inf
        result = self.analyze()
        self.assertEqual(result["intensity"]["nonfinite_voxels"], 2)
        json.dumps(result, allow_nan=False)

    def test_grid_mismatch_does_not_claim_class_intensity(self):
        affine = np.eye(4); affine[0, 3] = 1
        ip, lp = self.write_pair(label_affine=affine)
        result = self.subject()(str(ip), str(lp), case_id="synthetic")
        self.assertFalse(result["grid_match"])
        self.assertIn("image_label_grid_mismatch", result["flags"])
        self.assertIsNone(result["structures"]["MYO"]["mean"])

    def test_invalid_fractional_label_is_not_truncated(self):
        self.label = self.label.astype(np.float32)
        self.label[0, 0, 0] = 1.25
        result = self.analyze()
        self.assertEqual(result["invalid_label_voxels"], 1)
        self.assertIn("invalid_label_values", result["flags"])
        self.assertEqual(result["structures"]["LV"]["voxels"], 4)

    def test_components_use_documented_six_connectivity(self):
        self.label[:] = 0
        self.label[1, 1, 1] = self.label[6, 6, 6] = 1
        result = self.analyze(components=True)
        self.assertEqual(result["structures"]["LV"]["component_count"], 2)
        self.assertEqual(result["structures"]["LV"]["largest_component_fraction"], .5)

    def test_descriptor_mismatch_is_visible(self):
        result = self.analyze(expected_hashes={"image": "0"*64})
        self.assertIn("descriptor_image_sha_mismatch", result["flags"])
        self.assertTrue(result["source_unchanged"])

    def test_reported_author_spacing_does_not_replace_native_grid(self):
        result = self.analyze(author_spacing=[.25, .25, 1.0])
        self.assertEqual(result["header_spacing"], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(result["structures"]["MYO"]["author_volume_ml"], .00025)
        self.assertEqual(result["header_units"][0], "unknown")

    def test_all_background_is_valid_analysis_not_annotated_case(self):
        self.label[:] = 0
        result = self.analyze()
        self.assertEqual(len(result["missing_structures"]), 7)
        self.assertIsNone(result["foreground_bbox"])


if __name__ == "__main__":
    unittest.main()