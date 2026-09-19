import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_src"))
from metrics import binary_metrics


class MetricTests(unittest.TestCase):
    def test_identity(self):
        a = np.zeros((8, 8, 8), bool)
        a[2:5, 2:5, 2:5] = True
        result = binary_metrics(a, a, np.eye(4))
        self.assertEqual((result["dice"], result["hd_mm"], result["hd95_mm"]), (1, 0, 0))

    def test_physical_distance_with_shear(self):
        a, b = np.zeros((5, 5, 5), bool), np.zeros((5, 5, 5), bool)
        a[2, 2, 2], b[2, 3, 2] = True, True
        affine = np.array([[2, 1, 0, 10], [0, 3, 0, -5], [0, 0, 4, 7], [0, 0, 0, 1]])
        result = binary_metrics(a, b, affine)
        self.assertAlmostEqual(result["hd_mm"], np.sqrt(10))
        self.assertAlmostEqual(result["hd95_mm"], np.sqrt(10))

    def test_empty_predictions_reported(self):
        a, b = np.zeros((3, 3, 3), bool), np.zeros((3, 3, 3), bool)
        self.assertIsNone(binary_metrics(a, b, np.eye(4))["dice"])
        a[1, 1, 1] = True
        result = binary_metrics(b, a, np.eye(4))
        self.assertEqual(result["dice"], 0)
        self.assertTrue(np.isinf(result["hd_mm"]))


if __name__ == "__main__":
    unittest.main()
