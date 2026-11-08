from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from chd_stunet.training.model import build_model, topology
from chd_stunet.prediction.infer import sliding_logits
from chd_stunet.prediction.export import score_segmentation


class PredictionNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        cls.model = build_model(topology([1, 1, 1], [32]*3)).eval()
        cls.model.decoder.deep_supervision = False

    def test_one_window_matches_direct_real_network(self):
        image = np.random.default_rng(91).normal(size=(1, 32, 32, 32)).astype('float32')
        with torch.inference_mode():
            expected = self.model(torch.from_numpy(image)[None])[0].numpy()
        actual, report = sliding_logits(self.model, image, [32]*3, torch.device('cpu'), batch=1, tta=False)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(report['windows'], 1)

    def test_batch_and_padding_preserve_window_fusion(self):
        image = np.random.default_rng(22).normal(size=(1, 37, 29, 35)).astype('float32')
        first, _ = sliding_logits(self.model, image, [32]*3, torch.device('cpu'), batch=1, tta=True)
        second, _ = sliding_logits(self.model, image, [32]*3, torch.device('cpu'), batch=2, tta=True)
        self.assertEqual(first.shape, (8, 37, 29, 35))
        np.testing.assert_allclose(first, second, rtol=3e-5, atol=3e-5)

    def test_empty_prediction_and_physical_distance(self):
        truth = np.zeros((7, 8, 9), dtype=np.uint8)
        truth[2, 3, 4] = 1
        prediction = np.zeros_like(truth)
        empty = score_segmentation(prediction, truth, np.diag([4., 2., 3., 1.]))
        self.assertEqual(empty['LV']['dice'], 0)
        self.assertIsNone(empty['LV']['hd95_mm'])
        self.assertEqual(empty['LV']['status'], 'missing_prediction')
        prediction[3, 3, 4] = 1
        metrics = score_segmentation(prediction, truth, np.diag([4., 2., 3., 1.]))
        self.assertEqual(metrics['LV']['hd95_mm'], 4.)
        self.assertIsNone(metrics['PA']['dice'])


if __name__ == '__main__':
    unittest.main()
