import unittest

import numpy as np

from dataset1282_analysis.xu_evaluation import class_metrics, summarize


class NativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.zeros((5,6,7),np.uint8)
        self.truth[1,2,3] = 1
        self.meta = dict(present_classes=[1],unknown_classes=list(range(2,10)),known_absent_classes=[10])

    def test_one_voxel_translation_uses_native_xyz_mm(self):
        prediction = np.zeros_like(self.truth)
        prediction[2,2,3] = 1
        prediction[0,0,0] = 10
        result = class_metrics(self.truth,prediction,(0.4,0.8,2),self.meta)
        self.assertEqual(result[1]['dice'],0)
        self.assertAlmostEqual(result[1]['hd95_mm'],0.4)
        self.assertAlmostEqual(result[1]['asd_mm'],0.4)
        self.assertEqual(result[10],dict(status='verified_absent',predicted_voxels=1))

    def test_missing_surface_and_partial_classes_have_explicit_denominators(self):
        a = class_metrics(self.truth,self.truth,(1,1,1),self.meta)
        b = class_metrics(self.truth,np.zeros_like(self.truth),(1,1,1),self.meta)
        report = summarize([a,b])
        self.assertEqual(report['by_label'][1]['dice']['mean'],0.5)
        self.assertEqual(report['by_label'][1]['asd_mm']['n'],1)
        self.assertEqual(report['by_label'][1]['empty_prediction'],1)
        self.assertEqual(report['by_label'][2]['unannotated'],2)
        self.assertIsNone(report['macro_ten_classes']['dice']['mean'])
        self.assertEqual(summarize([])['cases'],0)

    def test_rejects_bad_grid_and_conflicting_annotations(self):
        with self.assertRaises(ValueError):
            class_metrics(self.truth,self.truth,(1,0,1),self.meta)
        with self.assertRaises(ValueError):
            class_metrics(self.truth,self.truth,(1,1,1),dict(self.meta,present_classes=[1,2]))
