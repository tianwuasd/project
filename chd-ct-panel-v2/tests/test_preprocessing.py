"""Scientific contracts: cohort, ignore labels, train-only fitting and recovery."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class MethodContractTests(unittest.TestCase):
    def test_registry_contains_only_implemented_methods(self):
        from chd_stunet.preprocessing.registry import get_method, list_methods
        self.assertEqual([m['name'] for m in list_methods()], ['stunet-s','stunet-s-partial'])
        self.assertEqual(get_method('stunet-s')['expected_included'], 96)
        with self.assertRaises(ValueError):
            get_method('imaginary')

    def test_filter_preserves_old_roles_instead_of_reshuffling(self):
        from chd_stunet.preprocessing.cohort import frozen_roles, select_cohort
        ids = [f'ct_{1000+i}' for i in range(97)]
        old = frozen_roles(ids)
        rows = [dict(case_id=c, missing_label_ids=[], roles=old[c]) for c in ids]
        method = dict(exclude_cases=[ids[7]], expected_included=96)
        kept, excluded = select_cohort(rows, method)
        self.assertEqual(len(kept), 96)
        self.assertEqual(kept[7]['roles'], old[ids[8]])
        self.assertEqual(excluded[0]['case_id'], ids[7])

    def test_extra_mapping_preserves_true_labels_and_refuses_unknown(self):
        from chd_stunet.preprocessing.cohort import map_labels
        source = np.array([0,1,2,3,4,5,6,7,8,10,12,13,14,15], dtype=np.int16)
        self.assertEqual(map_labels(source).tolist(), list(range(9))+[8]*5)
        self.assertEqual(source[-1], 15)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            map_labels(np.array([9], dtype=np.int16))
        with self.assertRaises(ValueError):
            map_labels(np.array([1.5]))

    def test_foreground_sampling_excludes_ignore_and_background(self):
        from chd_stunet.preprocessing.fingerprint import sample_foreground
        result = sample_foreground(np.array([0,10,1000]), np.array([0,1,8]), 100)
        self.assertTrue(np.all(result == 10))

    def test_ignore_is_not_a_ninth_model_output(self):
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        from chd_stunet.preprocessing.cohort import TARGETS
        manager=LabelManager(TARGETS,regions_class_order=None)
        self.assertTrue(manager.has_ignore_label)
        self.assertEqual(manager.ignore_label,8)
        self.assertEqual(manager.num_segmentation_heads,8)

    def test_train_only_fit_rejects_validation_samples(self):
        from chd_stunet.preprocessing.fingerprint import fit_statistics
        rows = [dict(case_id='a', roles={'0':'train'}), dict(case_id='b',roles={'0':'val'})]
        stats, ids = fit_statistics(rows, {'a':np.array([10,20]), 'b':np.array([1000])}, 0)
        self.assertEqual(ids, ['a'])
        self.assertEqual(stats['mean'], 15.)
        self.assertEqual(stats['std'], 5.)

    def test_memory_estimate_includes_target_resampling_growth(self):
        from chd_stunet.acceleration.preprocess_cpu import estimate_memory
        small = estimate_memory([100,100,100], [1,1,1], [1,1,1])
        large = estimate_memory([100,100,100], [2,2,2], [1,1,1])
        self.assertGreater(large, small)

    def test_admission_does_not_credit_shared_rss_or_subtract_cpu_twice(self):
        from chd_stunet.acceleration.preprocess_cpu import admits
        stage = dict(cpu_budget=8, memory_budget_bytes=1000, workers=8)
        fresh = dict(cpu_budget=2, memory_budget_bytes=700, workers=2)
        active = [dict(estimate=400, private=300, threads=1)]
        self.assertTrue(admits(stage, fresh, active, 400, 1))
        self.assertFalse(admits(stage, fresh, active, 700, 1))
        self.assertFalse(admits(stage, fresh, [dict(estimate=400,private=0,threads=1)], 400, 1))


if __name__ == '__main__':
    unittest.main()
