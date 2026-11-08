import copy
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from chd_stunet.prediction.model import inference_plan, inspect_training, validate_producer
from chd_stunet.training.contracts import scientific_sources
from chd_stunet.prediction.inputs import validate_test_role


class PredictionContracts(unittest.TestCase):
    def test_unknown_training_science_is_refused(self):
        known = scientific_sources()
        validate_producer(known)
        known['src/chd_stunet/training/losses.py'] = 'unreviewed'
        with self.assertRaisesRegex(ValueError, '审计'):
            validate_producer(known)

    def test_prediction_plan_preserves_training_plan(self):
        source = {'configurations': {'3d_fullres': {'patch_size': [128]*3}}, 'fold': 0}
        before = copy.deepcopy(source)
        result = inference_plan(source, [64]*3)
        self.assertEqual(source, before)
        config = result['configurations']['3d_fullres']
        self.assertEqual(config['patch_size'], [64]*3)
        self.assertEqual(config['resampling_fn_probabilities_kwargs']['order'], 1)

    def test_test_role_checks_both_training_and_validation(self):
        row = {'case_id': 'case_a', 'roles': {'0': 'test'}, 'cohort_group': 'main'}
        model = {'fold': 0, 'signature': {'data': {'train_ids': [], 'val_ids': [], 'cases': []}}}
        validate_test_role(row, model)
        for role in ('train_ids', 'val_ids'):
            altered = copy.deepcopy(model)
            altered['signature']['data'][role] = ['case_a']
            with self.assertRaisesRegex(ValueError, '训练|验证'):
                validate_test_role(row, altered)
        row['roles']['0'] = 'aux_train'
        with self.assertRaises(ValueError):
            validate_test_role(row, model)

    def test_renamed_training_image_is_refused(self):
        row = {'case_id': 'different_name', 'roles': {'0': 'test'}, 'cohort_group': 'main',
               'source_hashes': {'image': 'same-image-digest'}}
        model = {'fold': 0, 'signature': {'data': {'train_ids': ['old_name'], 'val_ids': [],
                  'cases': [{'source_hashes': {'image': 'same-image-digest'}}]}}}
        with self.assertRaisesRegex(ValueError, '来源重复'):
            validate_test_role(row, model)

    @unittest.skipUnless(os.environ.get('CHD_TEST_TRAINING'), '需要已完成的真实训练路径')
    def test_real_old_checkpoint_is_read_without_modification(self):
        run = Path(os.environ['CHD_TEST_TRAINING'])
        signature_before = (run/'signature.json').read_bytes()
        record = inspect_training(run, 'best', smoke=True)
        self.assertEqual(record['fold'], 0)
        self.assertEqual(record['patch_size'], [64]*3)
        self.assertEqual((run/'signature.json').read_bytes(), signature_before)
        self.assertEqual(record['dataset_json']['labels']['PA'], 7)


if __name__ == '__main__':
    unittest.main()
