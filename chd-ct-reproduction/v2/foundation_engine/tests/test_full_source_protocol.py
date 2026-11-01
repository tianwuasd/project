"""All source labels are used while the independent target center stays held out."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import cohort
from test_cohort import cases
from server_data import write_json


class FullSourceProtocolTests(unittest.TestCase):
    def test_default_is_30_train_8_validation_and_20_independent_test(self):
        split=cohort.make_splits(cases())['imagechd_fold0']
        self.assertEqual((len(split['train']),len(split['val']),len(split['target'])),(30,8,20))
        self.assertEqual(split['selector'],[])
        self.assertEqual(split['unlabeled'],[])
        self.assertEqual(split['fingerprint_fit_ids'],split['train'])
        self.assertEqual(split['holdout'],'G')
        self.assertEqual(split['evaluation_scope'],'held_out_center')
        self.assertNotEqual(cohort.PROJECT_ID,'vista3d_wholeheart_nnunet_v1')

    def test_every_eligible_case_has_one_disjoint_role_for_every_fold(self):
        all_ids={c['case_id'] for c in cases() if c['status']=='eligible'}
        expected={'imagechd_fold0':(30,8,20),'imagechd_fold1':(30,8,20),'imagechd_fold2':(32,8,18)}
        for fold,split in cohort.make_splits(cases()).items():
            with self.subTest(fold=fold):
                parts=[set(split[k]) for k in ('train','val','target')]
                self.assertEqual(tuple(map(len,parts)),expected[fold])
                self.assertEqual(set.union(*parts),all_ids)
                self.assertFalse(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2])
                self.assertEqual(set(cohort.label_access_ids(split)),parts[0]|parts[1])
                self.assertFalse(set(cohort.label_access_ids(split))&parts[2])
                self.assertTrue(all(cid.startswith({'A':'Case1','B':'Case2','G':'Case7'}[split['holdout']]) for cid in parts[2]))
        self.assertEqual(cohort.make_splits(cases()),cohort.make_splits(list(reversed(cases()))))

    def test_tampered_roles_and_target_leakage_are_rejected(self):
        split=cohort.make_splits(cases())['imagechd_fold0']
        for cid in (split['val'][0],split['target'][0],'Case2009'):
            altered=copy.deepcopy(split);altered['train'][0]=cid
            with self.assertRaises(ValueError):cohort.validate_split(altered,cases())

    def test_old_preprocessing_is_unselectable_even_if_seven_hashes_match(self):
        import task_inputs
        from preparation_identity import current
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            write_json(work/'00_admin/server_config.json',dict(project_id='vista3d_wholeheart_nnunet_v1',protocol_sha256='old'))
            write_json(work/'04_data/manifests/splits_v1.json',dict(folds={'imagechd_fold0':dict(dataset_id=813)}))
            write_json(work/'04_data/derived/nnUNet_preprocessed/Dataset813_imagechd_fold0/server_preparation_signature.json',dict(implementation_sha256=current()))
            with patch('task_runtime.load_plan',return_value=dict(workspace=str(work),folds=['imagechd_fold0'])):
                hint=task_inputs.source_compatibility(work)
            self.assertIsNotNone(hint)
            self.assertIn('协议',hint)


if __name__=='__main__':unittest.main()
