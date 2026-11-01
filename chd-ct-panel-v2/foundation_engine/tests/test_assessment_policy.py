from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from assessment_policy import automatic_request, sample_scope, parse_formal_blocks, PREPROCESSORS
from test_cohort import cases
from cohort import make_splits


class AssessmentPolicyTests(unittest.TestCase):
    def test_only_primary_inputs_and_isolated_short_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw';raw.mkdir()
            profile=dict(name='test',runtime=str(Path(tmp)/'runtime'),results=str(Path(tmp)/'results'),data=str(raw),gpu_memory=20,conda='')
            value=automatic_request(profile,str(raw),'S1',kind='assessment')
            self.assertEqual(value['tasks'],['assessment'])
            self.assertEqual(value['job_kind'],'assessment')
            self.assertEqual(Path(value['results']),Path(tmp)/'results/assessments')
            self.assertEqual(value['gpu_policy'],'automatic')
            self.assertEqual(value['gpu_count'],0)
            self.assertFalse(value['allow_batch_growth'])
            self.assertEqual(value['global_batch_size'],4)
            self.assertEqual(value['max_global_batch_size'],4)
            self.assertEqual(value['assessment_budget']['train_steps'],12)
            self.assertEqual(value['assessment_budget']['measured_steps'],10)
            self.assertEqual(value['assessment_budget']['prediction_cases'],1)
            self.assertFalse(value['formal_training'])

    def test_formal_blocks_never_expose_environment_or_smoke(self):
        self.assertEqual(parse_formal_blocks('1,2,3'),['prepare','train','predict'])
        self.assertEqual(parse_formal_blocks('2'),['train'])
        self.assertEqual(parse_formal_blocks('预测'),['predict'])
        for value in ['smoke','environment','4','']:
            with self.assertRaises(ValueError):parse_formal_blocks(value)

    def test_source_sample_is_small_balanced_and_excludes_target_selector_and_unlabeled(self):
        values=cases();split=make_splits(values)['imagechd_fold0']
        before=repr(split)
        scope=sample_scope(values,split)
        self.assertEqual(len(scope['train_ids']),2)
        self.assertEqual(len(scope['validation_ids']),1)
        self.assertEqual(scope['prediction_ids'],scope['validation_ids'])
        self.assertEqual(len(scope['planning_fit_ids']),30)
        allowed=set(split['train']+split['val'])
        selected=set(scope['train_ids']+scope['validation_ids'])
        self.assertTrue(selected<=allowed)
        self.assertFalse(selected & set(split['target']+split['selector']+split['unlabeled']))
        by_id={c['case_id']:c for c in values}
        self.assertEqual(len({by_id[c]['group'] for c in scope['train_ids']}),2)
        self.assertEqual(repr(split),before)

    def test_invalid_primary_selection_is_rejected_before_submission(self):
        self.assertEqual(len(PREPROCESSORS),1)
        with tempfile.TemporaryDirectory() as tmp:
            profile=dict(runtime=str(Path(tmp)/'rt'),results=str(Path(tmp)/'out'),data=tmp,name='generic')
            with self.assertRaises(ValueError):automatic_request(profile,tmp,'unknown')
            with self.assertRaises(ValueError):automatic_request(profile,tmp,'S1',preprocessor='unknown')


if __name__=='__main__':unittest.main()
