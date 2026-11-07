from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset1282_analysis import xu_pipeline as pipeline
from dataset1282_analysis.contracts import atomic_json, read_json
from dataset1282_analysis.xu_pipeline_stages import Stage


class PipelineRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name).resolve()
        self.args = SimpleNamespace(resume=False, profile='smoke', technical_subset=True, crop64=None,
                                    crop128=None, reviews=None, stop_after='audit-roi')
        self.contract = {'implementation':{}, 'profile':'smoke'}
        self.stages = [Stage('localize', ('example-localize',), ('candidate',)),
                       Stage('audit-roi', ('audit-roi',), ('boxes.json',))]
        self.calls = []
        for target, options in [
                ('dataset1282_analysis.xu_pipeline._sources', dict(return_value={})),
                ('dataset1282_analysis.xu_pipeline_stages.iter_stages', dict(side_effect=lambda *a: iter(self.stages))),
                ('dataset1282_analysis.xu_pipeline._execute', dict(side_effect=self.execute))]:
            mock = patch(target, **options)
            mock.start()
            self.addCleanup(mock.stop)
        self.failure = None

    def execute(self, script, argv, name, out, owner, args):
        self.calls.append((name, list(argv)))
        directory = out/name
        directory.mkdir(exist_ok=True)
        if name=='localize':
            (directory/'candidate').write_text('frozen')
        elif self.failure:
            if self.failure=='reject':
                atomic_json(directory/'localization_qc.json', dict(accepted=False))
            (directory/'partial').write_text('incomplete')
            raise pipeline.StageError('injected stage failure')
        else:
            atomic_json(directory/'boxes.json', dict(accepted=True))

    def run_pipeline(self):
        pipeline._run_locked(None,self.args,self.out,{},self.contract)

    def test_fresh_only_failure_keeps_evidence_and_resumes_after_verified_stage(self):
        self.failure = 'transient'
        with self.assertRaises(pipeline.StageError):
            self.run_pipeline()
        self.assertEqual(read_json(self.out/'status.json')['state'],'failed')
        self.assertFalse((self.out/'receipts/audit-roi.json').exists())
        self.args.resume = True
        self.failure = None
        self.run_pipeline()
        self.assertEqual([c[0] for c in self.calls],['localize','audit-roi','audit-roi'])
        self.assertEqual(len(list((self.out/'attempts').glob('audit-roi-*/partial'))),1)
        self.assertEqual(read_json(self.out/'status.json')['state'],'stopped_after_stage')

    def test_rejected_roi_cannot_release_or_silently_rerun(self):
        self.failure = 'reject'
        with self.assertRaises(pipeline.StageError):
            self.run_pipeline()
        self.args.resume = True
        with self.assertRaisesRegex(pipeline.StageError,'already failed'):
            self.run_pipeline()
        self.assertEqual(len(self.calls),2)
        self.assertEqual(read_json(self.out/'status.json')['state'],'roi_rejected')
        self.assertFalse((self.out/'audit-roi/boxes.json').exists())
        atomic_json(self.out/'audit-roi/localization_qc.json', dict(accepted=True))
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            self.run_pipeline()

    def test_changed_artifact_and_configuration_refused_before_downstream(self):
        self.args.stop_after = 'localize'
        self.run_pipeline()
        self.args.resume = True
        self.args.stop_after = 'audit-roi'
        with self.assertRaisesRegex(ValueError,'identical inputs'):
            pipeline._run_locked(None,self.args,self.out,{},dict(self.contract, seed=123))
        (self.out/'localize/candidate').write_text('other')
        with self.assertRaisesRegex(ValueError,'artifacts changed'):
            self.run_pipeline()
        self.assertEqual(len(self.calls),1)

    def test_resume_requires_ownership_and_uses_native_checkpoint(self):
        self.args.resume = True
        with self.assertRaisesRegex(ValueError,'resume requires'):
            self.run_pipeline()
        self.args.resume = False
        self.stages[1] = Stage('audit-roi', ('example',), ('boxes.json',), 'partial')
        self.failure = 'transient'
        with self.assertRaises(pipeline.StageError):
            self.run_pipeline()
        self.args.resume = True
        self.failure = None
        self.run_pipeline()
        self.assertEqual(self.calls[-1],('audit-roi',['example','--resume']))
        self.assertFalse((self.out/'attempts').exists())

    def test_unowned_directory_is_not_adopted_or_moved(self):
        self.args.stop_after = 'localize'
        self.run_pipeline()
        occupied = self.out/'audit-roi'
        occupied.mkdir()
        (occupied/'unrelated').write_text('user file')
        self.args.resume = True
        self.args.stop_after = 'audit-roi'
        with self.assertRaisesRegex(ValueError,'Unowned'):
            self.run_pipeline()
        self.assertTrue((occupied/'unrelated').exists())

    def test_fusion_cannot_substitute_another_valid_checkpoint_in_same_run(self):
        stage = Stage('four-way', ('fuse',), ('job.json',))
        atomic_json(self.out/'job.json', dict(checkpoints=dict(all128='other-seed')))
        records = {'all128':{'files':{'best.pt':'selected-seed'}}}
        with self.assertRaisesRegex(ValueError,'consumed different all128'):
            pipeline._check_used_weights(stage,self.out,records,self.contract)

    def test_device_aliases_generate_the_same_stage_arguments(self):
        for device in ('cuda','cuda:0'):
            args = SimpleNamespace(prepared=self.out/'prepared', output=self.out/'experiment',
                                   device=device, crop64=None, crop128=None, reviews=None,
                                   threads=2, technical_subset=True,prefetch=False,wait_seconds=0)
            with (patch('dataset1282_analysis.xu_geometry.PreparedCache'),
                  patch.object(pipeline,'_contract',return_value={'device':{'device':'cuda:0'}}),
                  patch.object(pipeline,'_run_locked') as run):
                pipeline._run(args)
                self.assertEqual(run.call_args.args[1].device,'cuda:0')
