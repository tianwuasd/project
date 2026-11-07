"""Epoch boundaries preserve optimizer/augmentation state and never release a partial stage."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dataset1282_analysis import xu_pipeline as pipeline
from dataset1282_analysis import xu_training as training
from dataset1282_analysis.contracts import atomic_json, read_json
from dataset1282_analysis.xu_models import BranchSpec
from dataset1282_analysis.xu_pipeline_stages import Stage


class EpochPauseTests(unittest.TestCase):
    def test_pause_resume_matches_uninterrupted_optimizer_weights_and_learning_rates(self):
        # Exercise the actual train/validation, loss, augmentation and Adam loops
        # with a tiny convolution; expensive paper architecture is checked separately.
        spec = BranchSpec(8, 2, 1, 2, 4, 15)
        def model(*_args):
            result = torch.nn.Conv3d(1, 11, 1)
            result.spec = spec
            return result
        grid = torch.arange(8**3).reshape(8, 8, 8)
        def sample(name, _branch):
            label = grid.remainder(11)
            if name in ('c', 'd', 'dev'):
                label = label.masked_fill(label == 3, 0)
            return grid.float()/grid.numel(), label
        options = dict(branch='crop64', profile='paper', supervision='marginal',
                       train_cases={'a':[], 'b':[], 'c':[3], 'd':[3]},
                       validation_cases={'dev':[3]}, load_sample=sample, data_identity='pause-test')
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp, patch.object(training, 'branch_spec', return_value=spec), \
                patch.object(training, '_training_model', side_effect=model):
            baseline, resumed = Path(tmp)/'baseline', Path(tmp)/'resumed'
            whole = training.train_branch(output_dir=baseline, **options)
            first = training.train_branch(output_dir=resumed, pause_after_epoch=1, **options)
            self.assertEqual((first['state'], first['epochs'], first['completed_epochs']), ('paused', 15, 1))
            second = training.train_branch(output_dir=resumed, pause_after_epoch=2, resume=True, **options)
            self.assertEqual((second['state'], second['completed_epochs']), ('paused', 2))
            self.assertEqual([h['learning_rate'] for h in second['history']], [2e-4, 2e-4])
            final = training.train_branch(output_dir=resumed, resume=True, **options)
            self.assertEqual((final['state'], final['completed_epochs']), ('complete', 15))
            columns = ('epoch', 'learning_rate', 'train_loss', 'validation_loss')
            self.assertEqual([[h[k] for k in columns] for h in whole['history']],
                             [[h[k] for k in columns] for h in final['history']])
            left = torch.load(baseline/'last.pt', weights_only=True)
            right = torch.load(resumed/'last.pt', weights_only=True)
            self.assertEqual(left['contract'], right['contract'])
            for key in left['model']:
                torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)
            self.assertEqual(left['optimizer']['param_groups'], right['optimizer']['param_groups'])
            for index, state in left['optimizer']['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, right['optimizer']['state'][index][key], rtol=0, atol=0)

    def test_both_serial_and_parallel_locators_pause_without_receipts_and_resume(self):
        for devices in (['cpu'], ['cuda:0', 'cuda:1']):
            with self.subTest(devices=devices), TemporaryDirectory() as tmp:
                out = Path(tmp).resolve()
                args = SimpleNamespace(resume=False, profile='paper', technical_subset=False,
                                       crop64=None, crop128=None, reviews=None, stop_after=None,
                                       device_names=devices, pause_after_epoch=1)
                stages = [Stage(name, ('train','--device',devices[0]), ('best.pt','training.json'), 'last.pt')
                          for name in ('crop64','crop128')]
                stages += [Stage('localize', ('localize',), ('candidates.json',))]
                calls = []
                def execute(_script, argv, name, *_args):
                    self.assertIn(name, ('crop64', 'crop128'), 'Partial weights reached localization')
                    calls.append((name, list(argv)))
                    folder = out/name
                    folder.mkdir(exist_ok=True)
                    for filename in ('best.pt', 'last.pt'):
                        (folder/filename).write_text('committed checkpoint')
                    pause = int(argv[argv.index('--pause-after-epoch')+1]) if '--pause-after-epoch' in argv else None
                    atomic_json(folder/'training.json', dict(state='paused' if pause else 'complete',
                                                            completed_epochs=pause or 15, epochs=15))
                with patch.object(pipeline, '_sources', return_value={}), \
                     patch.object(pipeline, '_capacity'), patch.object(pipeline, '_execute', side_effect=execute), \
                     patch('dataset1282_analysis.xu_pipeline_stages.iter_stages', side_effect=lambda *_:iter(stages)):
                    contract = dict(implementation={})
                    pipeline._run_locked(None,args,out,{},contract)
                    self.assertEqual({n for n,_ in calls}, {'crop64','crop128'})
                    self.assertEqual(read_json(out/'status.json')['state'], 'paused')
                    self.assertFalse((out/'receipts').exists())
                    self.assertFalse((out/'result.json').exists())
                    args.resume, args.pause_after_epoch = True, 2
                    pipeline._run_locked(None,args,out,{},contract)
                    self.assertEqual(read_json(out/'status.json')['paused']['crop128']['completed_epochs'], 2)
                    self.assertTrue(all('--resume' in argv for _,argv in calls[2:]))
                    args.pause_after_epoch, args.stop_after = None, 'crop128'
                    pipeline._run_locked(None,args,out,{},contract)
                    self.assertEqual(read_json(out/'status.json')['state'], 'stopped_after_stage')
                    for name in ('crop64','crop128'):
                        self.assertEqual(read_json(out/'receipts'/(name+'.json'))['state'], 'complete')
                    self.assertEqual(len(calls), 6)

    def test_only_paper_locators_accept_one_or_two_epoch_pause(self):
        for change in (dict(profile='smoke'), dict(branch='all64'), dict(pause_after_epoch=3)):
            options = dict(branch='crop64',profile='paper',pause_after_epoch=1,output_dir='unused',
                           train_cases={},validation_cases={},load_sample=None,data_identity='unused')
            with self.subTest(change=change), self.assertRaisesRegex(ValueError,'only paper locators'):
                training.train_branch(**(options | change))

    def test_prior_multi_gpu_contract_still_requires_exact_other_fingerprints(self):
        expected = dict(data_identity='data', source={'xu_training.py':'new', 'xu_models.py':'model',
                                                     'xu_supply.py':'supply','xu_distributed.py':'parallel'})
        old = dict(expected,source=dict(expected['source'], **{
            'xu_training.py':'54b970cb34279c6e94392d4632aa90a3173c4f3bf0026660ac1fa6a1111cd247'}))
        self.assertTrue(training._compatible_contract(old,expected))
        self.assertFalse(training._compatible_contract(dict(old,data_identity='other'),expected))
        changed = dict(old['source'], **{'xu_models.py':'other'})
        self.assertFalse(training._compatible_contract(dict(old,source=changed),expected))

    def test_locator_failure_is_not_reported_as_pause(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp).resolve()
            args = SimpleNamespace(resume=False, profile='paper', technical_subset=False,
                                   crop64=None, crop128=None, reviews=None, stop_after=None,
                                   device_names=['cpu'], pause_after_epoch=2)
            stages = [Stage(name, ('train','--device','cpu'), ('best.pt','training.json'), 'last.pt')
                      for name in ('crop64','crop128')]
            def execute(_script, _argv, name, *_args):
                if name == 'crop128':
                    raise pipeline.StageError('injected optimizer failure')
                folder = out/name
                folder.mkdir()
                for filename in ('best.pt','last.pt'):
                    (folder/filename).write_text('committed checkpoint')
                atomic_json(folder/'training.json', dict(state='paused',completed_epochs=2,epochs=15))
            with patch.object(pipeline, '_sources', return_value={}), \
                 patch.object(pipeline, '_capacity'), patch.object(pipeline, '_execute', side_effect=execute), \
                 patch('dataset1282_analysis.xu_pipeline_stages.iter_stages', side_effect=lambda *_:iter(stages)):
                with self.assertRaisesRegex(pipeline.StageError, 'optimizer failure'):
                    pipeline._run_locked(None,args,out,{},dict(implementation={}))
            status = read_json(out/'status.json')
            self.assertEqual((status['state'], status['stage'], status['finished']), ('failed','crop128',[]))
            self.assertEqual(status['paused']['crop64']['completed_epochs'], 2)
            self.assertFalse((out/'receipts').exists())

    def test_unrequested_incomplete_summary_cannot_be_committed(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp).resolve()
            args = SimpleNamespace(technical_subset=False, profile='paper')
            stage = Stage('crop64', ('train',), ('best.pt','training.json'), 'last.pt')
            def execute(*_args):
                atomic_json(out/'crop64/training.json',dict(state='paused',completed_epochs=2,epochs=15))
            with patch.object(pipeline, '_sources', return_value={}), \
                 patch.object(pipeline, '_capacity'), patch.object(pipeline, '_execute', side_effect=execute):
                with self.assertRaisesRegex(ValueError, 'Unexpected incomplete'):
                    pipeline._run_stage(stage,args,out,{},dict(implementation={}), 'contract',{}, {},1)
            self.assertFalse((out/'receipts').exists())


if __name__ == '__main__':
    unittest.main()
