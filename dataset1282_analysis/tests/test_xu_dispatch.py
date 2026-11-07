from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset1282_analysis import xu_pipeline as pipeline
from dataset1282_analysis.contracts import read_json
from dataset1282_analysis.observability.status import operation_lock
from dataset1282_analysis.xu_dispatch import Children, run_wave, select_devices, waves
from dataset1282_analysis.xu_pipeline_stages import Stage


class BranchSchedulingTests(unittest.TestCase):
    def test_two_device_waves_stop_at_requested_boundary(self):
        def stages():
            for name in ('crop64','crop128','localize','audit-roi','all64','all128'):
                yield Stage(name, ('train','--device','cuda:0'), ('best.pt',))
        devices = ['cuda:0','cuda:1']
        groups = waves(stages(), devices, 'all64')
        first = next(groups)
        self.assertEqual([row.name for row in first], ['crop64','crop128'])
        self.assertEqual(first[1].argv[-1], devices[1])
        self.assertEqual([tuple(row.name for row in g) for g in groups],
                         [('localize',),('audit-roi',),('all64',),('all128',)])
        with self.assertRaisesRegex(ValueError,'dependency order'):
            list(waves(iter([next(stages())]), devices, None))

    def test_allocation_refuses_duplicate_physical_cards_and_accepts_more_than_two(self):
        with patch('dataset1282_analysis.xu_dispatch.device_identity') as identify:
            identify.side_effect = lambda name, _: dict(device=name,uuid='one')
            with self.assertRaisesRegex(ValueError,'distinct'):
                select_devices(None, 'cuda:0,cuda:1', None)
            identify.side_effect = lambda name, _: dict(device=name,uuid=name)
            self.assertEqual(len(select_devices(None, 'cuda:0,cuda:1,cuda:2', None)),3)

    def exercise_processes(self, fail):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            out = root/'run'
            with operation_lock(out, 'pair-probe', out) as owner:
                children = Children(out, owner)
                children.configure_launch(2)
                def worker(name):
                    args = SimpleNamespace(threads=1, children=children, overlay={},device=name)
                    peer = 'right' if name == 'left' else 'left'
                    mode = ('fail' if name == 'left' else 'wait') if fail else 'ok'
                    pipeline._execute('../tests/fixtures/xu_pair_child.py',[out,name,peer,mode],name,out,owner,args)
                    return name
                if fail:
                    before = time.monotonic()
                    with self.assertRaisesRegex(pipeline.StageError,'exited 7'):
                        run_wave(('left','right'), worker, children)
                    self.assertLess(time.monotonic()-before,20)
                    self.assertFalse((out/'right.done').exists())
                else:
                    self.assertEqual(run_wave(('left','right'),worker,children),['left','right'])
                    self.assertTrue(all((out/(name+'.done')).is_file() for name in ('left','right')))
                self.assertEqual(read_json(out/'.runtime/operation.lock')['workers'], [])
                self.assertEqual(len(list((out/'logs').glob('*.log'))), 2)

    def test_both_real_child_processes_run_before_the_barrier(self):
        self.exercise_processes(False)

    def test_failure_reaps_the_owned_sibling(self):
        self.exercise_processes(True)

    def test_paper_pair_probes_each_size_on_its_training_device_and_rechecks_both(self):
        from threading import Barrier
        from dataset1282_analysis.contracts import atomic_json
        shared = {'lanes':2}
        barrier = Barrier(2)
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp).resolve()
            args = SimpleNamespace(resume=False, profile='paper', supervision='marginal', technical_subset=False, crop64=None,
                                   crop128=None, reviews=None, stop_after='crop128', device='cuda:0',
                                   device_names=['cuda:0','cuda:1'], resource_policy='v2')
            stages = [Stage(n, ('train','--device','cuda:0'), ('best.pt',)) for n in ('crop64','crop128')]
            def plan(stage, branch, _out, _children, concurrent, phase, checked):
                device = stage.argv[-1]
                self.assertIs(branch.wave_reservation, shared)
                self.assertEqual(concurrent, 2)
                if phase == 'launch':
                    self.assertEqual(checked[device, '3d'+stage.name[4:]], 123)
                events.append((stage.name, phase, device))
                return device, {}
            def execute(script, argv, name, _out, _owner, branch):
                if script == 'xu_capacity.py':
                    self.assertEqual(argv[argv.index('--device')+1], branch.device)
                    self.assertEqual(argv[argv.index('--policy')+1], 'v2')
                    path = Path(argv[argv.index('--output')+1])
                    atomic_json(path/'capacity.json',dict(state='passed',optimizer_steps=1,peak_cuda_reserved_bytes=123))
                else:
                    barrier.wait(timeout=5)
                    (out/name).mkdir()
                    (out/name/'best.pt').write_text('trained')
            with (patch.object(pipeline,'reserve_pair',return_value=shared) as reserve,
                  patch.object(pipeline,'plan_stage',side_effect=plan),
                  patch.object(pipeline,'_execute',side_effect=execute),
                  patch.object(pipeline,'_sources',return_value={}),
                  patch('dataset1282_analysis.xu_pipeline_stages.iter_stages',side_effect=lambda *_:iter(stages))):
                pipeline._run_locked(None,args,out,{},dict(implementation={},devices=args.device_names))
            reserve.assert_called_once()
            self.assertCountEqual(events,[(n,p,d) for n,d in zip(('crop64','crop128'),args.device_names)
                                         for p in ('admit','launch')])
            self.assertEqual(len(list((out/'capacity').glob('*/capacity.json'))),2)

    def test_completed_sibling_is_verified_when_resuming_failed_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp).resolve()
            args=SimpleNamespace(resume=False, profile='smoke',technical_subset=True,
                                 crop64=None, crop128=None, reviews=None, stop_after='crop128',
                                 device_names=['cuda:0','cuda:1'], device='cuda:0')
            stages = [Stage(n, ('train','--device','cuda:0'), ('best.pt',)) for n in ('crop64','crop128')]
            calls=[]
            def execute(_script, argv, name, *_args):
                calls.append((name,argv[-1]))
                target=out/name
                target.mkdir(exist_ok=True)
                if name=='crop128' and not args.resume:
                    limit=time.monotonic()+5
                    while not (out/'receipts/crop64.json').exists() and time.monotonic()<limit:
                        time.sleep(.01)
                    raise pipeline.StageError('injected right-branch failure')
                (target/'best.pt').write_text(name)
            with (patch.object(pipeline,'_execute',side_effect=execute),
                  patch.object(pipeline,'_sources',return_value={}),
                  patch('dataset1282_analysis.xu_pipeline_stages.iter_stages',side_effect=lambda *_:iter(stages))):
                with self.assertRaisesRegex(pipeline.StageError,'injected'):
                    pipeline._run_locked(None,args,out,{},dict(implementation={}))
                status=read_json(out/'status.json')
                self.assertEqual(status['finished'],['crop64'])
                self.assertEqual(status['stage'],'crop128')
                args.resume=True
                pipeline._run_locked(None,args,out,{},dict(implementation={}))
                self.assertEqual(calls.count(('crop64','cuda:0')),1)
                self.assertEqual(calls.count(('crop128','cuda:1')),2)
                self.assertEqual(read_json(out/'status.json')['state'],'stopped_after_stage')
