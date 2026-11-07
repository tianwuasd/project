import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from dataset1282_analysis import xu_pair_probe as probe
from dataset1282_analysis.contracts import atomic_json, read_json


class PairProbeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)/'fresh'
        self.cards = [dict(device=f'cuda:{i}', uuid=f'card-{i}') for i in (0,1)]

    def exercise(self, mode):
        reservation = {'threads':2,'host_per_stage_bytes':1048576}
        def plan(stage, args, out, children, concurrency, purpose, checked):
            self.assertIs(args.wave_reservation, reservation)
            self.assertEqual((args.profile,args.resource_policy,concurrency,purpose), ('paper','v2',2,'launch'))
            if mode == 'no_budget':
                raise probe.ResourceUnavailable('insufficient GPU budget')
            return stage.argv[-1], {}

        def execute(script, argv, size, out, owner, args):
            self.assertEqual((script,argv), ('xu_pair_probe.py',['_worker',out,size]))
            i = probe.SIZES.index(size)
            card = self.cards[i]
            self.assertEqual(args.device, card['device'])
            if mode == 'failure' and i:
                raise RuntimeError('injected worker failure')
            start, end = (12,18) if i == 0 else (20 if mode == 'sequential' else 14,22)
            row = dict(size=size, device=card['device'], uuid=card['uuid'],
                       request=read_json(out/'request.json')['nonce'], pid=101+i, peer_pid=102-i,
                       ready_ns=10, probe_start_ns=start, probe_end_ns=end)
            atomic_json(out/size/'execution.json', row)
            side = int(size[2:])
            atomic_json(out/size/'capacity.json', dict(stage=size,device=card['device'],profile='paper',
                        input='synthetic',resource_policy='v2',state='passed',optimizer='Adam',
                        dtype='float32',shape=[4,1,side,side,side],optimizer_steps=1,loss=1.,
                        head_max_abs_update=0. if mode=='no_update' else 1.e-4,
                        peak_cuda_allocated_bytes=1024))

        with (patch.object(probe,'_selected',return_value=self.cards),
              patch.object(probe,'reserve_pair',return_value=reservation) as reserve,
              patch.object(probe,'plan_stage',side_effect=plan),
              patch.object(probe,'_execute',side_effect=execute),
              contextlib.redirect_stdout(io.StringIO())):
            code = probe.main(['--devices','cuda:0,cuda:1','--output',str(self.out),'--threads','2'])
            reserve.assert_called_once()
        return code, read_json(self.out/'result.json')

    def test_concurrent_formal_64_and_128_require_two_positive_updates(self):
        code, result = self.exercise('ok')
        self.assertEqual((code,result['state']), (0,'passed'))
        self.assertEqual(len(result['branches']),2)
        self.assertAlmostEqual(result['capacity_interval_overlap_seconds'], 4e-9, places=12)

    def test_sequential_execution_cannot_be_certified(self):
        code, result = self.exercise('sequential')
        self.assertEqual((code,result['state']), (1,'failed'))
        self.assertIn('did not overlap',result['reason'])

    def test_no_admission_runs_no_worker(self):
        code, result = self.exercise('no_budget')
        self.assertEqual((code,result['state']), (1,'not_admitted'))
        self.assertFalse((self.out/'3d64').exists())

    def test_child_failure_preserves_failure_evidence(self):
        code, result = self.exercise('failure')
        self.assertEqual((code,result['state']), (1,'failed'))
        self.assertIn('injected worker',result['reason'])

    def test_zero_weight_delta_cannot_be_certified(self):
        code, result = self.exercise('no_update')
        self.assertEqual((code,result['state']), (1,'failed'))
        self.assertIn('measured formal CUDA Adam update',result['reason'])

    def test_reusing_output_cannot_overwrite_evidence(self):
        self.exercise('ok')
        before = (self.out/'result.json').read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(probe.main(['--devices','cuda:0,cuda:1','--output',str(self.out)]),1)
        self.assertEqual((self.out/'result.json').read_bytes(), before)

    def test_git_checkout_outside_source_root_is_still_private_output_violation(self):
        (self.out.parent/'.git').mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(probe.main(['--devices','cuda:0,cuda:1','--output',str(self.out)]),1)
        self.assertFalse(self.out.exists())

    def test_worker_requires_matching_scope_before_using_device(self):
        card = dict(device='cuda:5',uuid='GPU-assigned')
        mapping = [dict(parent_device='cuda:5',device='cuda:0',uuid='GPU-assigned')]
        env = {'XU_GPU_ISOLATION':json.dumps(mapping)}
        self.assertEqual(probe.worker_card(card,env),dict(card,device='cuda:0'))
        with self.assertRaises(ValueError):
            probe.worker_card(dict(card,uuid='GPU-other'),env)
