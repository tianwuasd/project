from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dataset1282_analysis import xu_inference_pool as pool
from dataset1282_analysis import xu_inference_probe as probe
from dataset1282_analysis.contracts import read_json


class InferenceProbeTests(unittest.TestCase):
    def test_twelve_emulated_lanes_run_one_case_each_without_serial_replay(self):
        devices = [f'cuda:{i}' for i in range(12)]
        cards = [{'uuid':d} for d in devices]
        def load(device, model, data, barrier):
            def process(name):
                ready = time.monotonic_ns()
                barrier.wait(timeout=5)
                start = time.monotonic_ns()
                time.sleep(.07)
                return dict(case=name,device=device,uuid=device,ready_ns=ready,forward_start_ns=start,
                            cuda_completed_ns=time.monotonic_ns(),cuda_elapsed_ms=3.,finite=True)
            return process
        with (tempfile.TemporaryDirectory() as tmp,
              patch.dict(os.environ,XU_BRANCH_DEVICES=json.dumps(devices),XU_CUDA_ADMISSION='{}'),
              patch.object(probe,'apply_env_limit',return_value={'host_budget_bytes':32*1024**3}),
              patch('dataset1282_analysis.xu_resources.require_applied_group'),
              patch('dataset1282_analysis.xu_resources.applied_host_budget',return_value=32*1024**3),
              patch.object(pool,'_device_scope',return_value=nullcontext()),
              patch.object(probe,'shard_identity',return_value={'cards':cards}),
              patch.object(probe,'_inputs',return_value=(None,None)),
              patch.object(probe,'_load',side_effect=load)):
            data = probe.run_worker(Path(tmp),'cuda:0',devices)
            self.assertEqual(data,read_json(Path(tmp)/'inference.json'))
            self.assertEqual((data['state'],data['devices'],data['cases']),('passed',devices,12))
            self.assertEqual([r['device'] for r in data['forwards']],devices)
            self.assertGreater(data['host_forward_interval_overlap_seconds'],0)

    def test_missing_or_foreign_ticket_fails_without_loading_any_model(self):
        with (tempfile.TemporaryDirectory() as tmp,
              patch.dict(os.environ,{},clear=True),patch.object(probe,'_inputs') as inputs):
            data = probe.run_worker(Path(tmp),'cuda:0',['cuda:0','cuda:1'])
            inputs.assert_not_called()
            self.assertEqual(data['state'],'failed')
            self.assertIn('authorized',data['reason'])
            self.assertEqual(data,read_json(Path(tmp)/'inference.json'))

    def test_host_interval_and_cuda_event_are_both_required(self):
        devices = ['cuda:0','cuda:1']
        rows = [dict(device=d,uuid=d,finite=True,ready_ns=1,forward_start_ns=10+i*10,
                     cuda_completed_ns=15+i*10,cuda_elapsed_ms=.01) for i,d in enumerate(devices)]
        identity = {'cards':[{'uuid':d} for d in devices]}
        with self.assertRaisesRegex(ValueError,'do not overlap'):
            probe._overlap(rows,devices,identity)
        rows[1].update(forward_start_ns=12,cuda_elapsed_ms=0)
        with self.assertRaisesRegex(ValueError,'measured'):
            probe._overlap(rows,devices,identity)
