"""Exercise the real rank entry before any GPU/model work, with fake devices.

The server can select GPU 5 for rank 1. A CPU test cannot execute NCCL, but
it can reject the missing device_id that makes PyTorch 2.15 bind to rank 1.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chd_stunet.training.distributed_engine import run_rank


class StopAtCommunicator(RuntimeError):
    """End the startup test without allocating CUDA or constructing a model."""


class StartupTests(unittest.TestCase):
    def setUp(self):
        scratch=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=scratch)
        self.root=Path(self.temporary.name)

    def tearDown(self):self.temporary.cleanup()

    def invoke(self,ordinals,rank,*,uuid_mismatch=False):
        output=self.root/('r'+str(rank)+'-'+str(len(ordinals)))
        members=['GPU-test-'+str(i) for i in ordinals]
        job=dict(output=str(output),members=members,ordinals=ordinals,
                 generation='current',rendezvous=(self.root/'rdzv').as_uri())
        path=self.root/'job.json';path.write_text(json.dumps(job),encoding='utf-8')
        device=torch.device('cuda',ordinals[rank])
        admission={'uuid':'GPU-wrong' if uuid_mismatch else members[rank]}
        with patch('chd_stunet.training.distributed_engine.admit_device',return_value=(device,admission)), \
             patch('chd_stunet.training.distributed_engine.dist.init_process_group',side_effect=StopAtCommunicator('communicator stopped')) as init, \
             patch('chd_stunet.training.distributed_engine.dist.is_initialized',return_value=False), \
             patch('chd_stunet.training.distributed_engine.torch.set_num_threads'), \
             patch('chd_stunet.training.distributed_engine.torch.use_deterministic_algorithms'), \
             patch('chd_stunet.training.distributed_engine.prepare_contract',side_effect=AssertionError('No model/input work during this test')):
            with self.assertRaises(ValueError if uuid_mismatch else StopAtCommunicator):run_rank(path,rank)
        return output,device,init

    def test_sparse_permuted_visible_ordinals_bind_actual_device(self):
        for ordinals in ([0,5,7,6],[3,2,1,0]):
            for rank in range(len(ordinals)):
                with self.subTest(ordinals=ordinals,rank=rank):
                    _,device,init=self.invoke(ordinals,rank)
                    self.assertEqual(init.call_args.args[0],'nccl')
                    self.assertEqual(init.call_args.kwargs.get('device_id'),device)
                    self.assertEqual(init.call_args.kwargs['rank'],rank)

    def test_one_rank_retains_gloo_without_cuda_binding(self):
        _,_,init=self.invoke([5],0)
        self.assertEqual(init.call_args.args[0],'gloo')
        self.assertNotIn('device_id',init.call_args.kwargs)

    def test_communicator_failure_writes_rank_phase_and_device(self):
        output,_,_=self.invoke([0,5,7,6],1)
        row=json.loads((output/'control/fatal-current-1.json').read_text(encoding='utf-8'))
        self.assertEqual((row['rank'],row['stage'],row['device']),(1,'process_group','cuda:5'))
        self.assertIn('communicator stopped',row['traceback'])
        events=[json.loads(line) for line in (output/'ranks/rank_1/logs/events.jsonl').read_text(encoding='utf-8').splitlines()]
        binding=next(row for row in events if row['event']=='parallel.device')
        self.assertEqual((binding['device'],binding['uuid']),('cuda:5','GPU-test-5'))

    def test_uuid_mismatch_refuses_before_group_and_records_failure(self):
        output,_,init=self.invoke([0,5],1,uuid_mismatch=True)
        init.assert_not_called()
        row=json.loads((output/'control/fatal-current-1.json').read_text(encoding='utf-8'))
        self.assertEqual(row['stage'],'gpu_admission')
        self.assertIn('UUID changed',row['error'])


if __name__=='__main__':unittest.main()
