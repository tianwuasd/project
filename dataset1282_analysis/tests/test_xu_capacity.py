from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from dataset1282_analysis import xu_capacity
from dataset1282_analysis.contracts import read_json
from dataset1282_analysis.resources import GIB


class CapacityTests(unittest.TestCase):
    def test_formal_specs_count_current_models_without_allocating_weights(self):
        for stage, n, shape in (
            ('3d64',90461879,[4,1,64,64,64]), ('3d128',50957351,[4,1,128,128,128]),
            ('blood-encoder',497780707,[4,1,512,512]), ('blood-sequence',369347,[1,7,512,512]),
        ):
            with self.subTest(stage=stage):
                spec = xu_capacity.specification(stage)
                self.assertEqual(spec['shape'],shape)
                self.assertEqual(spec['trainable_parameters'],n)
                self.assertGreater(spec['estimated_working_bytes'],16*n)

    def test_reject_before_allocating_on_low_gpu_or_unknown_host(self):
        spec = xu_capacity.specification('3d128')
        reason, info = xu_capacity.admission(spec,64*GIB,8*GIB)
        self.assertEqual(reason,'below_gpu_budget')
        self.assertLess(info['cuda_allocator_budget_bytes'],info['cuda_free_bytes'])
        self.assertEqual(xu_capacity.admission(spec,None,64*GIB)[0],'below_host_budget')
        with TemporaryDirectory() as tmp, patch.object(xu_capacity,'snapshot',
                return_value={'available_memory_bytes':GIB}), patch.object(xu_capacity,'_step_3d') as step:
            rc = xu_capacity.main(['--stage','3d128','--device','cpu','--output',tmp])
            self.assertNotEqual(rc,0)
            result = read_json(Path(tmp)/'capacity.json')
            self.assertEqual(result['optimizer_steps'],0)
            self.assertEqual(result['state'],'not_admitted')
            step.assert_not_called()

    def test_cpu_entry_requires_real_optimizer_update(self):
        from dataset1282_analysis.xu_models import build_model

        with TemporaryDirectory() as tmp, patch.object(xu_capacity,'snapshot',
                return_value={'available_memory_bytes':256*GIB}), patch(
                'dataset1282_analysis.xu_models.build_model',
                side_effect=lambda name,profile: build_model(name,'smoke')):
            result = xu_capacity.probe('3d64','cpu','complete',tmp)
            self.assertEqual(result['state'],'passed',result.get('reason'))
            self.assertEqual(result['optimizer_steps'],1)
            self.assertGreater(result['head_max_abs_update'],0.)
            with patch.object(xu_capacity,'_step_3d',return_value=(1.,0.)):
                self.assertEqual(xu_capacity.probe('3d64','cpu','complete',tmp)['state'],'failed')
            self.assertEqual(read_json(Path(tmp)/'capacity.json')['state'],'failed')

    def test_blood_encoder_and_ordered_sequence_take_optimizer_steps(self):
        from dataset1282_analysis.xu_blood_models import SliceUNet

        with patch('dataset1282_analysis.xu_blood_models.SliceUNet',
                   side_effect=lambda profile: SliceUNet('smoke')):
            for mode in ('blood-encoder','blood-sequence'):
                with self.subTest(mode=mode):
                    with patch('dataset1282_analysis.xu_blood._batch',
                            return_value=(torch.randn(1,7 if mode=='blood-sequence' else 1,32,32),
                                          torch.randint(3,(1,32,32)))):
                        loss, change = xu_capacity._step_blood(mode,torch.device('cpu'))
                    self.assertGreater(loss,0.)
                    self.assertGreater(change,0.)
