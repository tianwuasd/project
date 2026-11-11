from pathlib import Path
from contextlib import ExitStack
import os
import json
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from dataset1282_analysis import xu_capacity
from dataset1282_analysis.contracts import read_json
from dataset1282_analysis.resources import GIB


class CapacityTests(unittest.TestCase):
    def test_group_probe_demands_whole_layout_and_records_each_peak(self):
        names = ['cuda:2','cuda:0','cuda:1']
        stats = {'cuda:2':2,'cuda:0':1,'cuda:1':1}
        def step(stage,device,supervision,record):
            record['distribution'] = {'cases_by_device':dict(stats)}
            return 2.0,0.1
        limits = dict(policy='v2',host_budget_bytes=24*GIB,per_device={
            n:{'allocator_budget_bytes':24*GIB} for n in names})
        with TemporaryDirectory() as out, ExitStack() as m:
            m.enter_context(patch.object(xu_capacity,'snapshot',return_value={'available_memory_bytes':48*GIB}))
            m.enter_context(patch.object(xu_capacity,'apply_env_limit',return_value=limits))
            m.enter_context(patch.object(xu_capacity,'_step_3d',side_effect=step))
            for key,value in {'is_available':True,'mem_get_info':(40*GIB,48*GIB),
                    'get_device_name':'test','get_device_properties':SimpleNamespace(uuid='test'),
                    'set_device':None,'set_per_process_memory_fraction':None,'reset_peak_memory_stats':None,
                    'max_memory_allocated':GIB}.items():
                m.enter_context(patch.object(torch.cuda,key,return_value=value))
            m.enter_context(patch.object(torch.cuda,'max_memory_reserved',side_effect=lambda n:(1+names.index(str(n)))*GIB))
            verified = xu_capacity.probe('3d128',names[0],'marginal',out,policy='v2')
            self.assertEqual(verified['state'],'passed',verified.get('reason'))
            self.assertEqual(verified['peak_cuda_reserved_bytes'],GIB)
            self.assertEqual([r['peak_cuda_reserved_bytes'] for r in verified['per_device'].values()],
                             [GIB,2*GIB,3*GIB])
            stats['cuda:0'] = 0
            mismatch = xu_capacity.probe('3d128',names[0],'marginal',out,policy='v2')
            self.assertEqual(mismatch['state'],'failed')
            self.assertIn('promised stage allocation',mismatch['reason'])
            limits['per_device']['cuda:1']['allocator_budget_bytes'] = GIB
            denied = xu_capacity.probe('3d128',names[0],'marginal',out,policy='v2')
            self.assertEqual(denied['state'],'not_admitted')
            self.assertEqual(denied['per_device']['cuda:1']['reason'],'below_gpu_budget')

    def test_stage_device_demands_respect_chunks_and_primary_optimizer(self):
        names = ['cuda:2','cuda:0','cuda:1']
        spec = xu_capacity.specification('3d128')
        values = xu_capacity.device_demands('3d128',names,spec)
        self.assertEqual(list(values),names)
        self.assertLess(values[names[0]],spec['estimated_working_bytes'])
        self.assertGreater(values[names[0]],values[names[1]])
        self.assertEqual(values[names[1]],values[names[2]])
        more = [f'cuda:{i}' for i in range(8)]
        blood = xu_capacity.device_demands('blood-encoder',more)
        self.assertEqual([k for k in blood if blood[k] > 0],more[:4])
        frames = xu_capacity.device_demands('blood-sequence',more)
        self.assertEqual([k for k in frames if frames[k] > 0],more[:7])
        self.assertEqual(xu_capacity.device_demands('3d128',names[:1],spec)[names[0]],
                         spec['estimated_working_bytes'])

    def test_blood_cpu_grouping_keeps_order_global_loss_and_update(self):
        from dataset1282_analysis.xu_blood import BloodForward, _objective
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.width = 1
                self.weight = torch.nn.Parameter(torch.tensor(1.3))
            def features(self,x):
                return self.weight*x
            def forward(self,x):
                y = self.features(x)
                return torch.cat([y,-y,.25*y],1)
        class Sequence(Encoder):
            def forward(self,frames):
                return super().forward(sum((i+1)*x/28 for i,x in enumerate(frames)))
        def one(temporal,grouped):
            enc, seq = Encoder().eval(), Sequence()
            shape = (1,7,8,8) if temporal else (4,1,8,8)
            x = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)/64
            y = (torch.arange(len(x)*64).reshape(len(x),8,8)%3).long()
            y[:,0,:] = -1
            worker = BloodForward(enc,'cpu',temporal,4) if grouped else None
            if worker is not None:
                logits = worker(x,seq)
            elif temporal:
                with torch.no_grad():
                    frames = [enc.features(x[:,i:i+1]) for i in range(7)]
                logits = seq(frames)
            else:
                logits = enc(x)
            loss = _objective(logits,y)
            model = seq if temporal else enc
            opt = torch.optim.Adam(model.parameters(),lr=.002)
            loss.backward()
            if temporal:
                self.assertIsNone(enc.weight.grad)
            opt.step()
            return logits.detach(),loss.detach(),model.weight.grad,model.weight.detach(),worker.distribution.stats if worker else None
        for temporal in (False,True):
            with self.subTest(temporal=temporal):
                reference = one(temporal,False)
                actual = one(temporal,True)
                for left,right in zip(reference[:4],actual[:4]):
                    torch.testing.assert_close(left,right)
                self.assertEqual(list(actual[4]['cases_by_device'].values()),
                                 [7] if temporal else [4])
    def test_resource_limits_use_stage_share_and_policy_reserve_once(self):
        spec = xu_capacity.specification('3d128')
        self.assertGreater(spec['estimated_host_bytes'],0)
        self.assertEqual(xu_capacity.admission(spec, 64*GIB, 50*GIB, total=80*GIB,
            policy='v2',host_limit=GIB-1)[0], 'below_host_budget')
        reason, facts = xu_capacity.admission(spec,64*GIB,50*GIB,total=80*GIB,
            policy='v2',host_limit=8*GIB,cuda_limit=38*GIB)
        self.assertIsNone(reason)
        self.assertEqual(facts['cuda_allocator_budget_bytes'],38*GIB)
        self.assertEqual(facts['host_budget_bytes'],8*GIB)

    def test_frozen_encoder_weights_only_and_previous_v2_exact_compatibility(self):
        from dataset1282_analysis import xu_blood
        from dataset1282_analysis.xu_blood_models import SliceUNet
        spec = xu_capacity.specification('blood-sequence')
        with torch.device('meta'):
            plane = 512*512*SliceUNet('paper').width*4
        self.assertEqual(spec['estimated_working_bytes'],
            16*spec['trainable_parameters'] + 4*spec['other_resident_parameters'] + 96*plane)
        expected = dict(method='xu-blood-v2',source=xu_blood._source(),supervision='complete',
                        train=['a'],development=['b'])
        old = dict(expected,source=dict(xu_blood._PREVIOUS_V2_SOURCE))
        self.assertTrue(xu_blood._matches(old,expected))
        self.assertFalse(xu_blood._matches(dict(old,supervision='marginal'),expected))
        self.assertFalse(xu_blood._matches(
            dict(old,source=dict(old['source'],extra=expected['source']['xu_blood.py'])),expected))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            xu_blood._matches(dict(old,source=dict(old['source'],extra='x')),expected)

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
