from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from dataset1282_analysis import xu_inference_pool as pool
from dataset1282_analysis.contracts import read_json
from dataset1282_analysis.xu_geometry import CaseGeometry


class TinyCache:
    request_digest = 'synthetic'
    def geometry(self,name):
        return CaseGeometry((16,)*3,(1.,)*3,(0,1,2),((0,16),)*3,(16,)*3)
    def case_ids(self,role):
        return ('a','b','c') if role=='train' else ('d','e','f')
    def read_image(self,name,shape,box=None):
        arr = np.zeros(shape,dtype='uint8')
        r = shape[0]//4
        arr[r:3*r,r:3*r,r:3*r] = 1
        return arr
    def read_native_label(self,name):
        raise AssertionError('Accessed a reference during frozen prediction')


class InferencePoolTests(unittest.TestCase):
    def setUp(self):
        self.devices = tuple(f'cuda:{i}' for i in range(5))
        self.names = tuple('abcdefghijk')
        self.cache = TinyCache()

    def test_round_robin_uses_all_devices_and_returns_stable_order(self):
        plan = pool.CasePlan(self.names,self.devices,self.devices,None,None)
        first_cases = threading.Barrier(5,timeout=8)
        made = []
        def make(device):
            made.append(device)
            def predict(name):
                if name in self.names[:5]:
                    first_cases.wait()
                return device
            return predict
        with patch.object(pool,'_device_scope',return_value=nullcontext()):
            result = pool.map_cases(plan,make)
        self.assertCountEqual(made,self.devices)
        self.assertEqual(tuple(result),self.names)
        self.assertEqual(list(result.values()),[self.devices[i%5] for i in range(11)])

    def test_budget_and_case_count_bound_admission_before_any_worker_is_loaded(self):
        env = dict(XU_BRANCH_DEVICES=json.dumps(self.devices),XU_HOST_BUDGET_BYTES=str(4*1024**3))
        with (patch.dict(os.environ,env),
              patch('dataset1282_analysis.xu_resources.applied_host_budget',return_value=4*1024**3),
              patch('dataset1282_analysis.xu_resources.require_applied_group') as admission):
            self.assertEqual(pool.case_plan('cuda:0',self.names,self.cache).devices,self.devices[:2])
            self.assertEqual(pool.case_plan('cuda:0',['a'],self.cache).devices,self.devices[:1])
            admission.assert_called_with(list(self.devices))
            with patch('dataset1282_analysis.xu_resources.applied_host_budget',return_value=1):
                with self.assertRaisesRegex(ValueError,'Insufficient host'):
                    pool.case_plan('cuda:0',self.names,self.cache)
        with patch.dict(os.environ,env):
            with self.assertRaisesRegex(RuntimeError,'allocator limits'):
                pool.case_plan('cuda:0',self.names,self.cache)

    def test_locator_and_fusion_arrays_are_identical_with_serial_and_emulated_cuda_shards(self):
        from dataset1282_analysis.xu_fusion import _ensure_branch
        from dataset1282_analysis.xu_roi_workflow import localize
        args = SimpleNamespace(device='cpu',supervision='complete',profile='smoke',crop64='w64',
                               crop128='w128',padding_percent=20)
        names = self.cache.case_ids('development')
        env = dict(XU_BRANCH_DEVICES=json.dumps(self.devices),XU_HOST_BUDGET_BYTES=str(32*1024**3))
        def predictor(path,**kwargs):
            return lambda image:image, 'c'*64
        with (TemporaryDirectory() as tmp,
              patch.object(pool,'_device_scope',return_value=nullcontext()),
              patch('dataset1282_analysis.xu_resources.applied_host_budget',return_value=32*1024**3),
              patch('dataset1282_analysis.xu_resources.require_applied_group'),
              patch('dataset1282_analysis.xu_workflow.input_id',return_value='fixed'),
              patch('dataset1282_analysis.xu_workflow.frozen_predictor',side_effect=predictor)):
            roots = [Path(tmp)/'serial',Path(tmp)/'shards']
            checksums = []
            for i,root in enumerate(roots):
                root.mkdir()
                args.device = 'cuda:0' if i else 'cpu'
                with patch.dict(os.environ,env if i else {},clear=True):
                    localize(self.cache,args,root)
                    payload = read_json(root/'candidates.json')['payload']
                    plan = pool.case_plan(args.device,names,self.cache)
                    book = {'boxes':{n:payload['cases'][n]['box'] for n in names}}
                    checksums.append(_ensure_branch(self.cache,args,book,'crop64','w64','c'*64,
                                                    names,root/'branch',plan=plan))
            self.assertEqual((roots[0]/'candidates.json').read_bytes(),(roots[1]/'candidates.json').read_bytes())
            self.assertEqual(checksums[0],checksums[1])
            self.assertEqual(read_json(roots[1]/'inference-plan.json')['cases_per_device'],
                             {'cuda:0':2,'cuda:1':1,'cuda:2':1,'cuda:3':1,'cuda:4':1})

    def test_evaluation_is_stable_and_each_patient_prediction_precedes_its_reference(self):
        from dataset1282_analysis.xu_workflow import evaluate
        from dataset1282_analysis.contracts import sha256_file
        roles = ('a','b','c','d')
        seen = threading.local()
        class EvaluationCache(TinyCache):
            supervision = {n:dict(present_classes=[1],known_absent_classes=[],unknown_classes=list(range(2,11)))
                           for n in roles}
            def case_ids(self,role,complete_only=False):
                return () if complete_only else roles
            def read_image(self,name,shape,box=None):
                seen.name = name
                seen.predicted = False
                return super().read_image(name,shape,box)
            def read_native_label(self,name):
                if not seen.predicted or seen.name != name:
                    raise AssertionError('Used reference before predicting its own patient')
                return super().read_image(name,(16,)*3)
        def predict(image):
            seen.predicted = True
            return image
        env = dict(XU_BRANCH_DEVICES=json.dumps(self.devices))
        with (TemporaryDirectory() as tmp, patch.object(pool,'_device_scope',return_value=nullcontext()),
              patch('dataset1282_analysis.xu_resources.require_applied_group'),
              patch('dataset1282_analysis.xu_resources.applied_host_budget',return_value=32*1024**3),
              patch('dataset1282_analysis.xu_workflow.input_id',return_value='fixed')):
            base = Path(tmp)
            checkpoint = base/'model.pt'
            checkpoint.write_bytes(b'fixed test predictor')
            args = SimpleNamespace(device='cpu',supervision='marginal',profile='smoke',branch='crop64',
                                   boxes=None,checkpoint=checkpoint,save_predictions=True)
            for i,folder in enumerate(('serial','parallel')):
                directory = base/folder
                directory.mkdir()
                args.device = 'cuda:0' if i else 'cpu'
                with (patch.dict(os.environ,env if i else {},clear=True),
                      patch('dataset1282_analysis.xu_workflow.frozen_predictor',
                            return_value=(predict,sha256_file(checkpoint)))):
                    evaluate(EvaluationCache(),args,directory)
            for relative in ['evaluation.json',*(f'cases/{n}.nii.gz' for n in roles)]:
                self.assertEqual(sha256_file(base/'serial'/relative),sha256_file(base/'parallel'/relative))

    def test_fusion_assigns_a_resident_blood_predictor_to_each_active_case_shard(self):
        from dataset1282_analysis import xu_fusion as fusion
        from dataset1282_analysis.contracts import sha256_file
        names = self.cache.case_ids('development')
        loads, predictions = [], {}
        def blood_predictor(cache, checkpoint, device, **kwargs):
            loads.append(device)
            def predict(name,box,shape):
                predictions[name] = device
                return np.zeros(shape,dtype='uint8')
            return predict
        with (TemporaryDirectory() as tmp,
              patch.object(pool,'_device_scope',return_value=nullcontext()),
              patch('dataset1282_analysis.xu_resources.require_applied_group'),
              patch('dataset1282_analysis.xu_resources.applied_host_budget',return_value=32*1024**3),
              patch.dict(os.environ,XU_BRANCH_DEVICES=json.dumps(self.devices)),
              patch.object(fusion,'shard_identity',return_value={'synthetic':'case plan'}),
              patch('dataset1282_analysis.xu_workflow.artifact_identity',return_value='test'),
              patch('dataset1282_analysis.xu_workflow.input_id',return_value='fixture'),
              patch('dataset1282_analysis.xu_targets.anatomy',return_value={'MYO':1}),
              patch('dataset1282_analysis.xu_blood.load_blood_predictor',side_effect=blood_predictor)):
            base = Path(tmp)
            weights = base/'weights.bin'
            weights.write_bytes(b'weights')
            weight_hash = sha256_file(weights)
            book = dict(boxes={n:[[4,12]]*3 for n in names},
                        checkpoints={k:weight_hash for k in ('crop64','crop128')})
            args = SimpleNamespace(device='cuda:0',profile='smoke',supervision='marginal',boxes='book',
                    evaluate=False,resume=False,init64=None,init128=None,blood=weights,
                    **{k:weights for k in fusion.FINE_WEIGHTS})
            out = base/'fusion'
            out.mkdir()
            with (patch('dataset1282_analysis.xu_workflow.roi_book',return_value=book),
                  patch('dataset1282_analysis.xu_workflow.frozen_predictor',return_value=(lambda x:x,weight_hash))):
                fusion._fuse(self.cache,args,out)
            self.assertCountEqual(loads,self.devices[:len(names)])
            self.assertEqual(predictions,dict(zip(names,self.devices)))
            self.assertEqual(tuple(read_json(out/'complete.json')['cases']),names)
