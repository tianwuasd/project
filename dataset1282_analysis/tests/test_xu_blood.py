"""Geometry, leakage and both phases of the prepared 512-square blood branch."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dataset1282_analysis.xu_blood_grid import AxialViews
from dataset1282_analysis.xu_geometry import CaseGeometry

try:
    import torch
except ImportError:
    torch = None


class TinyPrepared:
    """Cache axes are X,Z,Y for this asymmetric synthetic scan."""
    request_digest = 'synthetic'
    scope = 'technical_subset'

    def __init__(self):
        self.geo = CaseGeometry((15,9,7),(0.5,1.5,2.),(2,0,1),((0,15),(0,7),(0,9)),(15,7,9))
        x,z,y = np.indices(self.geo.cache_shape)
        self.ct = ((x+y*10+z*100)/1000).astype(np.float32)
        self.label = ((x>=4)&(y>=4)).astype(np.int8)
        self.label[:,0,:]=-1
        self.supervision = {name:{'present_classes':list(range(1,11)),
                                  'unknown_classes':[], 'known_absent_classes':[]}
                            for name in ('a','b')}

    def geometry(self,name):
        return self.geo

    def case_ids(self,role):
        return ('a',) if role=='train' else ('b',)

    def read_image(self,name,box):
        return self.ct[tuple(slice(*p) for p in box)]


class TinyTargets:
    identity = 'unit-test-targets'

    def __init__(self, cache):
        self.cache=cache

    def has(self,name,kind):
        return True

    def read(self,name,kind,box):
        return self.cache.label[tuple(slice(*p) for p in box)]


class BloodGridTests(unittest.TestCase):
    def test_nontrivial_axes_physical_letterbox_and_nearest_inverse(self):
        cache=TinyPrepared()
        views=AxialViews(cache,'a',TinyTargets(cache))
        self.assertEqual(views.axes,(1,2,0))
        self.assertEqual(views.zyx,(7,9,15))
        self.assertEqual(views.scaled,(512,284))
        self.assertEqual(views.window(0),[0,0,0,0,1,2,3])
        self.assertEqual(views.window(6),[3,4,5,6,6,6,6])
        self.assertEqual(views.label(0).max(),-1)
        self.assertTrue((views.label(3)[:,0]==-1).all())
        zs,ys,xs=views.sampling(((1,13),(1,5),(2,8)),(12,4,6))
        got=np.stack([views.label(z)[np.ix_(ys,xs)] for z in zs]).transpose(views.permutation)
        np.testing.assert_array_equal(got,cache.label[1:13,1:5,2:8])
        observed=np.stack([views.image(z)[np.ix_(ys,xs)] for z in zs]).transpose(views.permutation)
        np.testing.assert_allclose(observed,cache.ct[1:13,1:5,2:8],atol=0.0002)


@unittest.skipUnless(torch is not None,'PyTorch research environment required')
class BloodTrainingTests(unittest.TestCase):
    def test_unknown_ventricle_excludes_development_and_blocks_formal_fit(self):
        from dataset1282_analysis.xu_blood import train
        cache=TinyPrepared()
        cache.supervision['b']={'present_classes':[1,3,4,5,6,7,8,9,10],
                                'unknown_classes':[2],'known_absent_classes':[]}
        class PartialTargets(TinyTargets):
            def has(self,name,kind):
                return name=='a'
        args=SimpleNamespace(targets=None,profile='paper',technical_unvalidated=False)
        with patch('dataset1282_analysis.xu_targets.TargetStore',return_value=PartialTargets(cache)), \
             patch('dataset1282_analysis.xu_targets.blood_ids',return_value=tuple(range(1,10))):
            with self.assertRaisesRegex(ValueError,'Require eligible'):
                train(cache,args,None)
            args.technical_unvalidated=True
            with self.assertRaisesRegex(ValueError,'restricted'):
                train(cache,args,None)

    def test_loss_ignores_padding_and_temporal_uses_both_directions(self):
        from dataset1282_analysis.xu_blood import _objective
        from dataset1282_analysis.xu_blood_models import BloodSequence
        torch.manual_seed(13)
        scores=torch.randn(1,3,3,4,requires_grad=True)
        y=torch.ones(1,3,4,dtype=torch.long)
        y[:,0,:]=-1
        _objective(scores,y).backward()
        self.assertEqual(scores.grad[:,:,0].abs().max().item(),0)
        frames=[torch.randn(1,2,8,8,requires_grad=True) for _ in range(7)]
        BloodSequence(2)(frames).sum().backward()
        for frame in frames:
            self.assertGreater(frame.grad.abs().max().item(),0)

    def test_train_resume_and_guarded_label_free_prediction(self):
        from dataset1282_analysis.xu_blood import train,load_blood_predictor
        cache=TinyPrepared()
        args=SimpleNamespace(targets=None,profile='smoke',technical_unvalidated=False,
                             seed=12,device='cpu',resume=False)
        saved_threads=torch.get_num_threads()
        torch.set_num_threads(min(2,saved_threads))
        self.addCleanup(torch.set_num_threads,saved_threads)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('dataset1282_analysis.xu_targets.TargetStore',return_value=TinyTargets(cache)), \
             patch('dataset1282_analysis.xu_targets.blood_ids',return_value=tuple(range(1,10))), \
             patch('dataset1282_analysis.xu_workflow.artifact_identity',return_value='artifacts'):
            out=Path(tmp)
            report=train(cache,args,out)
            self.assertEqual(report['selection'],'development')
            self.assertEqual(report['development'],1)
            self.assertIn('validation',report['history']['sequence'][0])
            predictor=load_blood_predictor(cache,out/'best.pt','cpu','smoke')
            pred=predictor('b',None,(3,2,4))
            self.assertEqual(pred.shape,(3,2,4))
            self.assertEqual(pred.dtype,np.uint8)
            args.resume=True
            self.assertEqual(train(cache,args,out)['history'],report['history'])
            with self.assertRaises(ValueError):
                load_blood_predictor(cache,out/'best.pt','cpu','paper')
            args.seed+=1
            with self.assertRaisesRegex(ValueError,'incompatible'):
                train(cache,args,out)
