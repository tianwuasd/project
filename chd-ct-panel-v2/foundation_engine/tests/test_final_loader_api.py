"""Pinned real Blosc slicing admission with unchanged upstream crop/augmentation."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('nnunetv2'),'requires pinned nnU-Net CPU API')
class LazyBloscAdmission(unittest.TestCase):
    def test_actual_lazy_slices_hold_slot_and_transforms_release_it(self):
        import numpy as np
        import torch
        import blosc2
        import pickle
        from read_admission import create_gate,gate_scope,_LOCAL,admitted_call
        from nnunetv2.training.dataloading import data_loader
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        from nnunet_loader_adapter import configured_loader_class
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=np.arange(1*12*12*12,dtype=np.float32).reshape(1,12,12,12)
            seg=(data%2).astype(np.int8)
            for suffix,array in [('.b2nd',data),('_seg.b2nd',seg)]:
                blosc2.asarray(array,urlpath=str(root/('case'+suffix)),mode='w',cparams={'nthreads':1})
            with (root/'case.pkl').open('wb') as stream:pickle.dump({'class_locations':{1:np.argwhere(seg==1)}},stream)
            dataset=nnUNetDatasetBlosc2(str(root),identifiers=['case'])
            lazy=dataset.load_case('case')[0];self.assertIsInstance(lazy,blosc2.NDArray)
            labels=LabelManager({'background':0,'foreground':1},regions_class_order=None)
            reads=[];transforms=[];original=data_loader.crop_and_pad_nd
            def crop(*args,**kwargs):
                reads.append(getattr(_LOCAL,'depth',0));return original(*args,**kwargs)
            def transform(**batch):transforms.append(getattr(_LOCAL,'depth',0));return batch
            with patch.object(data_loader,'crop_and_pad_nd',crop):cls=configured_loader_class()
            adapted=cls(dataset,2,[8,8,8],[8,8,8],labels,transforms=transform,worker_threads=1)
            upstream=data_loader.nnUNetDataLoader(dataset,2,[8,8,8],[8,8,8],labels,transforms=transform)
            with gate_scope(create_gate(1)):
                np.random.seed(42);actual=next(adapted)
            np.random.seed(42);expected=next(upstream)
            self.assertTrue(reads);self.assertTrue(all(v==1 for v in reads));self.assertTrue(all(v==0 for v in transforms))
            np.testing.assert_array_equal(actual['keys'],expected['keys'])
            self.assertTrue(torch.equal(actual['data'],expected['data']));self.assertTrue(torch.equal(actual['target'],expected['target']))

if __name__=='__main__':unittest.main()
