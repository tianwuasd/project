from pathlib import Path
import copy
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class TrainingDataTests(unittest.TestCase):
    def test_coordinate_validation_decodes_each_chunk_only_once(self):
        from chd_stunet.training.data import verify_coordinate_labels
        class Array:
            shape=(1,6,6,6);chunks=(1,3,3,3)
            def __init__(self):
                self.data=np.zeros(self.shape,dtype=np.int8);self.data[0,2,2,2]=1;self.reads=0
            def __getitem__(self,key):self.reads+=1;return self.data[key]
        array=Array();coordinates=np.argwhere(array.data>=0).astype(np.int32)
        properties={'class_locations':{tuple([-1]+list(range(8))):coordinates,
                                       1:np.array([[0,2,2,2]],dtype=np.int32)}}
        verify_coordinate_labels(array,properties)
        self.assertEqual(array.reads,8)
        properties['class_locations'][1]=np.array([[0,1,1,1]],dtype=np.int32)
        with self.assertRaisesRegex(ValueError,'segmentation mismatch'):verify_coordinate_labels(array,properties)

    def test_legacy_coordinates_are_read_only_and_unknown_producer_refused(self):
        from chd_stunet.training.data import normalize_locations
        source={'class_locations':{1:np.array([[1,2,3]],dtype=np.int32),5:[]}}
        result,adapted=normalize_locations(source,[1,5,6,7],True)
        self.assertTrue(adapted)
        np.testing.assert_array_equal(result['class_locations'][1],[[0,1,2,3]])
        self.assertEqual(source['class_locations'][1].shape,(1,3))
        with self.assertRaises(ValueError):normalize_locations(source,[1,5,6,7],False)
        for coordinates in ([[1,1,2,3]],[[0,99,2,3]],[[0,-1,2,3]],[[0.,1.,2.,3.]]):
            with self.subTest(coords=coordinates),self.assertRaises(ValueError):
                normalize_locations({'class_locations':{1:np.asarray(coordinates)}},[1,5,6,7],False)

    def test_new_coordinates_work_with_actual_native_bbox(self):
        from chd_stunet.preprocessing.stunet_s import bounded_locations
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        seg=np.zeros((1,8,10,12),dtype=np.int8);seg[0,4,5,6]=1
        key=tuple([-1]+list(range(8)))
        locations=bounded_locations(seg,[1,list(key)])
        fake=SimpleNamespace(need_to_pad=np.zeros(3,dtype=int),patch_size=np.array([4]*3),
                             has_ignore=True,annotated_classes_key=key)
        lo,hi=nnUNetDataLoader.get_bbox(fake,seg.shape[1:],True,locations)
        self.assertTrue(all(a<=v<b for a,v,b in zip(lo,[4,5,6],hi)))

    def test_blur_backend_is_stable_after_transform_reconstruction(self):
        from chd_stunet.training.data import disable_wall_clock_benchmark
        from chd_stunet.acceleration.training_cpu import deterministic_cpu_rng
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
        from batchgeneratorsv2.transforms.utils.random import RandomTransform
        results=[]
        for _ in range(2):
            blur=GaussianBlurTransform(blur_sigma=(.7,1.),p_per_channel=1,benchmark=True)
            wrapper=RandomTransform(blur,apply_probability=1)
            disable_wall_clock_benchmark(wrapper)
            self.assertFalse(blur.benchmark)
            with deterministic_cpu_rng(412):
                results.append(wrapper(image=torch.arange(1*12*13*14,dtype=torch.float32).reshape(1,12,13,14))['image'])
        torch.testing.assert_close(results[0],results[1],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
