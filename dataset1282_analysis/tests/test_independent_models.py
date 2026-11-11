import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from dataset1282_analysis.independent.prediction import read_mask, save_mask
from dataset1282_analysis.independent.sam_med3d import prompt_patch, restore_mask, validate_point


class IndependentModelTests(unittest.TestCase):
    def test_anisotropic_prompt_and_native_mask_round_trip(self):
        spacing = [.7,1.2,3.1]
        shape = (45,63,32)
        point = [11,39,23]
        image = np.indices(shape,dtype=np.float32).sum(0)+100.
        patch,center,scale,offset = prompt_patch(image,spacing,point)
        self.assertEqual(patch.shape,(128,128,128))
        np.testing.assert_allclose(offset+center*scale,point)
        mask = np.zeros(patch.shape,dtype=np.uint8)
        mask[62:67,62:67,62:67] = 1
        native = restore_mask(mask,shape,scale,offset)
        self.assertEqual(native.shape,shape)
        self.assertEqual(native[tuple(point)],1)
        self.assertEqual(native[0,0,0],0)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            row = dict(shape=shape,spacing=spacing)
            artifact = save_mask(output,'cases/test/1.nii.gz',native,row)
            np.testing.assert_array_equal(read_mask(output,artifact,row,binary=True),native)
            self.assertEqual(nib.load(output/artifact['path']).header.get_xyzt_units()[0],'mm')

    def test_out_of_grid_or_noninteger_prompt_is_rejected(self):
        for point in ([-1,1,1],[1,1,9],[1.0,1,1],[True,1,1]):
            with self.subTest(point=point),self.assertRaises(ValueError):
                validate_point(point,(9,9,9))

    def test_stunet_spacing_plan_preserves_eleven_channels(self):
        import torch
        from dataset1282_analysis.independent.stunet import network, topology

        torch.set_num_threads(2)
        shape = (32,32,32)
        strides = topology((.6,.4,.4),shape)
        model = network(strides,False).eval()
        with torch.inference_mode():
            self.assertEqual(model(torch.zeros((1,1,*shape))).shape,(1,11,*shape))
