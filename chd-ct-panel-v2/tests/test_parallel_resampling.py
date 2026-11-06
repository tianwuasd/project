"""Native interpolation equivalence across output partitions and boundaries."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage
from scipy.ndimage import _nd_image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class ParallelResamplingTests(unittest.TestCase):
    def test_values_exact_for_native_modes_orders_and_noncontiguous_inputs(self):
        from chd_stunet.acceleration.scipy_parallel import parallel_zooms
        original=_nd_image.zoom_shift;rng=np.random.default_rng(512);count=0
        for shape in ((3,5,7),(1,9,5),(9,7),(17,13,11)):
            for order in (0,1,3):
                for mode in ('nearest','reflect','mirror','grid-constant'):
                    for grid in (False,True):
                        x=rng.normal(size=shape);factors=[1.7,.6,1.3][:len(shape)]
                        expected=ndimage.zoom(x,factors,order=order,mode=mode,grid_mode=grid)
                        with parallel_zooms(3,min_voxels=0) as evidence:
                            actual=ndimage.zoom(x,factors,order=order,mode=mode,grid_mode=grid)
                        np.testing.assert_array_equal(actual,expected)
                        self.assertIs(_nd_image.zoom_shift,original)
                        self.assertEqual(evidence['applied_threads'],3);count+=1
        self.assertEqual(count,96)
        x=rng.normal(size=(12,9,7)).transpose(2,0,1)
        for dtype in (np.float32,np.float64):
            for prefilter in (True,False):
                expected=ndimage.zoom(x.astype(dtype),[.8,1.4,1.2],order=3,mode='nearest',prefilter=prefilter)
                with parallel_zooms(4,min_voxels=0):
                    actual=ndimage.zoom(x.astype(dtype),[.8,1.4,1.2],order=3,mode='nearest',prefilter=prefilter)
                np.testing.assert_array_equal(actual,expected)

    def test_probe_unknown_version_and_exception_restore(self):
        from chd_stunet.acceleration.scipy_parallel import parallel_zooms
        original=_nd_image.zoom_shift
        with patch('scipy.__version__','unreviewed'):
            with parallel_zooms(4) as evidence:
                self.assertEqual(evidence['applied_threads'],1)
                self.assertIn('version',evidence['reason'])
                self.assertIs(_nd_image.zoom_shift,original)
        with self.assertRaisesRegex(ValueError,'test failure'):
            with parallel_zooms(2):raise ValueError('test failure')
        self.assertIs(_nd_image.zoom_shift,original)
        for invalid in (0,-1,True):
            with self.assertRaises(ValueError):
                with parallel_zooms(invalid):pass

    def test_shared_output_bounded_queue_and_real_progress(self):
        from chd_stunet.acceleration.scipy_parallel import parallel_zooms
        rng=np.random.default_rng(54);x=rng.normal(size=(11,13,17));progress=[]
        expected=ndimage.zoom(x,[1.7,.8,1.3],order=3,mode='nearest',grid_mode=True)
        with parallel_zooms(3,min_voxels=0,progress=lambda done,total:progress.append((done,total))) as evidence:
            actual=ndimage.zoom(x,[1.7,.8,1.3],order=3,mode='nearest',grid_mode=True)
        np.testing.assert_array_equal(actual,expected)
        self.assertGreater(evidence['completed_tiles'],3)
        self.assertLessEqual(evidence['peak_pending'],3)
        self.assertEqual(progress[-1],(actual.size,actual.size))
        self.assertEqual(sorted(d for d,t in progress),[d for d,t in progress])
        self.assertEqual(evidence['additional_full_volume_buffers'],0)


if __name__=='__main__':unittest.main()
