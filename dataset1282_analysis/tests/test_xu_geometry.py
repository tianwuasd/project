"""Synthetic landmarks verify planner-axis order, crop offsets and voxel centres."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dataset1282_analysis.contracts import atomic_json, digest, sha256_file
from dataset1282_analysis.xu_geometry import CaseGeometry, PreparedCache, prediction_box, _resize


class GeometryTests(unittest.TestCase):
    def test_asymmetric_crop_and_permutation_restores_xyz(self):
        native = np.zeros((9, 12, 7), dtype=np.uint8)
        native[2, 8, 4] = 6
        native[5, 1, 2] = 7
        transposed = native.T.transpose(1, 2, 0)
        cropped = transposed[1:10, 1:8, 2:6]
        # Unequal resampling factors make an accidental axis exchange visible.
        cached = np.repeat(np.repeat(np.repeat(cropped, 2, 0), 3, 1), 2, 2)
        geom = CaseGeometry(native.shape, (0.4, 0.6, 1.5), (1,2,0),
                            ((1,10), (1,8), (2,6)), cached.shape)
        np.testing.assert_array_equal(geom.restore_label(cached), native)
        box = ((14,16),(3,6),(4,6))  # projected native (2,8,4)
        self.assertEqual(geom.cache_box_to_native(box), ((2,3),(8,9),(4,5)))
        restored = geom.restore_label(np.full((5,7,3), 6, dtype='u1'), box)
        np.testing.assert_array_equal(np.argwhere(restored), [[2,8,4]])

    def test_half_pixel_sampling_and_strict_subvoxel_support(self):
        values = np.arange(4, dtype='f4').reshape((1,1,4))
        np.testing.assert_allclose(_resize(values,(1,1,2),1).ravel(), [0.5,2.5])
        labels = values.astype('i2') * 2
        np.testing.assert_array_equal(_resize(labels,(1,1,2),0).ravel(), [2,6])
        geom = CaseGeometry((3,1,1),(0.3,1.,2.),(0,1,2),((0,1),(0,1),(0,3)),(1,1,12))
        box = ((0,1),(0,1),(0,1))
        self.assertEqual(geom.cache_box_to_native(box), ((0,1),(0,1),(0,1)))
        self.assertFalse(geom.restore_label(np.ones((1,2,3),'u1'),box).any())

    def test_localizer_coverage_corners_single_branch_and_empty(self):
        a = np.zeros((2,3,4),'u1')
        b = np.zeros((4,6,8),'u1')
        a[0,0,0] = 1
        self.assertEqual(prediction_box(a,b,(7,10,13)), ((0,4),(0,4),(0,4)))
        b[-1,-1,-1] = 10
        self.assertEqual(prediction_box(a,b,(7,10,13)), ((0,7),(0,10),(0,13)))
        with self.assertRaisesRegex(ValueError,'empty'):
            prediction_box(a*0,b*0,(7,10,13))
        with self.assertRaises(ValueError):
            prediction_box(a.astype(float),b,(7,10,13))

    def test_invalid_geometry_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError,'permutation'):
            CaseGeometry((3,4,5),(1,1,1),(0,0,1),((0,3),(0,3),(0,3)),(2,2,2))
        with self.assertRaisesRegex(ValueError,'box'):
            CaseGeometry((3,4,5),(1,1,1),(0,1,2),((0,6),(0,4),(0,3)),(2,2,2))


def make_cache(root, roles=('train','development'), fit_cases=('a',), scope='technical_subset'):
    names = ('a','b')
    supervision = {}
    records = {}
    cohort = {'cases':{}}
    for idx,(name,role) in enumerate(zip(names,roles)):
        records[name] = dict(role=role,known_absent=[],shape=[4,3,2],spacing=[1.,2.,3.])
        present = list(range(1,11-idx))
        unknown = sorted(set(range(1,11))-set(present))
        supervision[name] = dict(present_classes=present,unknown_classes=unknown,
                                 known_absent_classes=[],ambiguous_background_allowed=[0]+unknown)
        cohort['cases'][name] = dict(role=role,patient=name,included=True,index_scan=True)
    dataset = root/'preprocessed/Dataset282_CHDSource'
    request = dict(scope=scope,method='HU-marginal-v1',nnunetv2='2.8.1',staged_extension='.nii',
                   cases=records,cohort_digest=digest(cohort))
    req_id = digest(request)
    plans = dict(image_reader_writer='NibabelIOWithReorient',transpose_forward=[0,1,2],
                 transpose_backward=[0,1,2],configurations={'3d_fullres':{'data_identifier':'cached'}})
    for name,value in {'partial_supervision.json':supervision,'splits_final.json':[{'train':['a'],'val':['b']}],
                       'nnUNetPlans.json':plans,'dataset.json':{}}.items():
        atomic_json(dataset/name,value)
    files = {p.name:sha256_file(p) for p in dataset.glob('*.json')}
    fit = dict(data_identifier='cached',fit_cases=list(fit_cases),plans_sha256=files['nnUNetPlans.json'])
    for item,value in [('request',request),('cohort',cohort),('fit',fit)]:
        atomic_json(root/(item+'.json'),value)
    atomic_json(root/'commits/fit.json',dict(job={'id':'fit'},contract=req_id,
                                            artifacts={'fit.json':sha256_file(root/'fit.json')}))
    atomic_json(dataset/'training_contract.json',dict(request_digest=req_id,artifacts=files))
    atomic_json(root/'complete.json',dict(state='completed',request_digest=req_id,scope=scope,fit=fit,
                coordinate_frame='local_array_excel_mm',train_cases=1,development_cases=1))
    return dataset


class CacheTests(unittest.TestCase):
    def test_frozen_roles_and_complete_only_without_volume_access(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root)
            with self.assertRaisesRegex(ValueError,'explicit admission'):
                PreparedCache(root)
            cache = PreparedCache(root,allow_technical_subset=True)
            self.assertEqual(cache.case_ids('train',complete_only=True), ('a',))
            self.assertEqual(cache.case_ids('development'), ('b',))
            self.assertEqual(cache.case_ids('development',complete_only=True), ())
            with self.assertRaises(ValueError):
                cache.case_ids('source_test')

    def test_rejects_held_out_case_in_fit_or_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root,fit_cases=('a','b'))
            with self.assertRaisesRegex(ValueError,'Fit, split'):
                PreparedCache(root,allow_technical_subset=True)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root,roles=('train','source_test'))
            with self.assertRaisesRegex(ValueError,'eligible source'):
                PreparedCache(root,allow_technical_subset=True)

    def test_changed_split_is_rejected_before_reading_arrays(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = make_cache(root)
            atomic_json(dest/'splits_final.json',[{'train':['a','b'],'val':[]}])
            with self.assertRaisesRegex(ValueError,'Changed preprocessing artifact'):
                PreparedCache(root,allow_technical_subset=True)


if __name__=='__main__':
    unittest.main()
