"""Native nnU-Net/Blosc2 API checks, inverse metadata, and refusal safety."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class PreprocessingIOTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'outputs/tests-tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=scratch)
        self.folder=Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(scratch.resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_bounded_locations_never_samples_ignore_as_annotated(self):
        from chd_stunet.preprocessing.stunet_s import bounded_locations
        seg=np.array([0,0,1,1,7,8,8,-1],dtype=np.int8).reshape(1,2,2,2)
        groups=[1,7,[-1,0,1,2,3,4,5,6,7]]
        locations=bounded_locations(seg,groups)
        self.assertEqual(len(locations[1]),2)
        annotated=locations[tuple(groups[-1])]
        self.assertEqual(len(annotated),6)
        self.assertEqual(annotated.shape[1],4)
        for coord in annotated:
            self.assertEqual(coord[0],0)
            self.assertNotEqual(seg[tuple(coord)],8)

    def test_affine_changes_only_derived_column_lengths(self):
        from chd_stunet.preprocessing.stunet_s import corrected_affine
        original=np.array([[0,-1,0,5],[1,0,0,6],[0,0,1,7],[0,0,0,1]],dtype=float)
        result=corrected_affine(original,[.5,.6,.75])
        np.testing.assert_array_equal(original[:3,3],result[:3,3])
        np.testing.assert_allclose(np.linalg.norm(result[:3,:3],axis=0),[.5,.6,.75])
        self.assertEqual(original[1,0],1)

    def test_four_column_locations_work_with_native_foreground_bbox(self):
        from types import SimpleNamespace
        from chd_stunet.preprocessing.stunet_s import bounded_locations
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        seg=np.zeros((1,8,10,12),dtype=np.int8)
        seg[0,4,5,6]=1
        key=tuple([-1]+list(range(8)))
        locations=bounded_locations(seg,[1,list(key)])
        loader=SimpleNamespace(need_to_pad=np.zeros(3,dtype=int),patch_size=np.array([4]*3),
                               has_ignore=True,annotated_classes_key=key)
        lower,upper=nnUNetDataLoader.get_bbox(loader,seg.shape[1:],True,locations)
        self.assertTrue(all(a<=v<b for a,v,b in zip(lower,[4,5,6],upper)))

    def test_refused_resume_keeps_every_old_file_byte(self):
        from chd_stunet.preprocessing.run import check_resume_before_writing
        old=self.folder/'old';old.mkdir()
        (old/'signature.json').write_text('{"identity":"old"}',encoding='utf-8')
        (old/'status.json').write_bytes(b'old status')
        before={p.name:p.read_bytes() for p in old.iterdir()}
        with self.assertRaisesRegex(ValueError,'Resume contract differs'):
            check_resume_before_writing(old,dict(identity='new'),[])
        self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})

    def test_completion_requires_all_three_file_hashes(self):
        from chd_stunet.preprocessing.run import cache_valid
        from chd_stunet.observability.runlog import write_json
        root=self.folder/'fold';(root/'completion').mkdir(parents=True)
        write_json(root/'completion/a.json',dict(contract='x',source_hashes={},file_sha256={}))
        self.assertFalse(cache_valid(root,'a','x',{}))

    def test_cache_rejects_a_changed_annotation_contract(self):
        from chd_stunet.preprocessing.run import cache_valid
        from chd_stunet.preprocessing.annotations import make_annotation
        from chd_stunet.data.analysis import sha256_file
        from chd_stunet.observability.runlog import write_json
        root=self.folder/'fold';(root/'data').mkdir(parents=True)
        files={}
        for suffix in ('.b2nd','_seg.b2nd','.pkl'):
            path=root/'data'/('a'+suffix);path.write_bytes(b'placeholder')
            files[path.name]=sha256_file(path)
        annotation=make_annotation([5])
        write_json(root/'completion/a.json',dict(contract='x',source_hashes={},file_sha256=files,
                                               annotation=annotation))
        self.assertTrue(cache_valid(root,'a','x',{},annotation))
        self.assertFalse(cache_valid(root,'a','x',{},make_annotation([])))

    def test_two_processes_cannot_share_output_lock(self):
        from chd_stunet.observability.locking import run_lock
        with run_lock(self.folder/'lock'):
            with self.assertRaises(RuntimeError):
                with run_lock(self.folder/'lock'):
                    self.fail('second owner entered')

    def test_native_transform_full_roundtrip_preserves_source(self):
        self._assert_roundtrip()

    def test_partial_transform_keeps_six_labels_and_annotation_contract(self):
        self._assert_roundtrip(missing=(5,))

    def test_partial_resampling_changes_grid_without_inventing_myo(self):
        self._assert_roundtrip(missing=(5,),resample=True)

    def _assert_roundtrip(self, missing=(), resample=False):
        from chd_stunet.preprocessing.registry import get_method
        from chd_stunet.preprocessing.fingerprint import make_plan
        from chd_stunet.preprocessing.stunet_s import preprocess_case
        from chd_stunet.preprocessing.cohort import TARGETS
        from chd_stunet.preprocessing.annotations import make_annotation
        from chd_stunet.data.analysis import sha256_file
        from chd_stunet.observability.runlog import write_json
        import blosc2
        raw=self.folder/'raw';raw.mkdir()
        image=(np.arange(16*14*12).reshape(16,14,12)%100+20).astype(np.int16)
        label=np.zeros(image.shape,dtype=np.int16)
        for i in range(1,8):
            if i not in missing:
                label[i:i+1,2:12,2:10]=i
        label[10:12,3:10,2:10]=15
        pair=dict(case_id='synthetic',image=str(raw/'image.nii.gz'),label=str(raw/'label.nii.gz'))
        for kind,array in [('image',image),('label',label)]:
            nib.save(nib.Nifti1Image(array,np.eye(4)),pair[kind])
        before={k:sha256_file(pair[k]) for k in ('image','label')}
        method=get_method('stunet-s-partial' if missing else 'stunet-s')
        annotation=make_annotation(missing)
        target_spacing=.5 if resample else 1.
        row=dict(case_id='fit_main',roles={'0':'train'},spacing_zyx=[target_spacing]*3,cropped_shape_zyx=[12,14,16])
        _,plan=make_plan([row],{'fit_main':np.arange(20,120,dtype=float)},0,method)
        gate=self.folder/'gate.json';write_json(gate,dict(slots=1))
        fold=self.folder/'fold'
        job=dict(pair=pair,gate=str(gate),method=method,expected_hashes=before,spacing_xyz=[1,1,1],
                 spacing_evidence=dict(source='synthetic'),plan=plan,dataset_json=dict(labels=TARGETS),
                 foldroot=str(fold),contract='synthetic',annotation=annotation,
                 cohort_group='auxiliary' if missing else 'main',
                 roles={str(f): 'aux_train' if missing else 'train' for f in range(3)})
        result=preprocess_case(job)
        self.assertTrue(result['success'])
        self.assertIn(8,result['output_label_ids'])
        self.assertEqual(result['annotation'],annotation)
        self.assertFalse(any(p.name.endswith('.json') for p in (fold/'data').iterdir()))
        stored_image=blosc2.open(str(fold/'data/synthetic.b2nd'),mode='r')
        stats=plan['foreground_intensity_properties_per_channel']['0']
        expected_image=image.astype(np.float32)
        np.clip(expected_image,stats['percentile_00_5'],stats['percentile_99_5'],out=expected_image)
        expected_image-=stats['mean']
        expected_image/=max(stats['std'],1e-8)
        if resample:
            self.assertEqual(list(stored_image.shape),[1,24,28,32])
            self.assertTrue(np.isfinite(stored_image[...]).all())
        else:
            np.testing.assert_allclose(stored_image[...],expected_image.transpose(2,1,0)[None],rtol=1e-6,atol=1e-6)
        del stored_image
        stored=blosc2.open(str(fold/'data/synthetic_seg.b2nd'),mode='r')
        expected=label.copy();expected[expected>7]=8
        if resample:
            self.assertEqual(set(np.unique(stored[...])),set(np.unique(expected)))
        else:
            np.testing.assert_array_equal(stored[...],expected.transpose(2,1,0)[None])
        del stored
        with (fold/'data/synthetic.pkl').open('rb') as f:
            properties=pickle.load(f)
        self.assertEqual(list(properties['shape_before_cropping']),[12,14,16])
        self.assertEqual(properties['annotation'],annotation)
        self.assertEqual(properties['cohort_group'],job['cohort_group'])
        self.assertEqual(properties['roles'],job['roles'])
        if missing:
            self.assertEqual(len(properties['class_locations'][5]),0)
            self.assertNotIn(5,properties['foreground_sampling_class_ids'])
            self.assertEqual(properties['annotated_area_semantics'],'nonignore_region_with_ambiguous_background')
        self.assertEqual(before,{k:sha256_file(pair[k]) for k in before})


if __name__=='__main__':
    unittest.main()
