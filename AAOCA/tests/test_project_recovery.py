import tempfile
import unittest
from pathlib import Path
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from test_converters import dicom
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_model import ProjectOptions
from patient_pipeline.project_ingest import ingest
from patient_pipeline.project_identity import resolve_identity
from patient_pipeline.project_process import process_units
from patient_pipeline.runtime import Control


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.src=self.root/'src';self.src.mkdir()
        self.s=ProjectStore(self.root/'out');self.study=generate_uid();self.series=generate_uid()
    def tearDown(self):self.s.close();self.tmp.cleanup()
    def ct(self,name,z,series=None,phase=None):
        path=self.src/name
        d=dicom(path,np.ones((2,3),dtype=np.int16),position=(0,0,z),study=self.study,series=series or self.series)
        if phase is not None:d.TemporalPositionIdentifier=phase;d.save_as(path,enforce_file_format=True)
        return path
    def run_project(self,**opts):
        o=ProjectOptions(**opts);c=Control();ingest(self.s,[self.src],c);resolve_identity(self.s,o,c);return process_units(self.s,o,c)
    def units(self,kind):return [u for _,u in self.s.items('units') if u['kind']==kind]
    def test_one_bad_series_does_not_block_other_ct_or_us(self):
        self.ct('a.dcm',0);self.ct('b.dcm',1);self.ct('single.dcm',0,generate_uid())
        d=dicom(self.src/'us.dcm',np.ones((2,2,3),dtype=np.uint8),modality='US',frames=2)
        d.FrameTimeVector=[20,20];d.save_as(self.src/'us.dcm',enforce_file_format=True)
        self.run_project()
        self.assertEqual(sorted(u['status'] for u in self.units('CT')),['failed','success'])
        self.assertEqual(self.units('US')[0]['status'],'failed')
    def test_failed_new_copy_preserves_previous_verified_ct(self):
        self.ct('a.dcm',0);self.ct('b.dcm',1);self.run_project()
        first=next(u for u in self.units('CT') if u['status']=='success');old_path=first['outputs'][0]
        self.ct('gap.dcm',4);self.run_project()
        after=self.units('CT')[0]
        self.assertEqual(after['outputs'][0],old_path);self.assertTrue(self.s.path(old_path).is_file())
        self.assertEqual(after['last_attempt']['status'],'failed')
    def test_offline_sources_do_not_erase_or_reconvert_success(self):
        self.ct('a.dcm',0);self.ct('b.dcm',1);self.run_project()
        old=self.units('CT')[0]['outputs'];self.src.rename(self.root/'offline');self.src.mkdir()
        self.run_project()
        self.assertEqual(self.units('CT')[0]['outputs'],old)
        self.assertNotIn('last_attempt',self.units('CT')[0])
    def test_complete_copy_repair_is_optional_and_proves_identity(self):
        p=self.ct('a.dcm',0);self.ct('b.dcm',1)
        d=pydicom.dcmread(p);d.PixelData=b'';d.save_as(self.src/'empty_copy.dcm',enforce_file_format=True)
        self.run_project();self.assertEqual(self.units('CT')[0]['status'],'failed')
        self.run_project(recover_copies=True)
        u=self.units('CT')[0];self.assertEqual(u['status'],'success');self.assertTrue(u['repairs'])
    def test_different_complete_pixels_same_sop_stay_conflict(self):
        p=self.ct('a.dcm',0);self.ct('b.dcm',1)
        d=pydicom.dcmread(p);d.PixelData=np.zeros((2,3),dtype=np.int16).tobytes();d.save_as(self.src/'different.dcm',enforce_file_format=True)
        self.run_project(recover_copies=True)
        self.assertEqual(self.units('CT')[0]['status'],'failed')
    def test_explicit_phases_split_but_unequal_grids_rejected(self):
        for phase in [1,2]:
            for z in range(3):self.ct(f'p{phase}_{z}.dcm',z,phase=phase)
        self.run_project();self.assertEqual(self.units('CT')[0]['status'],'failed')
        self.run_project(split_phases=True)
        u=self.units('CT')[0];self.assertEqual(u['status'],'success');self.assertEqual(len(u['outputs']),2)
        self.ct('p2_2.dcm',3,phase=2);self.run_project(split_phases=True)
        self.assertEqual(self.units('CT')[0]['last_attempt']['status'],'failed')
    def test_changed_content_at_same_path_is_not_silently_skipped(self):
        self.ct('a.dcm',0);p=self.ct('b.dcm',1);self.run_project()
        old=self.units('CT')[0]['outputs']
        d=pydicom.dcmread(p);d.PixelData=np.full((2,3),5,dtype=np.int16).tobytes();d.save_as(p,enforce_file_format=True)
        self.run_project()
        self.assertNotEqual(self.units('CT')[0]['outputs'],old)
    def test_missing_output_is_not_reported_as_preserved_success(self):
        self.ct('a.dcm',0);self.ct('b.dcm',1);self.run_project()
        self.s.path(self.units('CT')[0]['outputs'][0]).unlink()
        self.src.rename(self.root/'offline');self.src.mkdir();self.run_project()
        self.assertNotEqual(self.units('CT')[0]['status'],'success')
    def test_moved_sources_with_hash_proof_can_extend_offline_ct(self):
        self.ct('a.dcm',0);self.ct('b.dcm',1);self.run_project()
        self.src.rename(self.root/'old');self.src.mkdir()
        import shutil
        for p in (self.root/'old').iterdir():shutil.copy2(p,self.src/p.name)
        # Old paths are offline; source catalog points to the earlier src path. Move it to a new batch.
        new=self.root/'batch2';self.src.rename(new);self.src=new
        self.ct('c.dcm',2);self.run_project()
        u=self.units('CT')[0]
        self.assertEqual(u['slices'],3);self.assertNotIn('last_attempt',u)
