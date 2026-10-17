import tempfile
import unittest
from pathlib import Path

from patient_pipeline.project_store import ProjectStore


class StoreTests(unittest.TestCase):
    def test_append_retains_old_catalog_and_replaces_same_source(self):
        with tempfile.TemporaryDirectory() as d:
            with ProjectStore(d) as s:
                s.put_source({'key':'one','kind':'DICOM','pid':'001','stamp':'v1'})
                s.put_source({'key':'two','kind':'PDF','pid':'001','stamp':'v2'})
                s.put_source({'key':'one','kind':'DICOM','pid':'002','stamp':'v3'})
            with ProjectStore(d) as s:
                self.assertEqual(len(list(s.sources())),2)
                self.assertEqual(s.source('one')['pid'],'002')
                self.assertEqual(len(list(s.sources(pid='001'))),1)

    def test_failed_attempt_preserves_success_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            with ProjectStore(d) as s:
                s.save_unit('ct1',{'kind':'CT','pid':'001','status':'success',
                                 'signature':'a','outputs':['keep.nii.gz']})
                s.save_unit('ct1',{'kind':'CT','pid':'001','status':'failed',
                                 'signature':'b','error':'new copy incomplete'})
                u=s.unit('ct1')
                self.assertEqual(u['status'],'success')
                self.assertEqual(u['outputs'],['keep.nii.gz'])
                self.assertEqual(u['last_attempt']['status'],'failed')
                self.assertEqual(len(s.history('ct1')),2)

    def test_no_unknown_existing_output_adoption(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'foreign.txt').write_text('do not adopt')
            with self.assertRaises(ValueError):
                ProjectStore(d)
