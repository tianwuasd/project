import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
import pymupdf
from pydicom.uid import generate_uid
from test_converters import dicom
from run_project import run_project
from patient_pipeline.project_model import ProjectOptions


class EndToEndTests(unittest.TestCase):
    def test_three_batches_then_offline_report(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=r/'batch1';a.mkdir();out=r/'out'
            dicom(a/'us.dcm',np.ones((2,3),dtype=np.uint8),modality='US')
            run_project([a],out)
            read=lambda:json.loads((out/'项目数据快照.json').read_text(encoding='utf-8'))
            self.assertEqual(read()['patients'][0]['category'],'05_缺PDF和CT')
            a.rename(r/'old_offline');b=r/'batch2';b.mkdir();study=generate_uid();series=generate_uid()
            for z in range(3):dicom(b/f'{z}.dcm',np.ones((2,3),dtype=np.int16),position=(0,0,z),study=study,series=series)
            run_project([b],out);self.assertEqual(read()['patients'][0]['category'],'02_缺PDF')
            b.rename(r/'ct_offline');c=r/'batch3';c.mkdir()
            with pymupdf.open() as doc:
                doc.new_page().insert_text((40,40),'Patient ID: TEST_ONLY');doc.save(c/'report.pdf')
            run_project([c],out);self.assertEqual(read()['patients'][0]['category'],'01_三类结果齐全')
            run_project([],out,ProjectOptions(action='report'))
            self.assertEqual(read()['overview']['CT']['success'],1)
            self.assertEqual(read()['overview']['US']['success'],1)
            self.assertEqual(len(list((out/'report_history').iterdir())),4)
    def test_unknown_file_and_patient_filter_are_visible(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();out=root/'out'
            (src/'bad.dcm').write_bytes(b'bad')
            dicom(src/'us.dcm',np.ones((2,3),dtype=np.uint8),modality='US')
            run_project([src],out)
            snapshot=json.loads((out/'项目数据快照.json').read_text(encoding='utf-8'))
            self.assertEqual(snapshot['overview']['US']['success'],1)
            self.assertTrue(any(i['code']=='DICOM_READ_FAILED' for i in snapshot['issues']))
    def test_filtered_patients_are_registered_not_falsely_missing(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);src=r/'src';src.mkdir();out=r/'out'
            dicom(src/'us.dcm',np.ones((2,3),dtype=np.uint8),modality='US')
            run_project([src],out,ProjectOptions(patients=('DIFFERENT',)))
            snapshot=json.loads((out/'项目数据快照.json').read_text(encoding='utf-8'))
            self.assertEqual(snapshot['patients'][0]['availability']['US'],'已登记，尚未处理')
    def test_legacy_entry_cannot_overwrite_upgraded_project(self):
        from run_pipeline import validate_paths
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);src=r/'src';src.mkdir();out=r/'out';out.mkdir()
            (out/'project.json').write_text('{"schema":1}')
            (out/'processing_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'升级'):validate_paths(src,out)
