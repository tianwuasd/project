import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import openpyxl
import pymupdf
from test_converters import dicom
from run_project import run_project
from patient_pipeline.project_model import ProjectOptions
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_ingest import source_key
from pydicom.uid import generate_uid
import pydicom


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.src=self.root/'input';self.src.mkdir();self.out=self.root/'output'
    def tearDown(self):self.tmp.cleanup()
    def roster(self,rows):
        w=openpyxl.Workbook();s=w.active;s.append(['患者编号','姓名'])
        for row in rows:s.append(row)
        w.save(self.src/'roster.xlsx');w.close()
    def pdf(self,name,pages):
        with pymupdf.open() as d:
            for text in pages:d.new_page().insert_text((40,40),text)
            d.save(self.src/name)
    def snapshot(self):return json.loads((self.out/'项目数据快照.json').read_text(encoding='utf-8'))
    def test_optional_name_confirmation_survives_default_next_batch(self):
        self.roster([['001','Alice']]);self.pdf('Alice-report.pdf',['No id; real text'])
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        self.assertEqual(self.snapshot()['overview']['PDF']['success'],1)
        run_project([],self.out,ProjectOptions(action='retry'))
        self.assertEqual(self.snapshot()['overview']['PDF']['success'],1)
    def test_accepted_name_link_follows_identical_document_after_move(self):
        import shutil
        self.roster([['001','Alice']]);self.pdf('Alice-report.pdf',['No id; real text'])
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        later=self.root/'later';later.mkdir();shutil.copy2(self.src/'Alice-report.pdf',later/'Alice-report.pdf')
        self.src.rename(self.root/'offline')
        run_project([later],self.out)
        self.assertEqual(self.snapshot()['overview']['PDF']['success'],1)
        run_project([],self.out,ProjectOptions(action='retry'))
        self.assertEqual(self.snapshot()['overview']['PDF']['success'],1)
    def test_bad_replacement_pdf_retains_prior_output_patient_and_failure(self):
        self.pdf('report.pdf',['Patient ID: 001'])
        run_project([self.src],self.out)
        with ProjectStore(self.out) as s:old=next(u for _,u in s.items('units') if u['kind']=='PDF')['outputs']
        (self.src/'report.pdf').write_bytes(b'corrupt replacement')
        run_project([self.src],self.out)
        snap=self.snapshot();self.assertEqual(snap['overview']['PDF']['success'],1)
        self.assertEqual([p['id'] for p in snap['patients']],['001'])
        self.assertTrue(any('READ_FAILED' in i['code'] for i in snap['issues']))
        for rel in old:self.assertTrue((self.out/rel).is_file())
    def test_image_known_id_and_different_known_name_is_rejected(self):
        self.roster([['001','Alice'],['002','Bob']])
        p=self.src/'us.dcm';d=dicom(p,np.ones((2,3),dtype=np.uint8),modality='US')
        d.PatientID='001';d.PatientName='Bob';d.save_as(p,enforce_file_format=True)
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        self.assertEqual(self.snapshot()['overview']['US']['success'],0)
        self.assertTrue(any(i['family']=='编号关联' and i['severity']=='error' for i in self.snapshot()['issues']))
    def test_historical_confirmation_cannot_relabel_changed_file(self):
        self.roster([['001','Alice'],['002','Bob']]);p=self.src/'us.dcm'
        d=dicom(p,np.ones((2,3),dtype=np.uint8),modality='US');d.PatientID='001';d.PatientName='Alice';d.save_as(p,enforce_file_format=True)
        run_project([self.src],self.out)
        with ProjectStore(self.out) as s:s.put('confirmed_links',source_key(p),{'pid':'001','basis':'latest legacy index'})
        d=dicom(p,np.ones((2,3),dtype=np.uint8),modality='US');d.PatientID='002';d.PatientName='Bob';d.save_as(p,enforce_file_format=True)
        run_project([self.src],self.out)
        with ProjectStore(self.out) as s:
            r=s.source(source_key(p));self.assertNotEqual((r['pid'],r['verified']),('001',True))
    def test_mixed_pdf_pages_with_unknown_second_id_stay_pending(self):
        self.roster([['001','Alice']]);self.pdf('Alice-report.pdf',['Patient ID: 001','Patient ID: 002'])
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        self.assertEqual(self.snapshot()['overview']['PDF']['success'],0)
        self.assertEqual(self.snapshot()['overview']['PDF']['pending'],1)
    def test_phase_groups_must_share_matrix_dimensions(self):
        study=generate_uid();series=generate_uid()
        for phase,shape in [(1,(2,3)),(2,(4,5))]:
            for z in range(3):
                p=self.src/f'{phase}_{z}.dcm';d=dicom(p,np.ones(shape,dtype=np.int16),position=(0,0,z),study=study,series=series)
                d.TemporalPositionIdentifier=phase;d.save_as(p,enforce_file_format=True)
        run_project([self.src],self.out,ProjectOptions(split_phases=True))
        self.assertEqual(self.snapshot()['overview']['CT']['success'],0)
    def test_accepted_copy_repair_applies_when_appending_slices(self):
        study=generate_uid();series=generate_uid()
        for z in range(2):dicom(self.src/f'{z}.dcm',np.ones((2,3),dtype=np.int16),position=(0,0,z),study=study,series=series)
        d=pydicom.dcmread(self.src/'0.dcm');d.PixelData=b'';d.save_as(self.src/'empty.dcm',enforce_file_format=True)
        run_project([self.src],self.out,ProjectOptions(recover_copies=True))
        dicom(self.src/'2.dcm',np.ones((2,3),dtype=np.int16),position=(0,0,2),study=study,series=series)
        run_project([self.src],self.out)
        with ProjectStore(self.out) as s:
            u=next(u for _,u in s.items('units') if u['kind']=='CT')
            self.assertEqual(u['slices'],3);self.assertNotIn('last_attempt',u)
    def test_resolved_roster_conflict_is_history_not_active_error(self):
        self.roster([['001','Alice'],['001','Alice Old']]);policy=self.root/'policy.json'
        policy.write_text(json.dumps({'roster_overrides':{'001':{'name':'Alice','name_aliases':['Alice','Alice Old'],'aliases':['001']}}}))
        run_project([self.src],self.out,ProjectOptions(policy_path=str(policy)))
        self.assertFalse([i for i in self.snapshot()['issues'] if i['code']=='ROSTER_ID_CONFLICT' and i['severity']=='error'])
