import tempfile
import unittest
from pathlib import Path
import numpy as np
import pymupdf
import openpyxl
from test_converters import dicom
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_model import ProjectOptions
from patient_pipeline.project_ingest import ingest
from patient_pipeline.project_identity import resolve_identity
from patient_pipeline.runtime import Control


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.src=self.root/'input';self.src.mkdir();self.store=ProjectStore(self.root/'out')
    def tearDown(self):self.store.close();self.tmp.cleanup()
    def load(self,**options):
        ingest(self.store,[self.src],Control());resolve_identity(self.store,ProjectOptions(**options),Control())
    def pdf(self,name,text):
        with pymupdf.open() as d:
            d.new_page().insert_text((40,40),text)
            d.save(self.src/name)
    def roster(self,rows):
        w=openpyxl.Workbook();s=w.active;s.append(['患者编号','姓名'])
        for row in rows:s.append(row)
        w.save(self.src/'patients.xlsx');w.close()
    def test_pdf_only_bootstraps_id_without_roster(self):
        self.pdf('record.pdf','Patient ID: 000123\nClinical record')
        self.load()
        self.assertIn('000123',dict(self.store.items('patients')))
        r=next(self.store.sources(kind='PDF'))
        self.assertEqual(r['pid'],'000123');self.assertTrue(r['verified'])
    def test_unnamed_document_is_pending_and_text_retained(self):
        self.pdf('record.pdf','No identifier but text survives')
        self.load()
        r=next(self.store.sources(kind='PDF'))
        self.assertEqual(r['pid'],'');self.assertFalse(r['verified'])
        self.assertTrue(self.store.path(r['text_file']).exists())
    def test_name_is_opt_in_and_ambiguous_names_never_merge(self):
        self.roster([['001','Alice']]);self.pdf('Alice-report.pdf','clinical')
        self.load();self.assertFalse(next(self.store.sources(kind='PDF'))['verified'])
        resolve_identity(self.store,ProjectOptions(match_names=True),Control())
        self.assertTrue(next(self.store.sources(kind='PDF'))['verified'])
        self.roster([['001','Alice'],['002','Alice']]);self.load(match_names=True)
        self.assertFalse(next(self.store.sources(kind='PDF'))['verified'])
    def test_bad_timing_keeps_patient_and_ct_identity(self):
        p=self.src/'bad.dcm'
        d=dicom(p,np.ones((2,2,3),dtype=np.uint8),modality='US',frames=2)
        d.FrameTimeVector=[30,30];d.save_as(p,enforce_file_format=True)
        dicom(self.src/'ct.dcm',np.ones((2,3),dtype=np.int16))
        (self.src/'broken.dcm').write_bytes(b'broken')
        self.load()
        rows=list(self.store.sources(kind='DICOM'))
        us=next(r for r in rows if r.get('header',{}).get('tags',{}).get('Modality')=='US')
        self.assertEqual(us['pid'],'TEST_ONLY');self.assertIn('timing_error',us['header'])
        ct=next(r for r in rows if r.get('header',{}).get('tags',{}).get('Modality')=='CT')
        self.assertFalse(ct['identity_issues'])
    def test_roster_only_leading_zero_is_kept(self):
        self.roster([['000123','Alice']]);self.load()
        self.assertEqual(list(dict(self.store.items('patients'))),['000123'])
    def test_name_repair_links_external_id_without_phantom_patient(self):
        self.roster([['001','Alice']])
        p=self.src/'image.dcm';d=dicom(p,np.ones((2,3),dtype=np.uint8),modality='US')
        d.PatientID='EXTERNAL';d.PatientName='Alice';d.save_as(p,enforce_file_format=True)
        self.load(match_names=True)
        self.assertEqual(next(self.store.sources(kind='DICOM'))['pid'],'001')
        self.assertEqual(list(dict(self.store.items('patients'))),['001'])
    def test_pdf_conflicting_known_name_and_number_stays_pending(self):
        self.roster([['001','Alice'],['002','Bob']]);self.pdf('Bob-report.pdf','Patient ID: 001')
        self.load(match_names=True)
        self.assertFalse(next(self.store.sources(kind='PDF'))['verified'])
