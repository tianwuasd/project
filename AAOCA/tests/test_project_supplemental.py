"""Synthetic regressions for supplemental document/roster repairs."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openpyxl
import pymupdf

from test_converters import dicom
from run_project import run_project
from patient_pipeline.clinical import document_ids, read_roster
from patient_pipeline.project_ingest import discover, source_key
from patient_pipeline.project_model import ProjectOptions
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.runtime import Control


class SupplementalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.src=self.root/'input';self.src.mkdir();self.out=self.root/'output'
    def tearDown(self):self.tmp.cleanup()
    def roster(self,rows):
        p=self.src/'roster.xlsx';w=openpyxl.Workbook();s=w.active
        s.append(['门诊号','患者姓名','住院号','手术日期','分型','出生日期'])
        for row in rows:s.append(row)
        w.save(p);w.close();return p
    def pdf(self,text):
        p=self.src/'Alice-report.pdf'
        with pymupdf.open() as d:
            for t in text:d.new_page().insert_text((40,40),t)
            d.save(p)
        return p
    def snap(self):return json.loads((self.out/'项目数据快照.json').read_text(encoding='utf-8-sig'))
    def test_pdf_id_does_not_cross_line_to_hospital_name(self):
        self.assertEqual(document_ids('Patient ID\nShanghai Medical Center\n/EMR/001/100'),{'001'})
        self.assertEqual(document_ids('Patient ID: TEST_ONLY'),{'TEST_ONLY'})
        self.assertEqual(document_ids('Patient ID: A-002'),{'A-002'})
    def test_admissions_keep_rows_aliases_and_clinical_variation(self):
        p=self.roster([['001','Alice','VISIT_A','2020-01-01','type A','2019-01-01'],
                       ['001','Alice','VISIT_B','2021-01-01','type B','2019-01-01']])
        roster,issues=read_roster([p]);r=roster['001']
        self.assertFalse(r['conflict']);self.assertEqual(len(r['encounters']),2)
        self.assertTrue({'001','VISIT_A','VISIT_B'}<=set(r['aliases']))
        self.assertEqual(r['field_variants']['分型'],['type A','type B'])
        self.assertNotIn('分型',r['fields'])
        self.assertTrue(any(i['code']=='ROSTER_CLINICAL_REVIEW' for i in issues))
    def test_different_names_or_birth_dates_remain_conflicts(self):
        for second in [('Bob','2019-01-01'),('Alice','2018-01-01')]:
            with self.subTest(second=second):
                p=self.roster([['001','Alice','A','2020','type A','2019-01-01'],
                               ['001',second[0],'B','2021','type A',second[1]]])
                self.assertTrue(read_roster([p])[0]['001']['conflict'])
    def test_sidecars_are_ignored_as_directory_and_explicit_file(self):
        mac=self.src/'__MACOSX';mac.mkdir();p=mac/'._report.pdf';p.write_bytes(b'metadata')
        q=self.src/'._roster.xlsx';q.write_bytes(b'metadata')
        real=self.pdf(['Patient ID: 001'])
        self.assertEqual(discover([self.src,p,q],self.out,Control()),[real])
    def test_retry_refreshes_old_catalog_without_new_inputs(self):
        roster=self.roster([['001','Alice','A','2020','type A','2019'],
                            ['001','Alice','B','2021','type B','2019']])
        pdf=self.pdf(['Patient ID\nShanghai Medical Center\n/EMR/001/100'])
        d=dicom(self.src/'us.dcm',np.ones((2,3),dtype=np.uint8),modality='US')
        d.PatientID='001';d.PatientName='Alice';d.save_as(self.src/'us.dcm',enforce_file_format=True)
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        mac=self.src/'__MACOSX';mac.mkdir();side=mac/'._report.pdf';side.write_bytes(b'metadata')
        with ProjectStore(self.out) as store:
            for p in [roster,pdf]:
                s=store.source(source_key(p));s.pop('clinical_parser_version',None)
                if s['kind']=='PDF':s['document_ids']=['001','Shanghai']
                else:s['roster']['001']['conflict']=True
                store.put_source(s)
            sk=source_key(side);uk='PDF:synthetic-sidecar'
            store.put_source({'key':sk,'path':str(side),'kind':'PDF','unit_key':uk,'issues':[]})
            store.save_unit(uk,{'kind':'PDF','status':'failed','sources':[sk],'outputs':[],'issues':[]})
        run_project([],self.out,ProjectOptions(action='retry',match_names=True))
        snap=self.snap();self.assertEqual(snap['overview']['PDF']['total'],1)
        self.assertEqual(snap['overview']['PDF']['success'],1)
        self.assertEqual(snap['overview']['US']['success'],1)
        self.assertTrue(side.exists())
        with ProjectStore(self.out) as store:
            self.assertIsNone(store.source(source_key(side)))
            self.assertIsNotNone(store.get('excluded_sources',source_key(side)))
            self.assertTrue(store.unit('PDF:synthetic-sidecar')['retired'])
            self.assertEqual(store.source(source_key(pdf))['document_ids'],['001'])
        before=snap['overview'];run_project([],self.out,ProjectOptions(action='retry',match_names=True))
        self.assertEqual(self.snap()['overview'],before)
        self.assertEqual(self.snap()['processing']['images_converted'],0)
    def test_real_second_patient_id_still_blocks_name_matching(self):
        self.roster([['001','Alice','A','2020','type A','2019']])
        self.pdf(['Patient ID: 001','Patient ID: 002'])
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        self.assertEqual(self.snap()['overview']['PDF']['pending'],1)
    def test_footer_review_does_not_invalidate_patient_association(self):
        self.pdf(['Patient ID: 001\n第1/2页'])
        # Use ASCII digits with a Chinese-capable built-in font for the footer.
        p=self.src/'Alice-report.pdf';p.unlink()
        with pymupdf.open() as d:
            page=d.new_page();page.insert_text((40,40),'Patient ID: 001')
            page.insert_text((40,80),'第1/2页',fontname='china-s');d.save(p)
        run_project([self.src],self.out)
        s=self.snap();self.assertEqual(s['overview']['PDF']['success'],1)
        self.assertTrue(any(i['code']=='PDF_EXPORT_REVIEW' and i['severity']=='warning' and i['patient_id']=='001' for i in s['issues']))
    def test_cross_file_admissions_keep_all_rows(self):
        from patient_pipeline.clinical import merge_roster_entries
        first=read_roster([self.roster([['001','Alice','A','2020','type A','2019']])])[0]['001']
        second=read_roster([self.roster([['001','Alice','B','2021','type B','2019']])])[0]['001']
        merge_roster_entries(first,second)
        self.assertFalse(first['conflict']);self.assertEqual(len(first['encounters']),2)
        self.assertNotIn('分型',first['fields'])
    def test_pdf_parser_update_refreshes_cached_success_warnings(self):
        from patient_pipeline.project_process import signature
        p=self.src/'Alice-report.pdf'
        with pymupdf.open() as d:
            page=d.new_page();page.insert_text((40,40),'Patient ID: 001')
            page.insert_text((40,80),'第1/2页',fontname='china-s');d.save(p)
        run_project([self.src],self.out)
        with ProjectStore(self.out) as store:
            s=store.source(source_key(p));s['issues']=[]
            u=store.unit(s['unit_key']);u['issues']=[]
            u['signature']=u['input_signature']=signature([s],ProjectOptions())
            store.put('units',s['unit_key'],u)
            s.pop('clinical_parser_version');store.put_source(s)
        run_project([],self.out,ProjectOptions(action='retry'))
        self.assertTrue(any(i['code']=='PDF_EXPORT_REVIEW' for i in self.snap()['issues']))
    def test_unreadable_named_pdf_retains_owner_but_is_not_success(self):
        self.roster([['001','Alice','A','2020','type A','2019']])
        p=self.src/'Alice-report.pdf';p.write_bytes(b'%PDF-1.3\n%header only\n')
        run_project([self.src],self.out,ProjectOptions(match_names=True))
        s=self.snap();self.assertEqual(s['overview']['PDF']['failed'],1)
        self.assertEqual(s['overview']['PDF']['success'],0)
        self.assertEqual(s['patients'][0]['availability']['PDF'],'已有资料，未成功')
        self.assertTrue(any(i['code']=='PDF_HEADER_ONLY' and i['patient_id']=='001' for i in s['issues']))
        self.assertFalse(any(i['code']=='DOCUMENT_UNMATCHED' for i in s['issues']))
