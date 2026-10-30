"""Synthetic source-defect checks and re-export recovery workflow."""
import json
import tempfile
import unittest
from pathlib import Path

import openpyxl
import pymupdf

from run_project import run_project
from patient_pipeline.project_model import ProjectOptions
from patient_pipeline.project_store import ProjectStore


class PDFReexportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.src=self.root/'input';self.src.mkdir();self.out=self.root/'output'
        w=openpyxl.Workbook();s=w.active;s.append(['患者编号','姓名'])
        s.append(['001','Alice']);s.append(['002','Bob']);w.save(self.src/'roster.xlsx');w.close()
    def tearDown(self):self.tmp.cleanup()
    def run_project(self,action='append',inputs=True):
        return run_project([self.src] if inputs else [],self.out,ProjectOptions(action=action,match_names=True))
    def snap(self):return json.loads((self.out/'项目数据快照.json').read_text(encoding='utf-8-sig'))
    def good_pdf(self,path,ident):
        with pymupdf.open() as d:
            d.new_page().insert_text((40,40),'Patient ID: '+ident+'\nSynthetic clinical text')
            d.save(path)
    def test_identical_header_only_inputs_stay_separate_and_generate_requests(self):
        data=b'%PDF-1.3\n%binary-note\n'
        for name in ['Alice','Bob']:(self.src/(name+'-report.pdf')).write_bytes(data)
        self.run_project();s=self.snap()
        self.assertEqual(s['overview']['PDF']['failed'],2)
        self.assertEqual(s['overview']['PDF']['success'],0)
        self.assertEqual(s['overview']['PDF']['failed_reasons'],{'PDF_HEADER_ONLY':2})
        requests=s['pdf_reexport_requests'];self.assertEqual({r['patient_id'] for r in requests},{'001','002'})
        self.assertTrue(all(r['size_bytes']==len(data) for r in requests))
        self.assertTrue((self.out/'待补PDF清单.csv').is_file())
        self.assertIn('PDF待补原件',(self.out/'查看结果.html').read_text(encoding='utf-8-sig'))
    def test_valid_small_pdf_is_not_misclassified_by_size(self):
        p=self.src/'Alice-report.pdf';self.good_pdf(p,'001');self.assertLess(p.stat().st_size,4096)
        self.run_project();s=self.snap()
        self.assertEqual(s['overview']['PDF']['success'],1);self.assertEqual(s['pdf_reexport_requests'],[])
    def test_empty_pdf_has_explicit_reason(self):
        (self.src/'Alice-report.pdf').write_bytes(b'');self.run_project()
        self.assertEqual(self.snap()['overview']['PDF']['failed_reasons'],{'PDF_EMPTY_FILE':1})
    def test_retry_finds_reexport_at_registered_path_without_new_input(self):
        p=self.src/'Alice-report.pdf';p.write_bytes(b'%PDF-1.3\n%comment\n');self.run_project()
        self.good_pdf(p,'001')
        self.run_project(action='report',inputs=False)
        self.assertEqual(self.snap()['overview']['PDF']['success'],0)
        self.run_project(action='retry',inputs=False);s=self.snap()
        self.assertEqual(s['overview']['PDF']['success'],1)
        self.assertEqual(s['overview']['PDF']['failed'],0);self.assertEqual(s['pdf_reexport_requests'],[])
        with ProjectStore(self.out) as store:
            self.assertTrue(any(u.get('retired') for _,u in store.items('units')))
    def test_old_cached_failure_is_upgraded_without_reextracting_healthy_pdf(self):
        bad=self.src/'Alice-report.pdf';bad.write_bytes(b'%PDF-1.3\n%comment\n')
        self.good_pdf(self.src/'Bob-report.pdf','002');self.run_project()
        with ProjectStore(self.out) as store:
            for s in list(store.sources(kind='PDF')):
                s.pop('pdf_validation_version',None)
                if s.get('read_error'):
                    s['issues']=[{'code':'PDF_READ_FAILED','severity':'error','message':'old parser','patient_id':'001','source':s['path']}]
                store.put_source(s)
        self.run_project(action='retry',inputs=False);s=self.snap()
        self.assertEqual(s['overview']['PDF']['failed_reasons'],{'PDF_HEADER_ONLY':1})
        self.assertEqual(s['catalog_refresh']['metadata_refreshed'],1)
        self.assertEqual(s['processing']['documents_reused'],1)
