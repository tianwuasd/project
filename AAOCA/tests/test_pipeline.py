import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

from patient_pipeline.clinical import normal_id, normalize_text, choose_document_patient, read_roster
from patient_pipeline.imaging import read_header, classify_ct, deduplicate, quality_metrics
from patient_pipeline.reports import write_csv
from run_pipeline import validate_paths, discover, run
from patient_pipeline.engine import process_patient


def make_ct(path, ident='000123', sop=None, payload=True):
    fm=FileMetaDataset(); fm.TransferSyntaxUID=ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID=CTImageStorage; fm.MediaStorageSOPInstanceUID=sop or generate_uid()
    d=FileDataset(str(path),{},file_meta=fm,preamble=b'\0'*128)
    d.SOPClassUID=fm.MediaStorageSOPClassUID; d.SOPInstanceUID=fm.MediaStorageSOPInstanceUID
    d.PatientID=ident; d.Modality='CT'; d.StudyInstanceUID=generate_uid(); d.SeriesInstanceUID=generate_uid()
    d.Rows=2; d.Columns=3; d.BitsAllocated=16; d.BitsStored=16; d.HighBit=15
    d.PixelRepresentation=1; d.SamplesPerPixel=1; d.PhotometricInterpretation='MONOCHROME2'
    d.PixelSpacing=[1,1]; d.ImageOrientationPatient=[1,0,0,0,1,0]; d.ImagePositionPatient=[0,0,0]
    d.RescaleSlope=1; d.RescaleIntercept=0
    if payload:d.PixelData=np.arange(6,dtype=np.int16).tobytes()
    d.save_as(path,enforce_file_format=True)
    return d


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_id_leading_zeros_and_excel_numbers(self):
        self.assertEqual(normal_id('00123'),'00123')
        self.assertEqual(normal_id(123.0),'123')
        self.assertEqual(normal_id(None),'')
        self.assertEqual(normal_id(' １２３ '),'123')

    def test_text_normalization_retains_negative_and_page_content(self):
        s=normalize_text('未见\u3000冠状动脉狭窄\r\n\r\n⽣⽇：２０２０')
        self.assertIn('未见 冠状动脉狭窄',s);self.assertIn('生日:2020',s)

    def test_pdf_id_wins_over_unique_name(self):
        roster={'001':{'name':'张甲','aliases':['001']},'002':{'name':'张乙','aliases':['002']}}
        ident,basis=choose_document_patient('张甲-病历.pdf','/EMR/002/888',roster)
        self.assertIsNone(ident);self.assertIn('conflict',basis)

    def test_pdf_unique_id_matches(self):
        roster={'001':{'name':'张甲','aliases':['001','888']}}
        self.assertEqual(choose_document_patient('张甲-病历.pdf','/EMR/001/888',roster),('001','document_id'))

    def test_pdf_explicit_id_conflict_not_hidden_by_url(self):
        roster={'001':{'name':'张甲','aliases':['001']},'002':{'name':'张乙','aliases':['002']}}
        self.assertIsNone(choose_document_patient('张甲-病历.pdf','/EMR/001/888 门诊号:002',roster)[0])

    def test_later_pdf_page_conflict_quarantined(self):
        import fitz
        from patient_pipeline.clinical import process_documents
        p=self.root/'甲-病历.pdf';d=fitz.open()
        d.new_page().insert_text((40,40),'/EMR/001/888 patient record first page')
        d.new_page().insert_text((40,40),'/EMR/002/999 patient record second page')
        d.save(p);d.close()
        roster={'001':{'name':'甲','aliases':['001']},'002':{'name':'乙','aliases':['002']}}
        results,issues,_=process_documents([p],roster,self.root/'output')
        self.assertFalse(results[0].get('verified_link'))
        self.assertIn('pending_documents',results[0]['text_file'])

    def test_bad_optional_series_count_does_not_crash(self):
        p=self.root/'a.dcm';make_ct(p);r=read_header(p)
        r['tags']['NumberOfSeriesRelatedInstances']='bad'
        result=process_patient('000123',[r],{},self.root/'output',audit=True)
        self.assertEqual(result[1][0]['转换状态'],'拒绝转换')

    def test_bad_header_does_not_report_no_ct(self):
        r={'path':str(self.root/'bad.dcm'),'error':'bad header'}
        summary,*_=process_patient('123',[r],{},self.root/'output',audit=True,block_reason='无法读取影像头')
        self.assertIn('无法判定',summary['CT数据状态'])

    def test_name_only_is_not_verified(self):
        roster={'001':{'name':'张甲','aliases':['001']}}
        self.assertEqual(choose_document_patient('张甲-病历.pdf','无编号的病例',roster),('001','name_only_unverified'))

    def test_same_name_ambiguous(self):
        roster={'001':{'name':'张甲','aliases':['001']},'002':{'name':'张甲','aliases':['002']}}
        self.assertIsNone(choose_document_patient('张甲-病历.pdf','',roster)[0])

    def test_xlsx_id_and_conflicting_rows(self):
        import openpyxl
        w=openpyxl.Workbook();s=w.active;s.append(['住院号','门诊号','姓名','病历'])
        s.append([77,'001','张甲',1]);s.append([88,'001','张乙',1]);p=self.root/'roster.xlsx';w.save(p)
        roster,issues=read_roster([p])
        self.assertTrue(roster['001']['conflict']);self.assertTrue(issues)

    def test_header_reads_no_pixel_array(self):
        from unittest.mock import patch
        p=self.root/'ok.dcm';make_ct(p)
        with patch.object(pydicom.dataset.Dataset,'pixel_array',property(lambda _: (_ for _ in ()).throw(AssertionError('pixel decode')))):
            r=read_header(p)
        self.assertEqual(r['pixel_status'],'native_length_ok')
        self.assertEqual(r['patient_id'],'000123');self.assertEqual(r['tags']['Rows'],2)

    def test_ct_sop_mislabeled_us_rejected(self):
        p=self.root/'bad.dcm';d=make_ct(p);d.Modality='US';d.save_as(p,enforce_file_format=True)
        with self.assertRaisesRegex(ValueError,'SOPClassUID'):read_header(p)

    def test_nonfinite_optional_header_value_is_serializable(self):
        p=self.root/'nan.dcm';d=make_ct(p);d.CTDIvol=float('nan');d.save_as(p,enforce_file_format=True)
        json.dumps(read_header(p),allow_nan=False)

    def test_missing_and_truncated_pixels(self):
        p=self.root/'bad.dcm';make_ct(p,payload=False)
        self.assertEqual(read_header(p)['pixel_status'],'missing')
        make_ct(p);raw=p.read_bytes();p.write_bytes(raw[:-4])
        self.assertEqual(read_header(p)['pixel_status'],'truncated')

    def test_ct_does_not_automatically_mean_cta(self):
        self.assertEqual(classify_ct({'Modality':'CT'})[0],'CT_未确认CTA')
        self.assertEqual(classify_ct({'SeriesDescription':'Coronary CTA'})[0],'CTA_描述明确')
        self.assertEqual(classify_ct({'ProtocolName':'Cardiac','ContrastBolusAgent':'CE'})[0],'增强心脏CT_疑似CTA')

    def test_coronary_calcium_is_not_cta(self):
        self.assertEqual(classify_ct({'SeriesDescription':'Coronary calcium score noncontrast'})[0],'CT_未确认CTA')

    def test_identical_duplicate_vs_conflicting_sop(self):
        a=self.root/'a.dcm';make_ct(a);b=self.root/'b.dcm';b.write_bytes(a.read_bytes())
        rows=[read_header(a),read_header(b)]
        kept,removed,conflicts=deduplicate(rows)
        self.assertEqual((len(kept),len(removed),len(conflicts)),(1,1,0))
        b.write_bytes(b.read_bytes()+b'changed')
        kept,removed,conflicts=deduplicate(rows)
        self.assertTrue(conflicts)

    def test_quality_does_not_pretend_clinical_assessment(self):
        m=quality_metrics(np.zeros((2,3)),0,1)
        self.assertTrue(m['constant']);self.assertEqual(m['finite_fraction'],1)

    def test_csv_formula_neutralization(self):
        p=self.root/'report.csv';write_csv(p,[{'姓名':'=1+1','编号':'001'}],['姓名','编号'])
        with p.open(encoding='utf-8-sig',newline='') as f: row=next(csv.DictReader(f))
        self.assertEqual(row['姓名'],"'=1+1");self.assertEqual(row['编号'],'001')

    def test_output_safety_and_output_discovery(self):
        src=self.root/'input';src.mkdir();(src/'x.dcm').write_bytes(b'bad')
        with self.assertRaises(ValueError):validate_paths(src,src)
        existing=self.root/'existing';existing.mkdir()
        validate_paths(src,existing)  # Empty output folders are now supported.
        (existing/'unrelated.txt').write_text('keep')
        with self.assertRaises(ValueError):validate_paths(src,existing)
        old=src/'old';old.mkdir();(old/'processing_manifest.json').write_text('{}');(old/'y.dcm').write_bytes(b'bad')
        data=discover(src,src/'new')
        self.assertEqual([p.name for p in data['dicom']],['x.dcm'])

    def test_complete_synthetic_pipeline_and_source_unchanged(self):
        import openpyxl,fitz,hashlib
        from pydicom.uid import UltrasoundImageStorage
        src=self.root/'input';src.mkdir();case=src/'000123';case.mkdir()
        first=make_ct(case/'a.dcm')
        second=make_ct(case/'b.dcm');second.StudyInstanceUID=first.StudyInstanceUID;second.SeriesInstanceUID=first.SeriesInstanceUID
        second.ImagePositionPatient=[0,0,1];second.save_as(case/'b.dcm',enforce_file_format=True)
        us=make_ct(case/'us.dcm');us.Modality='US';us.SOPClassUID=UltrasoundImageStorage;us.file_meta.MediaStorageSOPClassUID=UltrasoundImageStorage
        us.save_as(case/'us.dcm',enforce_file_format=True)
        w=openpyxl.Workbook();s=w.active;s.append(['门诊号','姓名','病历']);s.append(['000123','Alice',1]);w.save(src/'roster.xlsx')
        doc=fitz.open();doc.new_page().insert_text((40,40),'/EMR/000123/888 clinical text for this patient.');doc.save(src/'Alice-clinical.pdf');doc.close()
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()}
        out=src/'result';status=run(src,out)
        self.assertIn(status,[0,2])
        manifest=json.loads((out/'processing_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual((manifest['ct_converted'],manifest['us_converted']),(1,1))
        table=json.loads((out/'details/患者总表.json').read_text(encoding='utf-8'))
        self.assertEqual(table[0]['已核实PDF数'],1)
        self.assertEqual(before,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})
        self.assertEqual(len(discover(src,src/'next')['dicom']),3)


if __name__=='__main__':unittest.main()
