import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patient_pipeline.runtime import Control, StopRequested, RunLock, Runtime
from run_pipeline import run
from patient_pipeline import engine

spec=importlib.util.spec_from_file_location('pipeline_fixtures',ROOT/'tests/test_pipeline.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def fixture(base,two_patients=False,with_ct=False,with_pdf=False):
    from pydicom.uid import UltrasoundImageStorage
    import fitz,openpyxl
    source=base/'input data';source.mkdir()
    for index in range(2):
        ident='000456' if index and two_patients else '000123'
        folder=source/ident;folder.mkdir(exist_ok=True);path=folder/f'us{index}.dcm'
        d=fixtures.make_ct(path,ident);d.Modality='US';d.SOPClassUID=UltrasoundImageStorage
        d.PixelRepresentation=0
        d.file_meta.MediaStorageSOPClassUID=UltrasoundImageStorage;d.save_as(path,enforce_file_format=True)
    if with_ct:
        first=fixtures.make_ct(source/'000123/a.dcm');second=fixtures.make_ct(source/'000123/b.dcm')
        second.StudyInstanceUID=first.StudyInstanceUID;second.SeriesInstanceUID=first.SeriesInstanceUID
        second.ImagePositionPatient=[0,0,1];second.save_as(source/'000123/b.dcm',enforce_file_format=True)
    if with_pdf:
        w=openpyxl.Workbook();s=w.active;s.append(['门诊号','姓名']);s.append(['000123','Alice']);w.save(source/'roster.xlsx')
        doc=fitz.open();doc.new_page().insert_text((40,40),'/EMR/000123/888 clinical record with enough text.')
        doc.save(source/'Alice-clinical.pdf');doc.close()
    return source


class ResumeTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_pause_resume(self):
        events=[];control=Control(events.append);control.pause();done=[]
        thread=threading.Thread(target=lambda:(control.checkpoint(),done.append(True)));thread.start()
        time.sleep(.08);self.assertEqual(done,[])
        self.assertTrue(any(e.get('state')=='paused' for e in events))
        control.resume();thread.join(1);self.assertEqual(done,[True])

    def test_stop_during_pause(self):
        control=Control();control.pause();stopped=[]
        def work():
            try:control.checkpoint()
            except StopRequested:stopped.append(True)
        thread=threading.Thread(target=work);thread.start();time.sleep(.04);control.stop();thread.join(1)
        self.assertEqual(stopped,[True])

    def test_output_lock(self):
        with RunLock(self.root):
            with self.assertRaisesRegex(RuntimeError,'使用'):
                with RunLock(self.root):pass
        with RunLock(self.root):pass

    def test_cache_validates_outputs_and_sources(self):
        src=self.root/'input.txt';src.write_text('abc');out=self.root/'out';out.mkdir()
        r=Runtime(out);artifact=out/'a.txt';artifact.write_text('result')
        sig=r.fingerprint([src]);r.save('job',sig,{'ok':True},[artifact])
        self.assertEqual(r.lookup('job',sig),{'ok':True})
        artifact.write_text('damage');self.assertIsNone(r.lookup('job',sig))
        artifact.write_text('result');src.write_text('changed')
        self.assertIsNone(r.lookup('job',r.fingerprint([src])))
        r.close()

    def test_resume_state_persists(self):
        out=self.root/'out';out.mkdir();r=Runtime(out)
        artifact=out/'a.txt';artifact.write_text('result');r.save('job','signature',{'n':1},[artifact]);r.close()
        r=Runtime(out);self.assertEqual(r.lookup('job','signature'),{'n':1});r.close()

    def test_archive_cannot_escape_output(self):
        out=self.root/'out';out.mkdir();src=self.root/'source';src.write_text('original');r=Runtime(out)
        with self.assertRaises(ValueError):r.archive(src)
        self.assertEqual(src.read_text(),'original');r.close()

    def test_rerun_skips_headers_images_and_pdf(self):
        src=fixture(self.root,with_ct=True,with_pdf=True);out=self.root/'output'
        run(src,out)
        with patch('run_pipeline.read_header',side_effect=AssertionError('header re-read')), \
             patch.object(engine,'convert_file',side_effect=AssertionError('US redone')), \
             patch.object(engine,'convert_series',side_effect=AssertionError('CT redone')), \
             patch('fitz.open',side_effect=AssertionError('PDF re-read')):
            run(src,out)
        m=json.loads((out/'processing_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(m['resume_counts']['images_reused'],3)
        self.assertEqual(m['resume_counts']['images_converted'],0)
        self.assertEqual(m['resume_counts']['documents_reused'],1)
        self.assertEqual(m['header_files_read'],0)

    def test_changed_source_only_rebuilds_affected_file(self):
        import pydicom
        src=fixture(self.root);out=self.root/'output';run(src,out)
        path=src/'000123/us0.dcm';d=pydicom.dcmread(path);d.PixelData=b'\x01\x00'*6;d.save_as(path,enforce_file_format=True)
        with patch.object(engine,'convert_file',wraps=engine.convert_file) as convert:run(src,out)
        self.assertEqual(convert.call_count,1)
        self.assertTrue(list((out/'back').rglob('image.png')))

    def test_damaged_output_rebuilt(self):
        src=fixture(self.root);out=self.root/'output';run(src,out)
        path=next(out.glob('patients/*/us/*/image.png'));path.write_bytes(b'damaged')
        with patch.object(engine,'convert_file',wraps=engine.convert_file) as convert:run(src,out)
        self.assertEqual(convert.call_count,1);self.assertGreater(path.stat().st_size,10)

    def test_stop_and_resume_does_not_repeat_completed_unit(self):
        src=fixture(self.root);out=self.root/'output';control=Control()
        def event(e):
            if e.get('counts',{}).get('images_converted',0)>=1:control.stop()
        control.callback=event
        with self.assertRaises(StopRequested):run(src,out,control=control)
        self.assertEqual(json.loads((out/'processing_manifest.json').read_text())['status'],'interrupted')
        with patch.object(engine,'convert_file',wraps=engine.convert_file) as convert:run(src,out)
        self.assertEqual(convert.call_count,1)
        self.assertEqual(len(list(out.glob('patients/*/us/*/image.png'))),2)

    def test_patient_failure_continues_and_preserves_previous_checkpoint(self):
        import run_pipeline
        src=fixture(self.root,two_patients=True);out=self.root/'output';run(src,out)
        original=run_pipeline.process_patient
        def flaky(ident,*args,**kwargs):
            if ident=='000123':raise ValueError('synthetic patient error')
            return original(ident,*args,**kwargs)
        with patch('run_pipeline.process_patient',side_effect=flaky):run(src,out)
        rows=json.loads((out/'details/患者总表.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows),2);self.assertTrue(any(r.get('超声成功文件')==1 for r in rows))
        with patch.object(engine,'convert_file',side_effect=AssertionError('completed output lost')):run(src,out)
        m=json.loads((out/'processing_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(m['resume_counts']['images_reused'],2)

    def test_force_and_wrong_input_guard(self):
        src=fixture(self.root);out=self.root/'output';run(src,out)
        with patch.object(engine,'convert_file',wraps=engine.convert_file) as convert:run(src,out,force=True)
        self.assertEqual(convert.call_count,2)
        other=self.root/'other';other.mkdir()
        with self.assertRaisesRegex(ValueError,'其他输入'):run(other,out)

    def test_incomplete_discovery_preserves_completed_ct(self):
        import run_pipeline,pydicom
        src=fixture(self.root,with_ct=True);out=self.root/'output'
        third=pydicom.dcmread(src/'000123/b.dcm');third.SOPInstanceUID=pydicom.uid.generate_uid()
        third.file_meta.MediaStorageSOPInstanceUID=third.SOPInstanceUID;third.ImagePositionPatient=[0,0,2]
        third.save_as(src/'000123/c.dcm',enforce_file_format=True)
        run(src,out);saved=next(out.glob('patients/*/ct/*/image.nii.gz'));before=saved.read_bytes()
        found=run_pipeline.discover(src,out);found['dicom']=[p for p in found['dicom'] if p.name!='c.dcm']
        found['scan_errors']=['synthetic inaccessible subdirectory']
        with patch('run_pipeline.discover',return_value=found):
            with self.assertRaisesRegex(RuntimeError,'扫描不完整'):run(src,out)
        self.assertEqual(saved.read_bytes(),before)
        with patch.object(engine,'convert_series',side_effect=AssertionError('checkpoint lost')):run(src,out)


if __name__=='__main__':unittest.main()
