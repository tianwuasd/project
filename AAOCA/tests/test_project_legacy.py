import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_reports import build_snapshot


def table(root,name,rows):
    with (root/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class LegacyTests(unittest.TestCase):
    def test_legacy_empty_pixel_error_is_explained_as_pixel_failure(self):
        from patient_pipeline.project_process import error_code
        self.assertEqual(error_code('empty','US'),'PIXEL_INVALID')
    def test_timing_failure_not_double_counted_as_other_image_and_phases_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)/'legacy';r.mkdir();(r/'details').mkdir();folder=r/'patients/001';folder.mkdir(parents=True)
            (r/'processing_manifest.json').write_text('{"status":"completed"}')
            table(r,'患者总表.csv',[{'患者编号':'001','姓名':'Alice','患者数据目录':'patients/001'}])
            table(r,'CT序列表.csv',[{'患者编号':'001','检查UID':'1.2.3','序列UID':'1.2.4','转换状态':'成功','去重后文件数':6,
                '声明切片或帧数':6,'输出目录':'patients/001/ct','错误':''}])
            phase=[]
            for n in [1,2]:
                p=folder/f'phase{n}/image.nii.gz';p.parent.mkdir();p.write_bytes(b'verified previous output')
                phase.append({'原检查UID':'1.2.3','原序列UID':'1.2.4','NIfTI文件':p.relative_to(r).as_posix()})
            table(r,'CT期相输出表.csv',phase)
            source=str(Path(d)/'offline/001/bad.dcm')
            h={'path':source,'bytes':10,'mtime_ns':20,'tags':{}}
            c=sqlite3.connect(r/'details/文件索引.sqlite')
            c.execute('CREATE TABLE files(path TEXT,pid TEXT,modality TEXT,study TEXT,series TEXT,sop TEXT,payload TEXT)')
            c.execute('INSERT INTO files VALUES(?,?,?,?,?,?,?)',(source,'001','','','','',json.dumps(h)));c.commit();c.close()
            (r/'details/问题清单.json').write_text(json.dumps([{'patient_id':'001','source':source,'code':'DICOM_HEADER_FAILED','message':'超声时间向量异常'}]),encoding='utf-8')
            with ProjectStore(Path(d)/'evaluation') as s:
                s.load_legacy(r);snapshot=build_snapshot(s)
                self.assertEqual(snapshot['overview']['CT']['success'],1)
                self.assertEqual(snapshot['overview']['CT']['output_files'],2)
                self.assertEqual(snapshot['overview']['US']['failed'],1)
                self.assertFalse([u for _,u in s.items('units') if u['kind']=='FILE'])
                self.assertTrue((folder/'phase1/image.nii.gz').is_file())
