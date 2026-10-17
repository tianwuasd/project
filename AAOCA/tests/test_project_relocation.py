"""Synthetic regression coverage for moved projects and report naming."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_reports import build_snapshot, render


class RelocationTests(unittest.TestCase):
    def test_relative_outputs_remain_valid_after_move(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'renamed'
            with ProjectStore(old) as store:
                rel='patients/example/ct/image.dat'
                p=store.path(rel);p.parent.mkdir(parents=True);p.write_bytes(b'synthetic')
                store.save_unit('unit',{'kind':'CT','pid':'example','status':'success','outputs':[rel],'issues':[]})
            shutil.move(old,new)
            with ProjectStore(new) as store:
                self.assertEqual(store.path(store.unit('unit')['outputs'][0]).read_bytes(),b'synthetic')

    def test_project_title_persists_and_is_escaped(self):
        with tempfile.TemporaryDirectory() as d, ProjectStore(d) as store:
            marker=store.root/'project.json';value=json.loads(marker.read_text(encoding='utf-8'))
            value['report_title']='TAPVC <研究> & 数据'
            marker.write_text(json.dumps(value),encoding='utf-8')
            page=render(build_snapshot(store),store)
            self.assertIn('<title>TAPVC &lt;研究&gt; &amp; 数据</title>',page)
            self.assertIn('<h1>TAPVC &lt;研究&gt; &amp; 数据</h1>',page)
            self.assertNotIn('AAOCA 数据项目总览',page)

    def test_tapvc_name_is_detected_when_title_not_set(self):
        with tempfile.TemporaryDirectory() as d, ProjectStore(Path(d)/'TAPVC_result') as store:
            self.assertIn('<title>TAPVC 数据项目总览</title>',render(build_snapshot(store),store))
