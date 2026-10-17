import tempfile
import unittest
from pathlib import Path
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_reports import build_snapshot, publish_project
from patient_pipeline.project_classify import CATEGORIES


class ReportTests(unittest.TestCase):
    def test_eight_categories_low_text_note_and_phase_counts(self):
        with tempfile.TemporaryDirectory() as d, ProjectStore(d) as s:
            for n,presence in enumerate(CATEGORIES):
                pid=f'{n:03d}';s.put('patients',pid,{'name':'<script>alert(1)</script>'})
                for yes,kind in zip(presence,['CT','US','PDF']):
                    if yes:
                        folder=Path('patients/待分类')/pid;dest=s.path(folder);dest.mkdir(parents=True,exist_ok=True)
                        s.put('folders',pid,folder.as_posix());file=dest/(kind+'.dat');file.write_text('test')
                        s.save_unit(kind+pid,{'kind':kind,'pid':pid,'status':'success','outputs':[file.relative_to(s.root).as_posix()],
                            'issues':[{'code':'PDF_TEXT_INCOMPLETE_NOTE','message':'low','severity':'info','patient_id':pid}] if kind=='PDF' else []})
            snap=build_snapshot(s)
            self.assertEqual(sorted(snap['category_counts'].values()),[1]*8)
            self.assertEqual(snap['patients'][0]['subcategory'],'01_无其他异常')
            publish_project(s,snap)
            page=(s.root/'查看结果.html').read_text(encoding='utf-8')
            self.assertNotIn('<script>alert(1)</script>',page)
            self.assertIn('&lt;script&gt;',page)
            for _,u in s.items('units'):
                self.assertTrue(s.path(u['outputs'][0]).is_file())
            publish_project(s,build_snapshot(s))
            self.assertEqual(len(list(s.items('units'))),12)
    def test_present_but_failed_is_not_labeled_never_received(self):
        with tempfile.TemporaryDirectory() as d, ProjectStore(d) as s:
            s.put('patients','001',{'name':'A'})
            s.save_unit('ct',{'kind':'CT','pid':'001','status':'failed','outputs':[],
                'issues':[{'code':'PIXEL_INVALID','message':'empty','severity':'error','patient_id':'001'}]})
            snap=build_snapshot(s);p=snap['patients'][0]
            self.assertEqual(p['availability']['CT'],'已有资料，未成功')
            self.assertEqual(p['availability']['US'],'尚未提供')
            self.assertEqual({i['family'] for i in snap['issues']},{'数据本身','资料缺失'})
