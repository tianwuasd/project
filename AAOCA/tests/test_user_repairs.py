import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair_policy import name_variants,apply_roster
from patient_pipeline.clinical import choose_document_patient,process_documents
from convert_us import frame_timing
from common import ConversionError
from pydicom.dataset import Dataset


class UserRepairTests(unittest.TestCase):
    def test_reversible_chinese_name_before_unicode_normalization(self):
        for name in ['测试甲','测试乙','测试丙','测试丁']:
            raw=name.encode('gb18030').decode('latin1')+' / ROMAN NAME'
            self.assertIn(name,name_variants(raw))

    def test_confirmed_groups_remain_separate_with_aliases(self):
        roster={i:{'name':n,'aliases':[i],'fields':{},'conflict':True} for i,n in [('000001','示例甲'),('000002','示例乙')]}
        changes={i:{'name':r['name'],'name_aliases':[r['name'],alias],'aliases':[i,visit],'fields':{},'source_records':[],'field_history':{}}
                 for (i,r),(alias,visit) in zip(roster.items(),[('示例甲别名','VISIT_A'),('示例乙别名','VISIT_B')])}
        with patch('repair_policy.load_policy',return_value={'version':'test','roster_overrides':changes,'id_aliases':{}}):
            result,_=apply_roster(roster,[])
        self.assertEqual(len(result),2)
        self.assertFalse(any(r['conflict'] for r in result.values()))
        self.assertEqual(choose_document_patient('示例甲别名-病例.pdf','',result),('000001','user_confirmed_name'))
        self.assertEqual(choose_document_patient('示例乙别名-病例.pdf','',result),('000002','user_confirmed_name'))

    def test_authorized_name_accepts_unknown_id_without_global_default_change(self):
        r={'001':{'name':'甲','name_aliases':['甲'],'aliases':['001'],'_name_merge_approved':True}}
        self.assertEqual(choose_document_patient('甲-病例.pdf','/EMR/999/123',r),('001','user_confirmed_name'))
        r['001']['_name_merge_approved']=False
        self.assertIsNone(choose_document_patient('甲-病例.pdf','/EMR/999/123',r)[0])

    def test_name_page_conflict_and_low_text_are_information(self):
        import fitz
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'甲-病例.pdf';doc=fitz.open()
            doc.new_page().insert_text((30,30),'Patient record /EMR/999/123 unknown source identifier')
            doc.new_page();doc.save(source);doc.close()
            roster={'001':{'name':'甲','name_aliases':['甲'],'aliases':['001'],'_name_merge_approved':True}}
            result,issues,_=process_documents([source],roster,root/'output')
            self.assertTrue(result[0]['verified_link'])
            self.assertEqual(result[0]['empty_pages'],[2])
            self.assertTrue(all(i.get('severity')=='info' for i in issues))
            self.assertNotIn('DOCUMENT_REVIEW',{i['code'] for i in issues})
            self.assertTrue((root/'output'/result[0]['text_file']).is_file())

    def test_time_vector_is_still_rejected_and_unchanged(self):
        d=Dataset();d.FrameTimeVector=[60,80,60]
        with self.assertRaises(ConversionError):frame_timing(d,3)
        self.assertEqual(list(map(float,d.FrameTimeVector)),[60,80,60])


if __name__=='__main__':unittest.main()
