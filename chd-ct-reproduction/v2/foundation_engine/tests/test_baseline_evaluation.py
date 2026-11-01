from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from evaluation import score_case
from evaluation_summary import summarize
from server_data import read_json,write_json,digest

class EvaluationTests(unittest.TestCase):
    def test_baseline_summary_and_undefined_case(self):
        for empty in [False,True]:
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);folder=root/'08_results/evaluation'
                model={'fold':'imagechd_fold0','method':'S1','targets':['synthetic']}
                write_json(folder/'frozen_models.json',{'models':[model]})
                sha=digest(folder/'frozen_models.json')
                reference=np.zeros((4,4,4),dtype=np.uint8)
                if not empty:reference[1:3,1:3,1:3]=1
                score=score_case(reference,reference,np.eye(4))
                write_json(folder/'imagechd_fold0_B0/synthetic.json',dict(status='completed',signature={'frozen_sha256':sha},
                    modality='CT',outputs={},results={'raw':score,'lcc':score}))
                summarize(root)
                result=read_json(root/'09_reports/final_evaluation/summary.json')['results']['CT_raw']['methods']['S1']
                self.assertEqual(result['center_equal_dice'],None if empty else 1.)
                self.assertNotIn('B1',(root/'09_reports/final_evaluation/README.md').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
