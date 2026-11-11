import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ShortSamTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DATASET1282_RESEARCH_PYTHON'),
                         'Set DATASET1282_RESEARCH_PYTHON for SAM CLI preconditions')
    def test_short_cli_failure_rethrows_and_retains_only_logs(self):
        root=Path(__file__).resolve().parents[1]
        parent=Path(os.environ.get('DATASET1282_SHORT_TEST_ROOT',r'D:\projects\back\t17\sr'))
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            folder=Path(folder)
            for options in (['--threads','0'],['--resume']):
                output=folder/('resume' if '--resume' in options else 'threads')
                result=subprocess.run([os.environ['DATASET1282_RESEARCH_PYTHON'],'-B',
                    str(root/'scripts/sam_med3d_train.py'),'run','--prepared',str(folder/'absent-prepared'),
                    '--weights',str(folder/'absent-weights.pt'),'--output',str(output),
                    '--gpus','1',*options],env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),
                    capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('ValueError',result.stderr)
                self.assertFalse(list(output.rglob('.scratch')))
                self.assertTrue(all(p.suffix=='.log' for p in output.rglob('*') if p.is_file()))
                logs=list(output.glob('sam-short-*/acceptance.log'))
                self.assertEqual(len(logs),1)
                events=[json.loads(line) for line in logs[0].read_text(encoding='utf-8').splitlines()]
                finish=next(event for event in events if event['event']=='finish')
                self.assertFalse(finish['passed'])
                self.assertIn('ValueError',finish['error'])


if __name__=='__main__':
    unittest.main()
