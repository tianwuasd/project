import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from dataset1282_analysis.short_workspace import ShortWorkspace


class ShortPreprocessingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DATASET1282_RESEARCH_PYTHON'),
                         'Set DATASET1282_RESEARCH_PYTHON for real nnU-Net acceptance')
    def test_official_numerical_recovery_retains_only_logs(self):
        source=Path(__file__).resolve().parents[1]
        output=Path(os.environ.get('DATASET1282_SHORT_TEST_ROOT',r'D:\projects\back\t17\sr'))
        research=os.environ['DATASET1282_RESEARCH_PYTHON']
        with ShortWorkspace(output,prefix='v') as workspace:
            # 固定本次代码副本，保证并行编辑源文件时恢复契约保持一致。
            code=workspace.scratch/'code'
            code.mkdir()
            for name in ('src','configs','scripts','tests'):
                shutil.copytree(source/name,code/name,ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('bootstrap.py','run.py','requirements-lock.txt'):
                shutil.copyfile(source/name,code/name)
            env=dict(os.environ,TEMP=str(workspace.scratch),TMP=str(workspace.scratch),
                     PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1')
            with (workspace.logs/'short-test.log').open('w',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-B',str(code/'scripts/short_test.py'),
                    '--work-root',str(workspace.scratch/'results'),'--research-python',research],
                    cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
            self.assertEqual(result.returncode,0,(workspace.logs/'short-test.log').read_text(encoding='utf-8'))
            self.assertFalse(list((workspace.scratch/'results').rglob('.scratch')))
            self.assertTrue(all(p.suffix=='.log' for p in (workspace.scratch/'results').rglob('*') if p.is_file()))
            self.assertFalse((code/'.runtime').exists())
        self.assertFalse(workspace.scratch.exists())
        self.assertTrue(all(p.suffix=='.log' for p in workspace.logs.rglob('*') if p.is_file()))


if __name__=='__main__':
    unittest.main()
