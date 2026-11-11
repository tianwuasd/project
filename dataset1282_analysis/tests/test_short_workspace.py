import json
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dataset1282_analysis.contracts import atomic_json, sha256_file
from dataset1282_analysis.short_workspace import ShortWorkspace
from dataset1282_analysis.smoke import synthetic_fixture


class ShortWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=Path(os.environ.get('DATASET1282_SHORT_TEST_ROOT',r'D:\projects\back\t17\short-retention'))
        cls.parent.mkdir(parents=True,exist_ok=True)

    def test_success_failure_and_repeated_runs_preserve_only_logs(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            folder=Path(folder)
            source=folder/'source'
            source.mkdir()
            data,excel=synthetic_fixture(source)
            identity={str(p):sha256_file(p) for p in source.rglob('*') if p.is_file()}
            output=folder/'output'
            runs=[]
            for fail in (False,True,False):
                workspace=ShortWorkspace(output,protected_paths=[data,excel])
                runs.append(workspace.run_root)
                with self.assertRaisesRegex(RuntimeError,'deliberate failure') if fail else nullcontext():
                    with workspace:
                        synthetic_fixture(workspace.scratch/'fixture')
                        atomic_json(workspace.scratch/'acceptance.json',dict(passed=not fail,details='retained diagnostics'))
                        if fail:
                            raise RuntimeError('deliberate failure')
                self.assertFalse(workspace.scratch.exists())
                self.assertTrue(all(p.is_file() and p.suffix=='.log' for p in workspace.run_root.iterdir()))
                rows=[json.loads(line) for line in (workspace.logs/'acceptance.log').read_text(encoding='utf-8').splitlines()]
                self.assertEqual(next(row for row in rows if row['event']=='finish')['passed'],not fail)
                self.assertEqual(next(row for row in rows if row['event']=='report')['contents']['passed'],not fail)
            self.assertEqual(len(set(runs)),3)
            self.assertEqual(identity,{str(p):sha256_file(p) for p in source.rglob('*') if p.is_file()})

    def test_protected_source_and_parent_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            source=Path(folder)/'source'
            source.mkdir()
            for output in (source,source/'child',Path(folder),source/'..'/'output'):
                with self.assertRaises(ValueError):
                    ShortWorkspace(output,protected_paths=[source])
            with self.assertRaises(ValueError):
                ShortWorkspace(Path(__file__).resolve().parents[1]/'output')
            with self.assertRaisesRegex(ValueError,'/data_nas'):
                ShortWorkspace('/data_nas/zhangruichen/short-check')

    def test_corrupt_report_still_removes_all_products(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            workspace=ShortWorkspace(folder)
            with self.assertRaises(json.JSONDecodeError):
                with workspace:
                    (workspace.scratch/'acceptance.json').write_text('{',encoding='utf-8')
            self.assertFalse(workspace.scratch.exists())

    def test_real_smoke_profiles_and_failure_leave_only_logs(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            folder=Path(folder)
            data,excel=synthetic_fixture(folder/'source')
            # quick/capacity 使用真实可读取的影像；synthetic 自行生成损坏输入检查失败诊断。
            manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))[:2]
            atomic_json(data/'manifest.json',manifest)
            identities={str(p):sha256_file(p) for p in (folder/'source').rglob('*') if p.is_file()}
            home=folder/'home'
            home.mkdir()
            output=folder/'output'
            atomic_json(home/'project.local.json',dict(data_root=str(data),metadata_excel=str(excel),
                        work_root=str(output),environment=sys.prefix,max_workers=1,read_concurrency=1))
            env=dict(os.environ,DATASET1282_CODE_ROOT=str(root),DATASET1282_HOME=str(home),
                     PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='-1')
            for profile in ('synthetic','quick','capacity'):
                result=subprocess.run([sys.executable,'-B','-c',
                    'from dataset1282_analysis.smoke import main; raise SystemExit(main())',
                    '--profile',profile],env=env,capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            failed=subprocess.run([sys.executable,'-B','-c',
                'from dataset1282_analysis.smoke import main; raise SystemExit(main())',
                '--profile','quick','--run-id','invalid/run'],env=env,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('Invalid run identifier',failed.stderr)
            self.assertTrue(all(p.suffix=='.log' for p in output.rglob('*') if p.is_file()))
            self.assertFalse(list(output.rglob('.scratch')))
            self.assertEqual(identities,{str(p):sha256_file(p) for p in (folder/'source').rglob('*') if p.is_file()})
            self.assertEqual([p.name for p in home.iterdir()],['project.local.json'])


if __name__=='__main__':
    unittest.main()
