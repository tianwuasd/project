"""Prevent a later requirements install from silently replacing the Torch plan."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import setup_env


class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=scratch)
        self.folder=Path(self.temporary.name)
        (self.folder/'requirements-preprocess.txt').write_text('example==1',encoding='utf-8')
        self.local=self.folder/'project.local.json'
        self.local.write_text('{"keep":"original"}',encoding='utf-8')

    def tearDown(self):self.temporary.cleanup()

    def run_profile(self,action,version,cuda):
        calls=[]
        def execute(command,*args,**kwargs):
            calls.append(command)
            output=(json.dumps({'deployment_torch':{'version':version,'cuda':cuda}})+'\n'
                    if '-c' in command else 'installed\n')
            return subprocess.CompletedProcess(command,0,stdout=output,stderr='')
        with patch.object(setup_env,'ROOT',self.folder),patch.object(sys,'argv',
                ['setup_env.py',action,'--profile','preprocess']), \
             patch.object(setup_env,'run_logged',side_effect=execute),redirect_stdout(io.StringIO()):
            code=setup_env.main()
        name='install' if action=='--install' else 'check'
        result=json.loads(next((self.folder/'outputs/environment').glob('*_'+name+'.json')).read_text(encoding='utf-8'))
        return code,calls,result

    def test_requirements_resolution_keeps_torch_pin_and_cpu_contract(self):
        code,calls,record=self.run_profile('--install','2.8.0+cpu',None)
        self.assertEqual(code,0)
        requirements=next(command for command in calls if '-r' in command)
        self.assertIn('torch==2.8.0',requirements)
        self.assertTrue(record['deployment_contract']['matches'])
        self.assertIn('preprocess_python',json.loads(self.local.read_text()))

    def test_install_that_drifts_to_another_torch_cannot_save_configuration(self):
        code,_,record=self.run_profile('--install','2.15.0','13.0')
        self.assertEqual(code,1);self.assertFalse(record['passed'])
        self.assertFalse(record['deployment_contract']['matches'])
        self.assertEqual(json.loads(self.local.read_text()),{'keep':'original'})

    def test_existing_trusted_environment_api_check_reports_version_difference(self):
        code,_,record=self.run_profile('--check','2.12.1+cu126','12.6')
        self.assertEqual(code,0);self.assertTrue(record['passed'])
        self.assertFalse(record['deployment_contract']['matches']);self.assertTrue(record['warnings'])
        self.assertEqual(json.loads(self.local.read_text()),{'keep':'original'})


if __name__=='__main__':unittest.main()
