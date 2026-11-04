"""Offline code update: staged hash checks and preservation of private files."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch,MagicMock
ROOT=Path(__file__).resolve().parents[1]


class UpdateTests(unittest.TestCase):
    def test_existing_environment_authentication_is_used_without_lookup(self):
        sys.path.insert(0,str(ROOT))
        from update import github_token
        with patch.dict('os.environ',{'GH_TOKEN':'example-existing-credential'},clear=True), \
             patch('update.subprocess.run') as command:
            self.assertEqual(github_token(),'example-existing-credential')
            command.assert_not_called()

    def test_existing_git_credential_lookup_is_repository_scoped(self):
        sys.path.insert(0,str(ROOT))
        from update import github_token
        response=MagicMock(returncode=0,stdout='username=example\npassword=existing-token\n')
        with patch.dict('os.environ',{},clear=True), \
             patch('update.shutil.which',side_effect=lambda name:'git' if name=='git' else None), \
             patch('update.subprocess.run',return_value=response) as command:
            self.assertEqual(github_token(),'existing-token')
            call=command.call_args
            self.assertIn('host=github.com',call.kwargs['input'])
            self.assertIn('path=tianwuasd/project.git',call.kwargs['input'])
            self.assertEqual(call.kwargs['env']['GIT_TERMINAL_PROMPT'],'0')

    def test_authenticated_code_requests_stay_on_official_api_host(self):
        sys.path.insert(0,str(ROOT))
        from update import remote_fetch,NoRedirect
        opener=MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value=b'code'
        self.assertEqual(remote_fetch('code-manifest.json',opener,'existing-token'),b'code')
        request=opener.open.call_args.args[0]
        self.assertEqual(request.full_url,'https://api.github.com/repos/tianwuasd/project/contents/chd-ct-panel-v2/code-manifest.json?ref=main')
        self.assertEqual(request.get_header('Authorization'),'Bearer existing-token')
        self.assertIsNone(NoRedirect().redirect_request(request,None,302,'redirect',{},'https://another.example/code'))
        remote_fetch('code-manifest.json',opener,'existing-token','codex/example-branch')
        self.assertTrue(opener.open.call_args.args[0].full_url.endswith('?ref=codex%2Fexample-branch'))

    def test_offline_update_preserves_local_config_outputs_and_siblings(self):
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            folder=Path(temporary).resolve()
            self.assertTrue(folder.is_relative_to(scratch.resolve()))
            target=folder/'target';target.mkdir()
            source=folder/'source';source.mkdir()
            shutil.copyfile(ROOT/'update.py',target/'update.py')
            (target/'README.md').write_text('old',encoding='utf-8')
            (target/'project.local.json').write_bytes(b'private paths')
            (target/'outputs').mkdir();(target/'outputs/keep').write_bytes(b'original output')
            (folder/'sibling.txt').write_bytes(b'neighbor')
            payload=b'new documentation'
            (source/'README.md').write_bytes(payload)
            manifest=dict(project='chd-ct-panel-v2',files={'README.md':hashlib.sha256(payload).hexdigest()})
            (source/'code-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            command=[sys.executable,str(target/'update.py'),'--source',str(source)]
            checked=subprocess.run(command+['--check'],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(checked.returncode,0,checked.stderr)
            self.assertEqual((target/'README.md').read_text(encoding='utf-8'),'old')
            applied=subprocess.run(command+['--apply'],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(applied.returncode,0,applied.stderr)
            self.assertIn('SHA256',applied.stdout)
            receipts=list((target/'outputs/updates').glob('*/update-result.json'))
            applied_record=next(json.loads(p.read_text()) for p in receipts if json.loads(p.read_text())['action']=='applied')
            self.assertIn('SHA256',Path(applied_record['log_path']).read_text(encoding='utf-8'))
            self.assertEqual((target/'README.md').read_bytes(),payload)
            self.assertEqual((target/'project.local.json').read_bytes(),b'private paths')
            self.assertEqual((target/'outputs/keep').read_bytes(),b'original output')
            self.assertEqual((folder/'sibling.txt').read_bytes(),b'neighbor')
            self.assertTrue(list((target/'outputs/updates').glob('*/backup/README.md')))
            # Bad incoming code must fail BEFORE replacing any live code.
            (source/'README.md').write_bytes(b'tampered')
            manifest['files']['README.md']=hashlib.sha256(b'next documentation').hexdigest()
            (source/'code-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            rejected=subprocess.run(command+['--apply'],capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(rejected.returncode,0)
            self.assertEqual((target/'README.md').read_bytes(),payload)
            self.assertIn('更新未完成',rejected.stdout)


if __name__=='__main__':
    unittest.main()
