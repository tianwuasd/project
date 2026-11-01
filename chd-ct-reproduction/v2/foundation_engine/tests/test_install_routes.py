"""Installation tries direct first and one explicit proxy fallback, without global edits."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import runtime_setup as setup
from panel_publication.network import save


class InstallRoutesTests(unittest.TestCase):
    def fixture(self, root):
        save(root, proxy='http://127.0.0.1:17891')
        return dict(WHOLE_HEART_ENV_PREFIX=str(root/'envs/vista_baseline_py311'),
                    WHOLE_HEART_JOB=str(root/'job'), TMPDIR=str(root),
                    HTTP_PROXY='http://old.invalid:1234', HTTPS_PROXY='http://old.invalid:1234')

    def test_direct_success_never_attempts_proxy(self):
        with tempfile.TemporaryDirectory() as temp:
            env=self.fixture(Path(temp));seen=[]
            def run(command, selected):
                seen.append(selected)
                return subprocess.CompletedProcess(command,0)
            with patch.object(setup,'_install_child',side_effect=run):
                setup.run_install_command(['pip-command','install','example'],env)
            self.assertEqual(len(seen),1)
            self.assertNotIn('HTTPS_PROXY',seen[0])
            self.assertEqual(env['HTTPS_PROXY'],'http://old.invalid:1234')

    def test_direct_failure_retries_same_command_once_with_runtime_proxy(self):
        with tempfile.TemporaryDirectory() as temp:
            env=self.fixture(Path(temp));seen=[]
            def run(command, selected):
                seen.append((command,dict(selected)))
                if len(seen)==1:raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0)
            with patch.object(setup,'_install_child',side_effect=run):
                setup.run_install_command(['pip-command','install','example'],env)
            self.assertEqual(len(seen),2)
            self.assertEqual(seen[0][0],seen[1][0])
            self.assertEqual(seen[1][1]['HTTPS_PROXY'],'http://127.0.0.1:17891')

    def test_proxy_failure_stops_after_two_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            env=self.fixture(Path(temp))
            with patch.object(setup,'_install_child',side_effect=subprocess.CalledProcessError(1,['download'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    setup.run_install_command(['download'],env)
            self.assertEqual(run.call_count,2)

    def test_interruption_never_starts_proxy_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(setup,'_install_child',side_effect=KeyboardInterrupt) as run:
                with self.assertRaises(KeyboardInterrupt):setup.run_install_command(['download'],self.fixture(Path(temp)))
            self.assertEqual(run.call_count,1)

    def test_signal_exit_never_starts_proxy_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            env=self.fixture(Path(temp))
            for code in (-15,-2,129,130,137,143):
                with self.subTest(returncode=code), patch.object(setup,'_install_child',
                        side_effect=subprocess.CalledProcessError(code,['download'])) as run:
                    with self.assertRaises(subprocess.CalledProcessError):
                        setup.run_install_command(['download'],env)
                    self.assertEqual(run.call_count,1)

    def test_no_configured_proxy_does_not_reuse_inherited_proxy_or_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            env=dict(WHOLE_HEART_ENV_PREFIX=str(Path(temp)/'envs/test'),
                     HTTPS_PROXY='http://inherited.invalid:1234')
            with patch.object(setup,'_install_child',side_effect=subprocess.CalledProcessError(1,['download'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):setup.run_install_command(['download'],env)
            self.assertEqual(run.call_count,1)

    def test_partial_first_creation_is_updated_on_proxy_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=self.fixture(root);prefix=Path(env['WHOLE_HEART_ENV_PREFIX']);seen=[]
            def run(command, selected):
                seen.append(command[:])
                if len(seen)==1:
                    prefix.mkdir(parents=True)
                    raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0)
            command=['conda','env','create','--prefix',str(prefix),'--file','fixed.yml','--yes']
            with patch.object(setup,'_install_child',side_effect=run):setup.run_install_command(command,env)
            self.assertEqual(seen[0][2],'create')
            self.assertEqual(seen[1][2],'update')
            self.assertNotIn('--yes',seen[1])
            self.assertTrue(prefix.is_dir())

    def test_installer_uses_explicit_direct_opener_then_proxy_opener(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=self.fixture(root);response=io.BytesIO(b'installer')
            response.headers={'Content-Length':'9'}
            calls=[]
            def open_url(url,selected,timeout):
                calls.append(dict(selected))
                if len(calls)==1:raise OSError('direct connection failed')
                return response
            with patch.object(setup,'_open_installer',side_effect=open_url):
                setup.download_installer(root/'installer.sh',env)
            self.assertEqual(calls,[{}, {'http':'http://127.0.0.1:17891','https':'http://127.0.0.1:17891'}])
            self.assertEqual((root/'installer.sh').read_bytes(),b'installer')

    def test_explicit_installer_proxy_is_not_bypassed_by_inherited_no_proxy(self):
        from urllib.request import Request
        request=Request(setup.URL)
        handler=setup._ExplicitProxyHandler({'https':'http://127.0.0.1:17891'})
        with patch.dict(os.environ,{'NO_PROXY':'*','no_proxy':'*'}):
            handler.proxy_open(request,'http://127.0.0.1:17891','https')
        self.assertEqual(request.host,'127.0.0.1:17891')
        self.assertEqual(request._tunnel_host,'github.com')


if __name__=='__main__':unittest.main()
