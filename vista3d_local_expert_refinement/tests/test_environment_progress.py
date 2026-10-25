import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import runtime_setup as setup
from server_data import read_json


class EnvironmentTests(unittest.TestCase):
    def test_private_pip_destination_and_current_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.dict(os.environ, {'PIP_TARGET':'/elsewhere','PIP_PREFIX':'/bad','PIP_USER':'1','WHOLE_HEART_JOB':str(root/'job')}):
                env=setup.environment(root/'runtime')
            for key in ('PIP_TARGET','PIP_PREFIX','PIP_USER'):
                self.assertNotIn(key, env)
            self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)
            setup.report_progress(env, 'pytorch', '安装固定 PyTorch')
            record=read_json(root/'job/environment.json')
            self.assertEqual(record['phase'],'pytorch')
            self.assertEqual(record['status'],'running')

    def test_download_records_real_bytes_and_only_publishes_after_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); dest=root/'installer'
            env={'WHOLE_HEART_JOB':str(root/'job')}
            response=io.BytesIO(b'x'*20000)
            response.headers={'Content-Length':'20000'}
            with patch.object(setup.urllib.request,'urlopen',return_value=response):
                setup.download_installer(dest, env)
            record=read_json(root/'job/environment.json')
            self.assertEqual(record['downloaded_bytes'],20000)
            self.assertEqual(record['total_bytes'],20000)
            self.assertEqual(dest.stat().st_size,20000)

    def test_bad_installer_keeps_file_and_never_runs_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'cache').mkdir()
            (root/'cache'/setup.INSTALLER).write_bytes(b'corrupt')
            with patch.object(setup.subprocess,'run') as run, self.assertRaises(RuntimeError):
                setup.ensure_conda(root, {})
            run.assert_not_called()


if __name__=='__main__': unittest.main()
