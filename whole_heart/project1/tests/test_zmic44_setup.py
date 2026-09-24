"""只模拟硬件查询与后台提交，不安装软件、不占用GPU。"""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import zmic44_setup as wizard


class WizardTests(unittest.TestCase):
    def test_failed_first_binding_allows_only_lock_file(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / '.run.lock').touch()
            wizard.validate_work_directory(work)
            (work / 'unrelated.txt').touch()
            with self.assertRaises(ValueError):
                wizard.validate_work_directory(work)

    def test_single_gpu_uuid_and_refuse_active_card(self):
        rows = '0, GPU-zero, NVIDIA RTX 3090, 23000, 0\n1, GPU-one, NVIDIA RTX 3090, 24000, 0\n'
        with patch.object(wizard.subprocess, 'check_output', side_effect=[rows, '']):
            self.assertEqual(wizard.select_gpu('1'), 'GPU-one')
        with patch.object(wizard.subprocess, 'check_output', side_effect=[rows, 'GPU-one, 99\n']):
            with self.assertRaises(RuntimeError):
                wizard.select_gpu('1')
        with patch.object(wizard.subprocess, 'check_output', return_value=rows):
            with self.assertRaises(ValueError):
                wizard.select_gpu('0,1')

    def test_environment_is_personal_and_no_hardcoded_local_proxy(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with patch.dict(wizard.os.environ, {'PYTHONPATH': 'wrong', 'PYTHONHOME': 'wrong'}, clear=True):
                env = wizard.environment(base)
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('PYTHONHOME', env)
            self.assertNotIn('HTTPS_PROXY', env)
            self.assertEqual(env['WHOLE_HEART_ENV_PREFIX'], str(base / 'envs/project1_py311'))
            for name in ['CONDA_PKGS_DIRS', 'PIP_CACHE_DIR', 'TMPDIR']:
                self.assertTrue(Path(env[name]).is_relative_to(base))

    def test_invalid_installer_never_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            env = wizard.environment(base)
            (base / 'cache' / wizard.INSTALLER).write_bytes(b'not an installer')
            with patch.object(wizard.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    wizard.ensure_conda(base, env)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
