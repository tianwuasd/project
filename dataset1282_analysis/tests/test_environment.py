import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class EnvironmentTests(unittest.TestCase):
    def test_environment_entry_can_plan_without_creating_environment(self):
        self.assertTrue((ROOT / 'setup_env.py').is_file(), 'standalone environment entry is missing')
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / '中文 space' / 'environment'
            p = subprocess.run([sys.executable, str(ROOT / 'setup_env.py'), '--plan', '--env', str(env)],
                               cwd=d, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)['environment'], str(env))
            self.assertFalse(env.exists())

    def test_missing_environment_check_is_read_only(self):
        self.assertTrue((ROOT / 'setup_env.py').is_file(), 'standalone environment entry is missing')
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / 'absent'
            p = subprocess.run([sys.executable, str(ROOT / 'setup_env.py'), '--check', '--env', str(env)],
                               capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('missing_interpreter', p.stdout)
            self.assertFalse(env.exists())

    def test_config_rejects_unknown_fields_and_output_inside_data(self):
        self.assertTrue((ROOT / 'src/dataset1282_analysis/config.py').is_file(), 'config implementation missing')
        from dataset1282_analysis.config import resolve_config
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / 'local.json'
            local.write_text(json.dumps({'unknown': True}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                resolve_config({}, local, ROOT)
            local.write_text(json.dumps({'data_root': d, 'work_root': str(Path(d) / 'out')}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                resolve_config({}, local, ROOT)


if __name__ == '__main__':
    unittest.main()
