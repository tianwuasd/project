"""Missing runtime dependencies must fail before data work, even with a ready marker."""
import importlib
import importlib.metadata
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import runtime_setup
import task_resources
import software_smoke
from server_data import read_json
import tempfile


class DependencyTests(unittest.TestCase):
    def versions(self, name):
        return {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
                'nnunetv2': '2.8.1', 'threadpoolctl': '3.6.0'}.get(name, 'test')

    def test_missing_psutil_fails_in_environment_check(self):
        def load(name):
            if name == 'psutil':
                raise ModuleNotFoundError("No module named 'psutil'")
            return SimpleNamespace()
        with patch.object(importlib.metadata, 'version', side_effect=self.versions), \
                patch.object(importlib, 'import_module', side_effect=load):
            with self.assertRaisesRegex(RuntimeError, 'psutil'):
                runtime_setup.verify_dependencies()

    def test_broken_native_import_is_also_reported(self):
        def load(name):
            if name == 'blosc2':
                raise OSError('native library unavailable')
            return SimpleNamespace()
        with patch.object(importlib.metadata, 'version', side_effect=self.versions), \
                patch.object(importlib, 'import_module', side_effect=load):
            with self.assertRaisesRegex(RuntimeError, 'blosc2.*native library'):
                runtime_setup.verify_dependencies()

    def test_real_resource_observer_is_required_and_recorded(self):
        observation = {'post_case_rss_bytes': 1024, 'libraries': [], 'missing_evidence': []}
        with patch.object(importlib.metadata, 'version', side_effect=self.versions), \
                patch.object(task_resources, 'process_observation', return_value=observation) as observe, \
                patch.object(importlib, 'import_module', return_value=SimpleNamespace()):
            value = runtime_setup.verify_dependencies()
        observe.assert_called_once_with()
        self.assertEqual(value['status'], 'passed')
        self.assertEqual(value['process_observation'], observation)
        self.assertIn('psutil', value['imports'])

    def test_software_smoke_writes_dependency_failure_before_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'software-smoke.json'
            with patch.object(runtime_setup, 'verify_dependencies',
                              side_effect=RuntimeError('missing psutil')):
                with self.assertRaisesRegex(RuntimeError, 'psutil'):
                    software_smoke.main(target)
            value = read_json(target)
            self.assertEqual(value['status'], 'failed')
            self.assertIn('psutil', value['error'])
            self.assertFalse(value['full_size_training_ready'])


if __name__ == '__main__':
    unittest.main()
