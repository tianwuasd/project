import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dataset1282_analysis import gpu_environment as ge


class GpuEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)/'runtime'

    def ready(self, path=None):
        return dict(ready=True, python=str(path or Path(self.temp.name)/'existing/bin/python'),
                    cuda='12.6', cuda_available=True, device_count=2)

    def report(self):
        return json.loads((self.runtime/'training-environment.json').read_text())

    def test_reuse_never_installs_or_changes_external_environment(self):
        selected = self.ready(Path(self.temp.name)/"space and ' quote"/'bin/python')
        with patch.object(ge, 'scan', return_value=([selected], selected)), \
             patch.object(ge, 'install') as install, patch.object(ge, 'gpu_driver') as driver:
            result = ge.ensure(self.runtime)
        install.assert_not_called()
        driver.assert_not_called()
        self.assertEqual(result['action'], 'reuse_without_changes')
        text = (self.runtime/'training-env.sh').read_text()
        self.assertEqual(shlex.split(text), ['export', 'PY='+selected['python']])
        self.assertFalse((self.runtime/'training').exists())
        self.assertEqual(self.report()['state'], 'ready')

    def test_missing_runtime_installs_only_separate_training_environment(self):
        with patch.object(ge, 'scan', return_value=([{'ready': False}], None)), \
             patch.object(ge, 'gpu_driver', return_value=['560.0']), \
             patch.object(ge, 'install', return_value=self.ready()) as install:
            result = ge.ensure(self.runtime)
        install.assert_called_once_with(self.runtime.resolve(), None, None)
        self.assertEqual(result['action'], 'install_separate_training_environment')

    def test_hidden_gpu_does_not_trigger_download(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(ge, 'scan', return_value=([], None)), patch.object(ge, 'install') as install:
            with self.assertRaisesRegex(RuntimeError, 'hides all GPUs'):
                ge.ensure(self.runtime)
        install.assert_not_called()
        self.assertFalse((self.runtime/'training-env.sh').exists())

    def test_cuda_installed_but_unavailable_is_not_reinstalled(self):
        row = dict(ready=False, cuda='13.2', cuda_available=False)
        with patch.object(ge, 'scan', return_value=([row], None)), \
             patch.object(ge, 'gpu_driver', return_value=['560.0']), patch.object(ge, 'install') as install:
            with self.assertRaisesRegex(RuntimeError, 'GPU access failed'):
                ge.ensure(self.runtime)
        install.assert_not_called()

    def test_failed_install_removes_old_pointer_and_records_failure(self):
        self.runtime.mkdir()
        (self.runtime/'training-env.sh').write_text('stale success')
        with patch.object(ge, 'scan', return_value=([], None)), \
             patch.object(ge, 'gpu_driver', return_value=['560.0']), \
             patch.object(ge, 'install', side_effect=RuntimeError('network failed')):
            with self.assertRaisesRegex(RuntimeError, 'network failed'):
                ge.ensure(self.runtime)
        self.assertFalse((self.runtime/'training-env.sh').exists())
        self.assertEqual(self.report()['state'], 'failed')
        self.assertIn('network failed', self.report()['error'])

    def test_install_does_not_mark_failed_final_probe_ready(self):
        with patch.object(ge, 'scan', return_value=([], None)), \
             patch.object(ge, 'gpu_driver', return_value=['560.0']), \
             patch.object(ge, 'install', return_value=dict(ready=False, errors={'api':'broken'})):
            with self.assertRaisesRegex(RuntimeError, 'failed verification'):
                ge.ensure(self.runtime)
        self.assertFalse((self.runtime/'training-env.sh').exists())
        self.assertEqual(self.report()['selected']['errors'], {'api':'broken'})

    def test_install_refuses_unowned_existing_directory(self):
        target = self.runtime/'training'
        target.mkdir(parents=True)
        keep = target/'user-file'
        keep.write_text('unchanged')
        with patch.object(ge, 'run_logged') as run:
            with self.assertRaisesRegex(RuntimeError, 'not owned'):
                ge.install(self.runtime)
        run.assert_not_called()
        self.assertEqual(keep.read_text(), 'unchanged')

    def test_check_does_not_create_runtime(self):
        with patch.object(ge, 'scan', return_value=([], None)), patch.object(ge, 'install') as install:
            self.assertEqual(ge.main(['--check','--runtime',str(self.runtime)]), 1)
        install.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_interpreter_does_not_resolve_final_symlink(self):
        # Works on Windows without requiring privileges to create a symlink.
        with patch.object(Path, 'resolve', side_effect=AssertionError('venv escaped')):
            path = ge.interpreter(self.runtime/'bin/python')
        self.assertEqual(path, self.runtime/'bin/python')

    def test_only_owned_target_is_used_by_install_commands(self):
        self.runtime.mkdir()
        base = str(Path(self.temp.name)/'base/python')
        with patch.object(ge, 'choose_base', return_value=base), \
             patch.object(ge, 'run_logged') as run, patch.object(ge.subprocess, 'run'), \
             patch.object(ge, 'probe', return_value=self.ready()):
            ge.install(self.runtime)
        commands = [c.args[0] for c in run.call_args_list]
        target_python = str(ge.venv_python(self.runtime/'training'))
        self.assertEqual(commands[0], [base,'-m','venv',str(self.runtime/'training')])
        self.assertTrue(all(c[0] == target_python for c in commands[1:]))
        self.assertIn('torch==2.12.1+cu126', commands[1])
        self.assertIn('https://download.pytorch.org/whl/cu126', commands[1])


if __name__ == '__main__':
    unittest.main()
