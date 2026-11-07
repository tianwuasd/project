import os
from pathlib import Path
import sys
import tempfile
import unittest

from dataset1282_analysis import gpu_environment as ge
from dataset1282_analysis.gpu_query_environment import ensure_binding, probe_binding
from test_xu_acceleration_live import environment


class GpuEnvironmentTests(unittest.TestCase):
    def test_missing_interpreter_is_reported_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory)/'missing/python'
            self.assertEqual(ge.probe(missing)['error'],'missing_interpreter')
            self.assertFalse(missing.parent.exists())

    def test_actual_interpreter_probe(self):
        result = ge.probe(sys.executable)
        self.assertEqual(result['python'],str(ge.interpreter(sys.executable)))
        self.assertIn('torch',result['versions'])
        self.assertNotIn('torch_api',result['errors'])
        self.assertNotIn('data_api',result['errors'])

    def test_unowned_install_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'training'
            target.mkdir()
            keep = target/'user-file'
            keep.write_text('unchanged',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'not owned'):
                ge.install(root)
            self.assertEqual(keep.read_text(),'unchanged')

    def test_hidden_allocation_rejected_before_driver_query(self):
        with environment(CUDA_VISIBLE_DEVICES=''):
            with self.assertRaisesRegex(RuntimeError,'hides all GPUs'):
                ge.gpu_driver()

    def test_actual_query_dependency_in_private_target(self):
        paths = [Path(p) for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p]
        candidates = [p for p in paths if (p/'cuda/bindings').is_dir()]
        if not candidates:
            self.skipTest('当前未配置隔离的 cuda-bindings 测试依赖')
        result = probe_binding(sys.executable,candidates[0])
        self.assertTrue(result['ready'])
        self.assertGreater(result['devices'],0)

    def test_query_check_does_not_install(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)/'absent'
            selected = dict(python=sys.executable,python_version=list(sys.version_info[:3]))
            result = ensure_binding(runtime,selected,install=False)
            self.assertIn('ready',result)
            self.assertFalse(runtime.exists())

    def test_interpreter_keeps_requested_venv_path(self):
        path = Path.cwd()/'environment'/'bin'/'python'
        self.assertEqual(ge.interpreter(path),path)
