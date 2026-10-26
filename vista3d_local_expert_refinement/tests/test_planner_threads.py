"""Planning threads and serial training augmentation are different contracts."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]


def configure_for(root):
    # Exercise the actual helper without importing imaging/ML dependencies.
    tree = ast.parse((CODE / '05_src/prepare_nnunet.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_paths')
    namespace = dict(os=os, ROOT=Path(root))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CODE / '05_src/prepare_nnunet.py'), 'exec'), namespace)
    return namespace['configure_paths']


class PlannerThreads(unittest.TestCase):
    root = CODE / 'synthetic-path-only-not-created'

    def test_planning_overrides_inherited_serial_da_with_shared_cpu_budget(self):
        root = self.root
        with patch.dict(os.environ, {'WHOLE_HEART_CPU_THREADS': '4', 'nnUNet_n_proc_DA': '0'}, clear=True):
            configure_for(root)()
            self.assertEqual(os.environ['nnUNet_n_proc_DA'], '4')
            self.assertEqual(os.environ['nnUNet_preprocessed'], str(Path(root) / '04_data/derived/nnUNet_preprocessed'))

    def test_without_cpu_budget_planning_uses_one_thread(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_for(self.root)()
            self.assertEqual(os.environ['nnUNet_n_proc_DA'], '1')

    def test_case_worker_uses_its_own_single_thread_budget(self):
        with patch.dict(os.environ, {'WHOLE_HEART_CPU_THREADS': '1', 'nnUNet_n_proc_DA': '4'}, clear=True):
            configure_for(self.root)()
            self.assertEqual(os.environ['nnUNet_n_proc_DA'], '1')

    def test_nonpositive_cpu_budget_cannot_reach_torch_as_zero(self):
        for value in ['0', '-1']:
            with self.subTest(value=value), patch.dict(os.environ, {'WHOLE_HEART_CPU_THREADS': value}, clear=True):
                configure_for(self.root)()
                self.assertEqual(os.environ['nnUNet_n_proc_DA'], '1')

    def test_training_smoke_and_evaluation_still_explicitly_use_serial_da(self):
        for name in ['server_train.py', 'smoke_train.py', 'evaluation.py', 'evaluation_parallel.py']:
            with self.subTest(name=name), patch.dict(os.environ, {'WHOLE_HEART_CPU_THREADS': '4'}, clear=True):
                tree = ast.parse((CODE / '05_src' / name).read_text(encoding='utf-8'))
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
                start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'configure_paths')
                # Execute the real configuration statements before Torch is imported.
                prefix = ast.Module(body=main.body[start:start+2], type_ignores=[])
                exec(compile(prefix, name, 'exec'), dict(os=os, configure_paths=configure_for(self.root)))
                self.assertEqual(os.environ['nnUNet_n_proc_DA'], '0')


if __name__ == '__main__':
    unittest.main()
