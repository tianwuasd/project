"""Server venv dispatch must not confuse symlink targets with interpreter identity."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dataset_bootstrap',ROOT/'bootstrap.py')
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class LaunchTests(unittest.TestCase):
    def test_distinct_venv_paths_remain_distinct_even_with_same_binary(self):
        with patch.object(Path,'resolve',return_value=Path('/same/python')):
            self.assertNotEqual(bootstrap.interpreter_path('/usr/bin/python'),
                                bootstrap.interpreter_path('/research/bin/python'))

    def test_lexical_redundancy_does_not_trigger_reexec(self):
        self.assertEqual(bootstrap.interpreter_path('/env/bin/./python'),
                         bootstrap.interpreter_path('/env/bin/python'))


if __name__=='__main__': unittest.main()
