"""No descriptor/CWD bootstrap assumptions; code-only updater boundaries."""
from pathlib import Path
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


class PreprocessingCLI(unittest.TestCase):
    def test_missing_descriptor_stays_none_not_code_root(self):
        from chd_stunet.cli import main
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            previous=Path.cwd()
            os.chdir(temporary)
            try:
                with patch('chd_stunet.cli.read_config',return_value=({},{})), \
                     patch('chd_stunet.cli.prepare_paths',return_value=dict(raw_root=temporary)), \
                     patch('chd_stunet.preprocessing.run.run_prepare',return_value=dict(failed_cases=[])) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['prepare','--output','outputs/cli-contract','--fold','1']),0)
                    self.assertIsNone(run.call_args.kwargs['descriptor'])
                    self.assertEqual(run.call_args.args[0],Path(temporary).resolve())
                    self.assertEqual(run.call_args.args[1],(ROOT/'outputs/cli-contract').resolve())
            finally:
                os.chdir(previous)

    def test_list_methods_does_not_load_dataset(self):
        from chd_stunet.cli import main
        with patch('chd_stunet.cli.read_config',side_effect=AssertionError('must be lazy')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['prepare','--list-methods']),0)

    def test_updater_refuses_private_data_and_parent_paths(self):
        from update import allowed,validate_manifest
        for path in ('project.local.json','outputs/data.json','../other/run.py','/run.py','src/../../secret.py'):
            self.assertFalse(allowed(path),path)
        self.assertTrue(allowed('src/chd_stunet/preprocessing/run.py'))
        with self.assertRaises(ValueError):
            validate_manifest(dict(project='chd-ct-panel-v2',files={'outputs/data.json':'0'*64}))


if __name__=='__main__':
    unittest.main()
