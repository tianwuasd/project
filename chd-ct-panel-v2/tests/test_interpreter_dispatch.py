"""Interpreter selection must preserve virtualenv identity across binary symlinks."""
from contextlib import ExitStack,redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chd_stunet import cli


class InterpreterDispatchTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=scratch)
        self.folder=Path(self.temporary.name)
        self.python=self.folder/'environment'/'python.exe'
        self.python.parent.mkdir();self.python.touch()

    def tearDown(self):self.temporary.cleanup()

    def context(self):
        stack=ExitStack()
        stack.enter_context(patch.object(cli,'ROOT',self.folder))
        stack.enter_context(patch.object(cli,'read_config',return_value=({},dict(preprocess_python=str(self.python)))))
        stack.enter_context(patch.object(cli,'prepare_paths',return_value=dict(raw_root=str(self.folder))))
        stack.enter_context(patch.dict(os.environ,{'CHD_ANALYSIS_REEXEC':'0'}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_same_resolved_binary_does_not_mean_same_environment(self):
        # Linux venv/bin/python commonly resolves to /usr/bin/python3. The
        # configured lexical path, rather than that shared inode, selects venv.
        with self.context(),patch.object(Path,'resolve',return_value=Path('/same/base/python')), \
             patch.object(cli.subprocess,'call',return_value=17) as launch, \
             patch('chd_stunet.preprocessing.run.run_prepare',return_value={'failed_cases':[]}) as work:
            self.assertEqual(cli.main(['prepare','--fold','0']),17)
            launch.assert_called_once();work.assert_not_called()
            self.assertEqual(os.path.normcase(launch.call_args.args[0][0]),os.path.normcase(str(self.python)))

    def test_already_in_requested_environment_does_not_relaunch(self):
        with self.context(),patch.object(sys,'executable',str(self.python)), \
             patch.object(cli.subprocess,'call') as launch, \
             patch('chd_stunet.preprocessing.run.run_prepare',return_value={'failed_cases':[]}) as work:
            self.assertEqual(cli.main(['prepare','--fold','0']),0)
            launch.assert_not_called();work.assert_called_once()

    def test_mismatched_reexec_is_still_refused(self):
        with self.context(),patch.dict(os.environ,{'CHD_ANALYSIS_REEXEC':'1'}), \
             patch.object(cli.subprocess,'call') as launch:
            with self.assertRaisesRegex(RuntimeError,'configured environment'):
                cli.main(['prepare','--fold','0'])
            launch.assert_not_called()

    def test_real_stdlib_virtualenv_is_the_child_prefix(self):
        environment=self.folder/'real-venv'
        venv.EnvBuilder(with_pip=False).create(environment)
        self.python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        (self.folder/'run.py').write_text(
            "import json,sys;from pathlib import Path;"
            "Path(__file__).with_name('child.json').write_text(json.dumps("
            "{'prefix':sys.prefix,'executable':sys.executable}))",encoding='utf-8')
        with self.context():self.assertEqual(cli.main(['prepare','--fold','0']),0)
        child=json.loads((self.folder/'child.json').read_text())
        self.assertEqual(Path(child['prefix']).resolve(),environment.resolve())


if __name__=='__main__':unittest.main()
