import sys
import tempfile
import subprocess
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'skill' / 'scripts'))
from new_run import create_run


class RunTests(unittest.TestCase):
    def test_cli_runs_without_external_timezone_database(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'cli_run'
            script = Path(__file__).parents[1] / 'skill/scripts/new_run.py'
            result = subprocess.run([sys.executable, '-X', 'utf8', str(script), '测试', '--output', str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((target / 'state.json').exists())

    def test_new_workspace_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'run'
            create_run('研究问题', p)
            self.assertTrue((p / 'state.json').is_file())
            with self.assertRaises(FileExistsError):
                create_run('另一个问题', p)
