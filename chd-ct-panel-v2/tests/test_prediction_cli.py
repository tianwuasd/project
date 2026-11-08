from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PredictionCLI(unittest.TestCase):
    def test_help_exposes_prediction_and_smoke(self):
        for entry, arguments in [('run.py', ['predict', '--help']), ('smoke.py', ['--stage', 'predict', '--help'])]:
            process = subprocess.run([sys.executable, str(ROOT/entry), *arguments], capture_output=True,
                                     text=True, encoding='utf-8', timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            for option in ('--training-run', '--background', '--resume', '--gpus', '--skip-producer-audit'):
                self.assertIn(option, process.stdout)

    def test_stop_requires_existing_prediction(self):
        process = subprocess.run([sys.executable, str(ROOT/'run.py'), 'predict', '--stop'],
                                 capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('--output', process.stderr)


if __name__ == '__main__':
    unittest.main()
