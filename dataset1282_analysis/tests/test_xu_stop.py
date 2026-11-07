import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StopTests(unittest.TestCase):
    def test_real_task_stops_and_unrelated_process_survives(self):
        self.exercise_stop(False)

    def test_repository_pipe_can_stop_before_updating_checkout(self):
        self.exercise_stop(True)

    def test_waiting_for_resources_can_be_stopped_without_signalling_foreign_process(self):
        self.exercise_stop(False,waiting=True)

    def exercise_stop(self, piped, waiting=False):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            task = subprocess.Popen([sys.executable,'-B',str(ROOT/'tests/fixtures/xu_owned_sleep.py'),
                                     '--output',str(out/'task'),*(['--wait-resources'] if waiting else [])])
            foreign = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
            self.addCleanup(foreign.wait)
            self.addCleanup(foreign.terminate)
            try:
                deadline = time.monotonic()+10
                while not (out/'task/ready').exists() and time.monotonic()<deadline:
                    time.sleep(.03)
                self.assertTrue((out/'task/ready').exists())
                if waiting:
                    while not (out/'task/status.json').exists() and time.monotonic()<deadline:
                        time.sleep(.03)
                    self.assertEqual(json.loads((out/'task/status.json').read_text())['state'],'waiting_resources')
                script = ROOT/'scripts/xu_stop.py'
                result = subprocess.run([sys.executable,'-B','-' if piped else str(script),str(out/'task')],
                                        input=script.read_text(encoding='utf-8') if piped else None,
                                        capture_output=True,text=True,encoding='utf-8',cwd=ROOT,timeout=15)
                self.assertEqual(result.returncode,0,(result.stdout or '')+(result.stderr or ''))
                task.wait(timeout=5)
                self.assertIsNone(foreign.poll())
            finally:
                if task.poll() is None:
                    task.terminate()
                task.wait()

    def test_changed_birth_is_rejected_without_signalling(self):
        import os
        import socket
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'.runtime').mkdir()
            (out/'.runtime/operation.lock').write_text(json.dumps(dict(pid=os.getpid(),birth='mismatch',
                  hostname=socket.gethostname(),kind='xu-model-run',run_dir=str(out),nonce='test',workers=[])))
            result = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/xu_stop.py'),str(out)],
                                    capture_output=True,text=True,encoding='utf-8',cwd=ROOT,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('身份',result.stderr)
