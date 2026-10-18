"""Linux lock lifetime for a coordinator with workers from multiple folds."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux flock")
class CrossFoldLocks(unittest.TestCase):
    def test_workers_keep_root_and_fold_locks_after_coordinator_death(self):
        script = """import os, sys
from contextlib import ExitStack
from pathlib import Path
from chd_ct.recovery import output_lock
from chd_ct.imagechd.stage_queue import run_queue
root = Path(sys.argv[1])
with ExitStack() as stack:
    fds = tuple(stack.enter_context(output_lock(root / key))
                for key in ('', 'fold_0', 'fold_1'))
    run_queue(['fold_0/blood2d', 'fold_1/crop64'], ['GPU-a', 'GPU-b'], root,
              lambda task: [sys.executable, '-c', 'import time; time.sleep(30)'],
              os.environ.copy(), lock_fds=fds, dependencies={},
              log_path_for=lambda task: root / (task + '.log'))
"""
        probe = """import sys
from chd_ct.recovery import output_lock
with output_lock(sys.argv[1]): pass
"""
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = subprocess.Popen(
                [sys.executable, "-c", script, directory],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            children = []
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    queue = root / "queue.json"
                    if queue.exists():
                        stages = json.loads(queue.read_text())["stages"]
                        children = [r["pid"] for r in stages.values() if "pid" in r]
                        if len(children) == 2:
                            break
                    self.assertIsNone(parent.poll(), "coordinator exited before both folds started")
                    time.sleep(0.05)
                self.assertEqual(len(children), 2)
                parent.kill()
                parent.wait(timeout=5)
                for key in ("", "fold_0", "fold_1"):
                    denied = subprocess.run(
                        [sys.executable, "-c", probe, str(root / key)],
                        env=env, capture_output=True, text=True, timeout=5,
                    )
                    self.assertNotEqual(denied.returncode, 0, key)
                    self.assertIn("正在写入", denied.stderr)
                for child in children:
                    os.kill(child, signal.SIGTERM)
                for key in ("", "fold_0", "fold_1"):
                    deadline = time.monotonic() + 5
                    while True:
                        allowed = subprocess.run(
                            [sys.executable, "-c", probe, str(root / key)],
                            env=env, capture_output=True, text=True, timeout=5,
                        )
                        if allowed.returncode == 0 or time.monotonic() >= deadline:
                            break
                        time.sleep(0.05)
                    self.assertEqual(allowed.returncode, 0, allowed.stderr)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)
                for child in children:
                    try:
                        os.kill(child, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
