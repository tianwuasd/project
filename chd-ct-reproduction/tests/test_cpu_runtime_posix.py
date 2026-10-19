"""Real Linux CPU-worker lifetime tests, using stdlib stubs rather than CUDA."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "requires Linux flock and signals")
class CPUWorkerLifetime(unittest.TestCase):
    def test_children_keep_output_and_runtime_locks_after_coordinator_death(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "torch.py").write_text("def set_num_threads(n): pass\n")
            (root / "payload.py").write_text("import time\ndef hold(value):\n time.sleep(30)\n return value\n")
            env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + str(ROOT / "src"))
            script = """import os, sys
from pathlib import Path
from chd_ct.cpu_runtime import iter_cases
from chd_ct.recovery import output_lock
root=Path(sys.argv[1])
with output_lock(root/'runtime') as runtime, output_lock(root/'output') as out:
 os.environ['CHD_RUNTIME_LOCK_FD']=str(runtime)
 list(iter_cases('payload','hold',[{'value':1},{'value':2}],root/'output',
                 {'processes':2,'threads':1},(out,)))
"""
            parent = subprocess.Popen([sys.executable, "-c", script, directory], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            children = []
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    queues = list((root / "output").glob(".cpu-work/*/queue.json"))
                    if queues:
                        children = [r["pid"] for r in json.loads(queues[0].read_text())["stages"].values()]
                        if len(children) == 2:
                            break
                    self.assertIsNone(parent.poll())
                    time.sleep(0.05)
                self.assertEqual(len(children), 2)
                parent.kill()
                parent.wait(timeout=5)
                for name in ("output", "runtime"):
                    probe = subprocess.run([sys.executable, "-c",
                         "import sys; from chd_ct.recovery import output_lock\nwith output_lock(sys.argv[1]): pass",
                         str(root / name)], env=env, capture_output=True, timeout=5)
                    self.assertNotEqual(probe.returncode, 0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)
                for child in children:
                    try:
                        os.kill(child, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def test_worker_failure_terminates_other_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "torch.py").write_text("def set_num_threads(n): pass\n")
            (root / "payload.py").write_text(
                "import time\ndef work(value):\n if value == 1: raise ValueError('injected')\n time.sleep(30)\n return value\n")
            env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + str(ROOT / "src"))
            script = """import sys
from pathlib import Path
from chd_ct.cpu_runtime import iter_cases
list(iter_cases('payload','work',[{'value':1},{'value':2}],Path(sys.argv[1]),
                {'processes':2,'threads':1}))
"""
            result = subprocess.run([sys.executable, "-c", script, directory], env=env,
                                    capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"injected", result.stderr)
            from chd_ct.server.jobs import identity

            queue = next(root.glob(".cpu-work/*/queue.json"))
            for record in json.loads(queue.read_text())["stages"].values():
                self.assertIsNone(identity(record["pid"]))
