"""Actual SIGINT cleanup with the standard-library-only resume verifier."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform == "linux", "Real Linux signals and mount discovery")
class VerificationProcessTests(unittest.TestCase):
    def test_sigint_stops_all_readers_and_leaves_source_unchanged(self):
        self.interrupt_readers(repeated=False)

    def test_repeated_sigint_cannot_escape_reader_cleanup(self):
        self.interrupt_readers(repeated=True)

    def interrupt_readers(self, repeated):
        with tempfile.TemporaryDirectory(prefix="chd_verify_") as folder:
            root = Path(folder)
            for i in range(4):
                (root / f"{i}.bin").write_bytes(b"x" * (8 * 1024**2))
            before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir()}
            script = """
import sys, threading, time
from pathlib import Path
from chd_ct.server.resume import tree_hashes
from chd_ct.server import verification as v
root=Path(sys.argv[1])
record=dict(inputs=tree_hashes(root), outputs={}, input_roots=[str(root)], output_roots=[])
v.choose_threads=lambda *a:dict(workers=2,cpus=8,budget=6,storage_cap=2)
original=v.ReadProgress.advance
def slow(self, size):
    original(self,size)
    (root/'.reading').touch()
    time.sleep(0.6)
v.ReadProgress.advance=slow
shutdown=v.ThreadPoolExecutor.shutdown
def observed_shutdown(self,*a,**kw):
    (root/'.cleanup').touch()
    time.sleep(0.2)
    return shutdown(self,*a,**kw)
v.ThreadPoolExecutor.shutdown=observed_shutdown
try:
    v.verify_snapshots([('train',record)])
except KeyboardInterrupt:
    assert not any(t.name.startswith('chd-verify-') for t in threading.enumerate())
    print('READERS_JOINED',flush=True)
    sys.exit(7)
sys.exit(8)
"""
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            proc = subprocess.Popen(
                [sys.executable, "-c", script, str(root)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not (root / ".reading").exists() and proc.poll() is None:
                    if time.monotonic() > deadline:
                        self.fail("reader did not start")
                    time.sleep(0.01)
                proc.send_signal(signal.SIGINT)
                if repeated:
                    deadline = time.monotonic() + 5
                    while not (root / ".cleanup").exists() and proc.poll() is None:
                        if time.monotonic() > deadline:
                            self.fail("cleanup did not start")
                        time.sleep(0.01)
                    proc.send_signal(signal.SIGINT)
                out, err = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 7, out + err)
                self.assertIn("READERS_JOINED", out)
                self.assertIn("校验中断", out)
                self.assertNotIn("校验通过", out)
                self.assertEqual(before, {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in before})
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    unittest.main()
