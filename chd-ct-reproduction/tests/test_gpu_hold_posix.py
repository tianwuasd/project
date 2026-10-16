"""Real Linux holder lifetime/lock tests using a fake torch module, no GPU."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chd_ct.recovery import read_json, write_json  # noqa: E402
from chd_ct.server import gpu_hold, jobs, launcher  # noqa: E402


@unittest.skipUnless(sys.platform == "linux", "Requires Linux process identities and flock")
class HoldProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="chd_hold_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "torch.py").write_text(
            "def set_num_threads(n): pass\ndef empty(n, device): return object()\n"
            "class cuda:\n @staticmethod\n def synchronize(): pass\n"
        )
        self.env = patch.dict(
            os.environ,
            {
                "PYTHONPATH": str(self.root) + os.pathsep + str(Path(__file__).resolve().parents[1] / "src"),
                gpu_hold.ENV: "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.folder = gpu_hold.begin(
            dict(directory=str(self.root), steps=[dict(arguments=dict(device="cuda", gpu_count=2))])
        )
        gpu_hold.configure(["GPU-a", "GPU-b", "GPU-c"])
        self.addCleanup(gpu_hold.release, self.folder)
        self.query = patch.object(
            launcher, "query_gpus", return_value=[dict(uuid="GPU-a", busy=False, free=24000, util=0)]
        )
        self.query.start()
        self.addCleanup(self.query.stop)

    def test_reuse_keeps_same_process_and_release_frees_inherited_runtime_lock(self):
        runtime = self.root / "runtime"
        with jobs.acquire_lock(runtime) as lock:
            with patch.dict(os.environ, {"CHD_RUNTIME_LOCK_FD": str(lock.fileno())}):
                self.assertTrue(gpu_hold.acquire("GPU-a"))
        first = gpu_hold.owned_processes()
        self.assertTrue(gpu_hold.acquire("GPU-a"))
        self.assertEqual(first, gpu_hold.owned_processes())
        with self.assertRaises(ValueError):
            with jobs.acquire_lock(runtime):
                self.fail("holder failed to inherit the runtime lock")
        gpu_hold.release(self.folder)
        with jobs.acquire_lock(runtime):
            pass

    def test_owner_death_releases_holder_automatically(self):
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: owner.poll() is None and owner.kill())
        config = read_json(self.folder / "owner.json")
        config.update(pid=owner.pid, identity=jobs.identity(owner.pid))
        write_json(self.folder / "owner.json", config)
        self.assertTrue(gpu_hold.acquire("GPU-a"))
        record = gpu_hold.records(self.folder)[0]
        owner.kill()
        owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while gpu_hold.alive(record) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(gpu_hold.alive(record))

    def test_signal_during_spawn_cleans_holder_and_receipt(self):
        actual = subprocess.Popen
        children = []

        def launch(*args, **kwargs):
            child = actual(*args, **kwargs)
            children.append(child)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return child

        with patch.object(gpu_hold.subprocess, "Popen", launch):
            with self.assertRaises(KeyboardInterrupt):
                gpu_hold.acquire("GPU-a")
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(gpu_hold.records(self.folder), [])


if __name__ == "__main__":
    unittest.main()
