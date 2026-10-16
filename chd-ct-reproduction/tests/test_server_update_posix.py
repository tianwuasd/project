"""Real Linux cleanup and inherited runtime-lock checks, stdlib only."""

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
from chd_ct.server import inspection, jobs, updater  # noqa: E402


@unittest.skipUnless(sys.platform == "linux", "Requires Linux process groups and flock")
class UpdateProcessTests(unittest.TestCase):
    def test_timeout_kills_git_and_helper_processes(self):
        with tempfile.TemporaryDirectory(prefix="chd_git_timeout_") as directory:
            root = Path(directory)
            executable = root / "git"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import subprocess, sys, time\nfrom pathlib import Path\n"
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"Path({str(root / 'child')!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            )
            executable.chmod(0o700)
            with patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ["PATH"]}):
                with self.assertRaises(subprocess.TimeoutExpired):
                    updater.git(root, "fetch", timeout=0.5)
            child = int((root / "child").read_text())
            for _ in range(30):
                if jobs.identity(child) is None:
                    break
                time.sleep(0.1)
            self.assertIsNone(jobs.identity(child), "Git helper survived timeout")

    def test_interrupt_during_spawn_reaps_process(self):
        actual = subprocess.Popen
        spawned = []

        def interrupted_spawn(*args, **kwargs):
            child = actual([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            spawned.append(child)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return child

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(updater.subprocess, "Popen", interrupted_spawn):
                with self.assertRaises(KeyboardInterrupt):
                    updater.git(Path(directory), "fetch")
        self.assertIsNotNone(spawned[0].poll())

    def test_inherited_child_lock_prevents_update(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with jobs.acquire_lock(runtime) as lock:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"], pass_fds=(lock.fileno(),)
                )
            try:
                with self.assertRaisesRegex(ValueError, "尚未完全退出"):
                    with updater.idle_lock(runtime, None, timeout=0):
                        self.fail("updated while a child held the lock")
            finally:
                child.terminate()
                child.wait(timeout=5)
            with patch.object(inspection, "latest", return_value=None):
                with updater.idle_lock(runtime, None, timeout=0):
                    pass


if __name__ == "__main__":
    unittest.main()
