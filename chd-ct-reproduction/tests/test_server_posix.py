"""Real POSIX process/lock integration with a synthetic stdlib-only payload.

Run without pytest: python3 -m unittest discover -s tests -p test_server_posix.py -v
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = """import json, os, time
from pathlib import Path

def run(args, profile, *, output_dir=None, lock_held=False, recent_gpus=()):
    output_dir.mkdir(parents=True)
    print('PAYLOAD '+args.task, flush=True)
    (output_dir/'started').write_text(str(os.getpid()))
    time.sleep(profile.get('fixture_sleep', 0.3))
    if profile.get('fixture_fail') == args.task:
        return 7
    (output_dir/'passed').write_text(args.task)
    return 0
"""


@unittest.skipUnless(sys.platform == "linux", "Real session/flock tests require Linux")
class DetachedJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="chd_job_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "code"
        self.root.mkdir()
        shutil.copytree(
            ROOT / "src/chd_ct", self.root / "src/chd_ct", ignore=shutil.ignore_patterns("__pycache__")
        )
        shutil.copy(ROOT / "start_server.py", self.root / "start_server.py")
        (self.root / "src/chd_ct/server/imagechd.py").write_text(PAYLOAD)
        self.runtime = Path(self.temp.name) / "runtime"
        self.results = Path(self.temp.name) / "results"
        self.profile = dict(
            runtime=str(self.runtime),
            results=str(self.results),
            prepared=str(Path(self.temp.name) / "prepared"),
            source_updated="fixture",
            fixture_sleep=1.2,
        )
        self.config = Path(self.temp.name) / "profile.json"
        self.config.write_text(json.dumps(self.profile))
        self.command = [
            sys.executable,
            str(self.root / "start_server.py"),
            "--server-config",
            str(self.config),
            "--device",
            "cpu",
            "--python",
            sys.executable,
            "--non-interactive",
        ]
        self.pid = None
        self.addCleanup(self.stop_if_running)

    def stop_if_running(self):
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.state().get("status") not in {"queued", "running"}:
                    return
                time.sleep(0.05)

    def start(self, *options):
        p = subprocess.run(self.command + list(options), capture_output=True, text=True, timeout=10)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.directory = Path(json.loads((self.runtime / "last-job.json").read_text())["directory"])
        self.pid = json.loads((self.directory / "receipt.json").read_text())["pid"]
        return p

    def state(self):
        return json.loads((self.directory / "job_state.json").read_text())

    def finished(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = self.state()
            if state["status"] not in {"queued", "running"}:
                return state
            time.sleep(0.05)
        self.fail((self.directory / "launcher.log").read_text())

    def test_parent_exits_worker_survives_and_lock_prevents_duplicate(self):
        self.start("--tasks", "environment,diagnosis-demo")
        os.kill(self.pid, 0)
        self.assertEqual(os.getsid(self.pid), self.pid)
        denied = subprocess.run(
            self.command + ["--task", "diagnosis-demo"], capture_output=True, text=True, timeout=10
        )
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("已有任务", denied.stderr)
        state = self.finished()
        self.assertEqual(state["status"], "passed")
        self.assertEqual([s["status"] for s in state["steps"]], ["passed", "passed"])
        self.assertIn("PAYLOAD", (self.directory / "launcher.log").read_text())
        # Same runtime becomes available after the worker closes the inherited lock.
        self.start("--tasks", "diagnosis-demo")
        self.assertEqual(self.finished()["status"], "passed")

    def test_failure_stops_following_step_and_status_reports_it(self):
        self.profile["fixture_fail"] = "environment"
        self.config.write_text(json.dumps(self.profile))
        self.start("--tasks", "environment,diagnosis-demo")
        state = self.finished()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["steps"][1]["status"], "pending")
        self.assertFalse(Path(state["steps"][1]["output"]).exists())
        result = subprocess.run(self.command + ["--status"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("failed", result.stdout)

    def test_signal_records_interruption_and_releases_lock(self):
        self.profile["fixture_sleep"] = 5
        self.config.write_text(json.dumps(self.profile))
        self.start("--tasks", "environment,diagnosis-demo")
        deadline = time.monotonic() + 5
        while self.state()["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.05)
        os.kill(self.pid, signal.SIGTERM)
        state = self.finished()
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["steps"][1]["status"], "pending")

    def test_foreground_waits_and_propagates_failure(self):
        self.profile.update(fixture_fail="environment", fixture_sleep=0.1)
        self.config.write_text(json.dumps(self.profile))
        result = subprocess.run(
            self.command + ["--task", "environment", "--foreground"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("PAYLOAD environment", result.stdout)


if __name__ == "__main__":
    unittest.main()
