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

    def test_sigkill_worker_retains_child_lock_and_stop_handles_orphan(self):
        module = self.root / "src/chd_ct/server/imagechd.py"
        module.write_text("""import os, sys
from .bootstrap import run_logged
def run(args, profile, *, output_dir=None, **kwargs):
    output_dir.mkdir(parents=True)
    return run_logged([sys.executable, '-c', 'import time; time.sleep(30)'],
                      os.environ.copy(), output_dir / 'server.log')
""")
        self.start("--task", "environment")
        receipt = self.directory / "steps/01_environment/active-command.json"
        deadline = time.monotonic() + 5
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = json.loads(receipt.read_text())["pid"]
        try:
            os.kill(self.pid, signal.SIGKILL)
            time.sleep(0.1)
            denied = subprocess.run(
                self.command + ["--task", "environment"], capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("已有任务", denied.stderr)
            denied = subprocess.run(
                self.command + ["--resume-job"], capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("子进程仍存活", denied.stderr)
            stopped = subprocess.run(self.command + ["--stop"], capture_output=True, text=True, timeout=10)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            time.sleep(0.2)
            module.write_text(PAYLOAD)
            self.start("--task", "environment")
            self.assertEqual(self.finished()["status"], "passed")
        finally:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def test_stage_worker_inherits_output_lock_after_coordinator_sigkill(self):
        output = Path(self.temp.name) / "models"
        worker = Path(self.temp.name) / "stage.py"
        worker.write_text("import time; time.sleep(30)")
        script = """import os, sys
from pathlib import Path
from chd_ct.recovery import output_lock
from chd_ct.imagechd.stage_queue import run_queue
output = Path(sys.argv[1])
with output_lock(output) as fd:
    run_queue(['crop64'], ['GPU-fixture'], output,
              lambda stage: [sys.executable, sys.argv[2]], os.environ.copy(), lock_fds=(fd,))
"""
        env = dict(os.environ, PYTHONPATH=str(self.root / "src"))
        parent = subprocess.Popen(
            [sys.executable, "-c", script, str(output), str(worker)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        child = None
        try:
            queue = output / "queue.json"
            deadline = time.monotonic() + 5
            while not queue.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            child = json.loads(queue.read_text())["stages"]["crop64"]["pid"]
            parent.kill()
            parent.wait(timeout=5)
            probe = """from chd_ct.recovery import output_lock
import sys
with output_lock(sys.argv[1]): pass
"""
            denied = subprocess.run(
                [sys.executable, "-c", probe, str(output)], env=env, capture_output=True, text=True, timeout=5
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("正在写入", denied.stderr)
            os.kill(child, signal.SIGTERM)
            time.sleep(0.2)
            allowed = subprocess.run(
                [sys.executable, "-c", probe, str(output)], env=env, capture_output=True, text=True, timeout=5
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if child:
                try:
                    os.kill(child, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def test_resume_cli_reuses_completed_step_and_archives_interrupted_short_step(self):
        self.start("--tasks", "environment,diagnosis-demo")
        old_job = self.directory
        partial = old_job / "steps/02_diagnosis-demo"
        deadline = time.monotonic() + 5
        while not (partial / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue((partial / "started").exists())
        stopped = subprocess.run(self.command + ["--stop"], capture_output=True, text=True, timeout=10)
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertEqual(self.finished()["status"], "interrupted")
        time.sleep(0.1)
        self.start("--resume-job")
        self.assertEqual(self.directory, old_job)
        state = self.finished()
        self.assertEqual(state["status"], "passed")
        self.assertEqual([s["status"] for s in state["steps"]], ["reused", "passed"])
        self.assertTrue(list((old_job / "attempts").glob("*/job_state.json")))
        self.assertTrue(list(partial.parent.glob("02_diagnosis-demo.interrupted-*")))


if __name__ == "__main__":
    unittest.main()
