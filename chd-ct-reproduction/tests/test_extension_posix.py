"""Real Linux detached append and runtime locking, without ML dependencies."""

import subprocess
import sys
import time
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform == "linux", "Requires Linux runtime locks")
class ExtensionProcessTests(unittest.TestCase):
    def setUp(self):
        from test_server_posix import DetachedJobTests

        self.fixture = DetachedJobTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def wait_exit(self):
        fixture = self.fixture
        for _ in range(100):
            stat = Path(f"/proc/{fixture.pid}/stat")
            if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}:
                return
            time.sleep(0.02)
        self.fail("Worker did not release its session")

    def complete_original(self):
        fixture = self.fixture
        fixture.start("--task", "environment", "--name", "Original named workflow")
        self.assertEqual(fixture.finished()["status"], "passed")
        self.wait_exit()
        return fixture.directory

    def test_completed_workflow_appends_in_place_and_archives_attempt(self):
        fixture = self.fixture
        job = self.complete_original()
        original_plan = (job / "plan.json").read_bytes()
        old_step = job / "steps/01_environment"
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in old_step.rglob("*") if p.is_file()}
        fixture.start("--resume-job", str(job), "--append-tasks", "diagnosis-demo")
        state = fixture.finished()
        self.wait_exit()
        self.assertEqual(fixture.directory, job)
        self.assertEqual(state["name"], "Original named workflow")
        self.assertEqual(state["status"], "passed")
        self.assertEqual([s["status"] for s in state["steps"]], ["reused", "passed"])
        self.assertEqual(
            before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in old_step.rglob("*") if p.is_file()}
        )
        (attempt,) = (job / "attempts").iterdir()
        self.assertEqual((attempt / "plan.json").read_bytes(), original_plan)
        queried = subprocess.run(
            fixture.command + ["--query", "results", "--job-dir", str(job)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(queried.returncode, 0, queried.stderr)
        self.assertIn("Original named workflow", queried.stdout)
        self.assertIn("02_diagnosis-demo", queried.stdout)

    def test_upload_runtime_lock_prevents_append_before_any_write(self):
        import fcntl

        fixture = self.fixture
        job = self.complete_original()
        before = (job / "plan.json").read_bytes()
        with (fixture.runtime / ".job.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            denied = subprocess.run(
                fixture.command + ["--resume-job", str(job), "--append-tasks", "diagnosis-demo"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("已有任务", denied.stderr)
        self.assertEqual((job / "plan.json").read_bytes(), before)
        self.assertFalse((job / "attempts").exists())


if __name__ == "__main__":
    unittest.main()
