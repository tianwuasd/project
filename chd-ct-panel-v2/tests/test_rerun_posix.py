"""Actual detached rerun, original artifacts and runtime lock on Linux."""

import subprocess
import sys
import unittest


@unittest.skipUnless(sys.platform == "linux", "Requires Linux runtime locks")
class RerunProcessTests(unittest.TestCase):
    def setUp(self):
        from test_extension_posix import ExtensionProcessTests

        self.helper = ExtensionProcessTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)

    def test_completed_step_really_runs_in_detached_child(self):
        fixture = self.helper.fixture
        original = self.helper.complete_original()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original.rglob("*") if p.is_file()}
        fixture.start("--rerun-job", str(original), "--rerun-tasks", "environment")
        self.assertNotEqual(fixture.directory, original)
        state = fixture.finished()
        self.helper.wait_exit()
        self.assertEqual(state["status"], "passed")
        self.assertEqual(state["steps"][0]["status"], "passed")
        self.assertTrue((fixture.directory / "steps/01_environment/started").is_file())
        self.assertEqual(
            before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original.rglob("*") if p.is_file()}
        )

    def test_rerun_respects_runtime_lock_before_creating_new_results(self):
        import fcntl

        fixture = self.helper.fixture
        original = self.helper.complete_original()
        before = set(fixture.results.iterdir())
        with (fixture.runtime / ".job.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(
                fixture.command + ["--rerun-job", str(original)], capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("已有任务", result.stderr)
        self.assertEqual(before, set(fixture.results.iterdir()))
