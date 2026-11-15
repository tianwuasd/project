import json
import io
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from time import perf_counter
import unittest
from uuid import uuid4

from chd_mini.contracts import TOOL_ROOT, confined, file_sha, load_settings, read_json, write_json
from chd_mini.runner import RunWorkspace, inventory, copy_bounded
from chd_mini.retention import cap_logs
from chd_mini.processes import stop_owned
from chd_mini.contracts import verify_references


class FailureBoundaries(unittest.TestCase):
    def setUp(self):
        self.started = perf_counter()
        self.root = TOOL_ROOT / ".runs" / ("regression-" + uuid4().hex[:10])
        self.work, self.logs = self.root / "work", self.root / "logs"
        self.work.mkdir(parents=True)
        self.logs.mkdir()

    def tearDown(self):
        rows = inventory(self.work)
        write_json(self.logs / "cleanup.json", {"test": self.id(), "deleted_files": len(rows),
                   "released_bytes": sum(r["bytes"] for r in rows), "training_updates": 0,
                   "seconds": perf_counter() - self.started, "ram_peak": "未测量", "vram_peak": "未测量"})
        confined(self.work, self.root)
        shutil.rmtree(self.work)

    def test_changed_frozen_reference_is_rejected(self):
        path = self.work / "frozen.json"
        write_json(path, {"roles": {"case": "train"}})
        sources = {"split": str(path)}
        original = {"split": file_sha(path)}
        verify_references(sources, original)
        write_json(path, {"roles": {"case": "test"}})
        with self.assertRaisesRegex(ValueError, "split"):
            verify_references(sources, original)

    def test_stage_rejects_frozen_split_change_before_launch(self):
        path = self.work / "frozen.json"
        write_json(path, {"roles": {"case": "train"}})
        settings = load_settings(TOOL_ROOT / "project.local.json")
        settings["split"] = str(path)
        with self.assertRaisesRegex(ValueError, "split"):
            with RunWorkspace(settings, self.root.name + "-split") as task:
                write_json(path, {"roles": {"case": "test"}})
                task.stage("invalid-stage")
        summary = read_json(task.logs / "summary.json")
        self.assertEqual(summary["state"], "FAIL")
        self.assertFalse(summary["frozen_inputs_unchanged"])
        self.assertEqual(summary["stages"], [])

    def test_console_stops_writing_at_capacity(self):
        overflow = threading.Event()
        path = self.work / "bounded.log"
        with path.open("wb") as destination:
            receipt = copy_bounded(io.BytesIO(b"x" * 4096), destination, 1024, overflow)
        self.assertEqual(path.stat().st_size, 1024)
        self.assertEqual(receipt["omitted_bytes"], 3072)
        self.assertTrue(overflow.is_set())

    def test_partial_event_keeps_summary_and_raw_failure(self):
        settings = load_settings(TOOL_ROOT / "project.local.json")
        run_id = self.root.name + "-partial"
        with self.assertRaises((RuntimeError, ValueError)):
            with RunWorkspace(settings, run_id) as task:
                events = task.work / "workspace" / "training" / "events.jsonl"
                events.parent.mkdir(parents=True)
                events.write_bytes(b'{"event":"stage_started"}\n{"event":')
                raise RuntimeError("intentional real interrupted-event check")
        self.assertEqual(read_json(task.logs / "summary.json")["state"], "FAIL")
        self.assertIn(b'{"event":', (task.logs / "production-events.raw.log").read_bytes())
        self.assertEqual(read_json(task.logs / "cleanup.json")["state"], "completed")
        self.assertFalse(task.work.exists())

    def test_log_retention_is_bounded_with_receipt(self):
        path = self.work / "logs"
        path.mkdir()
        (path / "overflow.log").write_bytes(b"a" * 4096)
        receipt = cap_logs(path, limit=2048, reserve=512)
        self.assertGreater(receipt["removed_bytes"], 0)
        self.assertLessEqual(sum(r["bytes"] for r in inventory(path)), 2048)
        self.assertTrue((path / "retention.json").is_file())

    def test_already_exited_process_is_a_normal_shutdown(self):
        import psutil
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(0.05)"])
        birth = psutil.Process(child.pid).create_time()
        child.wait()
        stop_owned({child.pid: birth})
        self.assertIsNotNone(child.returncode)

    def test_owned_live_process_is_stopped(self):
        import psutil
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"])
        birth = psutil.Process(child.pid).create_time()
        try:
            stop_owned({child.pid: birth})
        finally:
            child.wait(timeout=5)
        self.assertIsNotNone(child.returncode)


if __name__ == "__main__":
    unittest.main()
