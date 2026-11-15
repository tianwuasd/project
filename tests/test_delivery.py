from pathlib import Path
import shutil
from time import perf_counter
import unittest
from uuid import uuid4
import zipfile

from chd_mini.contracts import TOOL_ROOT, confined, file_sha, write_json
from chd_mini.delivery import code_archive, reference_file, delivery_manifest
from chd_mini.runner import inventory


class Delivery(unittest.TestCase):
    def setUp(self):
        self.started = perf_counter()
        self.root = TOOL_ROOT / ".runs" / ("delivery-" + uuid4().hex[:10])
        self.work = self.root / "work"
        self.work.mkdir(parents=True)

    def tearDown(self):
        rows = inventory(self.work)
        write_json(self.root / "logs" / "cleanup.json", {"test": self.id(), "files": len(rows),
                   "released_bytes": sum(r["bytes"] for r in rows), "seconds": perf_counter() - self.started,
                   "cases": 0, "training_updates": 0, "ram_peak": "未测量", "vram_peak": "未测量"})
        confined(self.work, self.root)
        shutil.rmtree(self.work)
        self.assertFalse(self.work.exists())

    def test_committed_code_archive_is_exact_and_reused(self):
        first = code_archive(self.work, "CHD1282", "STUNet", "20261115")
        path = Path(first["path"])
        stamp = path.stat().st_mtime_ns
        self.assertEqual(file_sha(path), first["sha256"])
        with zipfile.ZipFile(path) as archive:
            self.assertIn("run.py", archive.namelist())
            self.assertFalse(any(".local." in name or name.startswith((".runs/", ".git/", ".venv/")) for name in archive.namelist()))
        second = code_archive(self.work, "CHD1282", "STUNet", "20261116")
        self.assertTrue(second["reused"])
        self.assertEqual(second["path"], first["path"])
        self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_reference_records_identity_without_copy(self):
        source = TOOL_ROOT / "configs" / "example.json"
        record = reference_file(source, "/data4/zhangruichen/course/chd-segmentation-assistant3/inputs/config.json")
        self.assertEqual(record["sha256"], file_sha(source))
        self.assertEqual(record["bytes"], source.stat().st_size)
        self.assertEqual(record["uploaded_status"], "unverified")
        self.assertFalse(record["copy_created"])
        self.assertFalse(any(self.work.iterdir()))

    def test_manifest_counts_reused_materials_once(self):
        source = TOOL_ROOT / "configs" / "example.json"
        reference = reference_file(source, "/course/inputs/example.json")
        first = delivery_manifest(self.work, "CHD1282", "STUNet", "20261115", [reference, reference], [], {})
        second = delivery_manifest(self.work, "CHD1282", "STUNet", "20261116", [reference, reference], [], {})
        self.assertEqual(first["referenced_unique_bytes"], source.stat().st_size)
        self.assertEqual(second["new_payload_bytes"], 0)
        self.assertEqual(second["actual_new_bytes"], Path(second["manifest"]).stat().st_size)
        self.assertEqual(second["new_or_changed"], [])
        self.assertTrue(any(r.get("scope") == "public_code_only" for r in second["reuse_in_place"]))


if __name__ == "__main__":
    unittest.main()
