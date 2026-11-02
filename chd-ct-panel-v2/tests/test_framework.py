"""Safety and admission invariants, not implementation-shape assertions."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from chd_stunet.acceleration.cpu import choose_workers,GIB
from chd_stunet.config import resolve_path
from chd_stunet.data.discovery import discover_pairs


class FrameworkTests(unittest.TestCase):
    def test_memory_reserve_cannot_be_bypassed_by_requested_workers(self):
        result=choose_workers(32,6*GIB,2*GIB,requested=100)
        self.assertLessEqual(result["max_workers"]*2*GIB,result["memory_budget_bytes"])
        self.assertGreaterEqual(result["reserve_bytes"],4*GIB)
        self.assertEqual(result["max_workers"],1)

    def test_insufficient_memory_is_explicit_no_worker_capacity(self):
        self.assertEqual(choose_workers(32,3*GIB,GIB)["max_workers"],0)

    def test_blank_path_does_not_resolve_to_cwd(self):
        with self.assertRaises(ValueError): resolve_path(" ")
        with self.assertRaises(ValueError): resolve_path(None)

    def test_discovery_reports_orphans_and_missing_labels(self):
        scratch=ROOT/"outputs"/"tests-tmp"
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            directory=Path(temporary).resolve()
            self.assertTrue(directory.is_relative_to(scratch.resolve()))
            for name in ("a_image.nii.gz","a_label.nii.gz","b_image.nii.gz","c_label.nii.gz"):
                (directory/name).write_bytes(b"synthetic file-discovery fixture")
            found=discover_pairs(directory)
            self.assertEqual([p["case_id"] for p in found["pairs"]],["a"])
            self.assertEqual(found["missing_labels"],["b"])
            self.assertEqual(found["orphan_labels"],["c"])


if __name__=="__main__": unittest.main()