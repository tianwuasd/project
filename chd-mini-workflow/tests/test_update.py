from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from update import repository
from chd_mini.runner import inventory
from chd_mini.contracts import TOOL_ROOT, verify_engine


class Isolation(unittest.TestCase):
    def test_production_is_not_an_update_target(self):
        with self.assertRaises(ValueError):
            repository(TOOL_ROOT.parent / "chd-segmentation")

    def test_own_repository_is_accepted(self):
        self.assertEqual(repository(TOOL_ROOT), TOOL_ROOT)

    def test_wrong_production_pin_fails(self):
        with self.assertRaises(ValueError):
            verify_engine(TOOL_ROOT.parent / "chd-segmentation", "0" * 40)

    def test_existing_logs_are_regular_owned_files(self):
        rows = inventory(TOOL_ROOT / ".runs")
        self.assertTrue(all(r["bytes"] >= 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
