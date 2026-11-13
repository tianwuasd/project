import unittest
from pathlib import Path


class RecoveryTests(unittest.TestCase):
    def test_resume_rejects_missing_checkpoint(self):
        from chd_segmentation.training.checkpoint import load_checkpoint
        with self.assertRaises(FileNotFoundError):
            load_checkpoint("Z:/chd-nonexistent-run-20261113", "identity")

    def test_json_scientific_notation_keeps_optimizer_number(self):
        from chd_segmentation.config import load_config
        result = load_config(Path(__file__).parent / "fixtures" / "exponent_config.json")
        self.assertIsInstance(result["training"]["weight_decay"], float)
        self.assertEqual(result["training"]["weight_decay"], 0.00003)


if __name__ == "__main__":
    unittest.main()
