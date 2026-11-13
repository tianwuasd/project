from pathlib import Path
import unittest

from chd_segmentation.artifacts.store import require_result_output
from chd_segmentation.config import load_config


PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT / "tests" / "fixtures" / "exponent_config.json"


class OutputPathTests(unittest.TestCase):
    def config(self):
        config = load_config(FIXTURE)
        config.update({"work_root": str(PROJECT.parent / "course"),
                       "result_root": str(PROJECT.parent / "result"),
                       "run_dir": str(PROJECT.parent / "course" / "run")})
        return config

    def test_results_can_live_outside_the_process_directory(self):
        config = self.config()
        output = Path(config["result_root"]) / "predictions"
        self.assertEqual(require_result_output(config, output), output.resolve())
        with self.assertRaises(ValueError):
            require_result_output(config, Path(config["work_root"]) / "predictions")

    def test_results_protect_real_source_files_and_their_parent_directory(self):
        config = self.config()
        config["result_root"] = str(PROJECT.parent)
        with self.assertRaises(ValueError):
            require_result_output(config, FIXTURE)
        config["result_root"] = str(PROJECT.parent.parent)
        with self.assertRaises(ValueError):
            require_result_output(config, PROJECT.parent)

    def test_results_protect_the_data_and_training_directories(self):
        config = self.config()
        config["result_root"] = str(PROJECT.parent)
        with self.assertRaises(ValueError):
            require_result_output(config, Path(config["run_dir"]) / "prediction")
        data_root = PROJECT.parent / "data"
        with self.assertRaises(ValueError):
            require_result_output(config, data_root / "prediction", [data_root])

    def test_omitting_result_root_preserves_the_existing_output_location(self):
        config = load_config(FIXTURE)
        config["work_root"] = str(PROJECT.parent / "course")
        self.assertIsNone(config["result_root"])
        self.assertEqual(require_result_output(config, Path(config["work_root"]) / "prediction"),
                         Path(config["work_root"]) / "prediction")


if __name__ == "__main__":
    unittest.main()
