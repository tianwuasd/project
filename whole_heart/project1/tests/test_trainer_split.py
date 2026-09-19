"""检查官方训练器默认回退不会破坏项目冻结的中心划分。"""
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_src"))
from prepare_nnunet import configure_paths
configure_paths()
from data_pipeline import save_json
from project_trainers import Project1B0


class TrainerSplitTests(unittest.TestCase):
    def test_missing_and_changed_splits_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"train": ["a", "b"], "val": ["c"], "target": ["d"]}
            save_json(root / "04_data/manifests/splits_v1.json", {"folds": {"ct_holdG": expected}})
            save_json(root / "fingerprint_provenance.json", {"fit_ids": ["a", "b"]})
            fake = SimpleNamespace(fold=0, plans_manager=SimpleNamespace(dataset_name="Dataset703_ct_holdG"), preprocessed_dataset_folder_base=str(root), print_to_log_file=lambda _: None)
            with patch("project_trainers.ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    Project1B0.do_split(fake)
                save_json(root / "splits_final.json", [{"train": ["a", "b"], "val": ["d"]}])
                with self.assertRaises(ValueError):
                    Project1B0.do_split(fake)
                save_json(root / "splits_final.json", [{"train": ["a", "b"], "val": ["c"]}])
                self.assertEqual(Project1B0.do_split(fake), (["a", "b"], ["c"]))
                fake.fold = 1
                with self.assertRaises(ValueError):
                    Project1B0.do_split(fake)


if __name__ == "__main__":
    unittest.main()
