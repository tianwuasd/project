from pathlib import Path
import os
import tempfile
import unittest

from filelock import Timeout

from chd_segmentation.artifacts.store import file_digest, operation_lock


FIXTURE = Path(__file__).parent / "fixtures" / "exponent_config.json"


class IdentityGuardTests(unittest.TestCase):
    def test_current_plan_must_match_prepared_source(self):
        from chd_segmentation.preprocessing.prepare import validate_prepared_config
        base = {"dataset": "chd1282", "scope": "technical", "fold": 0, "technical": {"roi_xyz": None}}
        record = {**base, "source_identity": {"manifest": file_digest(FIXTURE), "plan": "0" * 64},
                  "manifest": {"cases": []}}
        config = {**base, "manifest": str(FIXTURE), "plan": str(FIXTURE)}
        with self.assertRaises(ValueError):
            validate_prepared_config(record, config)

    def test_missing_source_digest_is_rejected(self):
        from chd_segmentation.preprocessing.transforms import CaseTransform
        row = {"image": str(FIXTURE)}
        with self.assertRaises(ValueError):
            CaseTransform({}, 0)._source(row, "image")

    def test_persisted_source_stamp_is_checked_on_new_reader(self):
        from chd_segmentation.preprocessing.transforms import CaseTransform
        row = {"image": str(FIXTURE), "sha256": {"image": file_digest(FIXTURE)},
               "source_stamp": {"image": [0, 0]}}
        with self.assertRaises(ValueError):
            CaseTransform({}, 0)._source(row, "image")

    def test_prediction_snapshot_refuses_active_training_lock(self):
        from chd_segmentation.training.checkpoint import prediction_snapshot
        # 此处只生成操作系统锁，未创建模型或病例副本。
        root = Path(os.environ["CHD_TEST_ROOT"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="chd-lock-", dir=root) as directory:
            with operation_lock(directory):
                with self.assertRaises(Timeout):
                    with prediction_snapshot(directory):
                        pass

    def test_evaluation_cannot_write_over_a_source_file(self):
        from chd_segmentation.prediction.metrics import evaluation_output
        prepared = {"manifest": {"data_root": str(FIXTURE.parent)}}
        config = {"work_root": str(FIXTURE.parent.parent), "run_dir": str(FIXTURE.parent.parent / "run"),
                  "manifest": str(FIXTURE), "plan": str(FIXTURE), "pretrained": str(FIXTURE)}
        before = file_digest(FIXTURE)
        with self.assertRaises(ValueError):
            evaluation_output(config, prepared, FIXTURE.parent / "predictions", FIXTURE)
        self.assertEqual(file_digest(FIXTURE), before)

    def test_evaluation_identity_includes_label_dictionary(self):
        from chd_segmentation.prediction.metrics import case_data_identity
        row = {"image": str(FIXTURE), "sha256": {"image": file_digest(FIXTURE)},
               "spacing_xyz": [1, 1, 1], "geometry_policy": "shared_index_excel", "unknown_classes": []}
        prepared = {"dataset": "chd1282", "dataset_json": {"labels": {"background": 0, "LV": 1}},
                    "plan": {}, "technical": {"roi_xyz": None}}
        first = case_data_identity(prepared, row)
        prepared["dataset_json"]["labels"]["LV"] = 2
        self.assertNotEqual(first, case_data_identity(prepared, row))


if __name__ == "__main__":
    unittest.main()
