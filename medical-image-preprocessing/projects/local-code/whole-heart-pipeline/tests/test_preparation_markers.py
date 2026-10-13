"""回归：真实 nnU-Net 后缀识别 + 旧服务器缓存无重算迁移。"""
import copy
import importlib.metadata
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_src"))
import server_prepare as prepare
from server_data import digest, read_json, write_json
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetDatasetBlosc2


class PreparationMarkerTests(unittest.TestCase):
    def make_cache(self, root):
        prep = root / "04_data/derived/nnUNet_preprocessed/Dataset703_ct_holdG"
        cache = prep / "Project1Plans_3d_fullres"
        cache.mkdir(parents=True)
        split = {"dataset_id": 703, "train": ["Case1002"], "val": ["Case1001"], "target": ["Case7001"]}
        write_json(root / "04_data/manifests/splits_v1.json", {"folds": {"ct_holdG": split}})
        write_json(root / "04_data/manifests/case_inventory.json", {"cases": []})
        write_json(prep / "Project1Plans.json", {"configurations": {"3d_fullres": {"data_identifier": cache.name}}})
        plan_hash = digest(prep / "Project1Plans.json")
        write_json(prep / "dataset_fingerprint.json", {"fixture": True})
        write_json(prep / "fingerprint_provenance.json", {"fit_ids": split["train"], "sha256": digest(prep / "dataset_fingerprint.json")})
        write_json(root / "09_reports/ct_holdG_preparation.json", {"plan_sha256": plan_hash})
        for case_id in split["train"] + split["val"]:
            # infer_dataset_class 只读文件名；哨兵内容用于确认修复没有改写影像。
            for suffix in [".b2nd", "_seg.b2nd", ".pkl"]:
                (cache / f"{case_id}{suffix}").write_bytes(b"unchanged-data-sentinel")
            write_json(cache / f"{case_id}.complete.json", {"plan_sha256": plan_hash})
        signature = {"split": split, "gpu_memory_gb": 5,
                     "inventory_sha256": digest(root / "04_data/manifests/case_inventory.json"),
                     "implementation_sha256": {name: digest(Path(prepare.__file__).with_name(name)) for name in ["server_prepare.py", "data_pipeline.py", "prepare_nnunet.py"]},
                     "nnunet_version": importlib.metadata.version("nnunetv2")}
        signature["implementation_sha256"]["server_prepare.py"] = prepare.LEGACY_PREPARE_SHA256
        write_json(prep / "server_preparation_signature.json", signature)
        return prep, cache, split, plan_hash, signature

    def test_completed_legacy_cache_migrates_without_export_or_data_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, cache, split, _, old_signature = self.make_cache(root)
            with self.assertRaisesRegex(AssertionError, "more than one file ending"):
                infer_dataset_class(str(cache))
            images = {p: (digest(p), p.stat().st_mtime_ns) for p in cache.iterdir() if p.suffix != ".json"}
            with patch.object(prepare, "ROOT", root), patch.object(prepare, "configure_paths"), \
                 patch.object(prepare, "export_fold", side_effect=AssertionError("must reuse existing cache")):
                prepare.prepare("ct_holdG", 5)
                # 第二次启动无需再次迁移或重新导出。
                prepare.prepare("ct_holdG", 5)
            self.assertIs(infer_dataset_class(str(cache)), nnUNetDatasetBlosc2)
            self.assertEqual(images, {p: (digest(p), p.stat().st_mtime_ns) for p in images})
            self.assertEqual(read_json(prep / "server_preparation_signature.before_marker_fix.json"), old_signature)
            self.assertEqual(len(list((prep / "server_case_completion" / cache.name).glob("*.complete.json"))), 2)

    def test_signature_rejects_unrelated_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, _, _, _, old = self.make_cache(root)
            for field in ["split", "gpu_memory_gb", "inventory_sha256", "implementation_sha256", "nnunet_version"]:
                changed = copy.deepcopy(old)
                if field == "implementation_sha256":
                    changed[field]["server_prepare.py"] = "unknown-code"
                else:
                    changed[field] = "changed"
                write_json(prep / "server_preparation_signature.json", changed)
                with patch.object(prepare, "ROOT", root), patch.object(prepare, "configure_paths"):
                    with self.assertRaisesRegex(ValueError, "代码不同"):
                        prepare.prepare("ct_holdG", 5)

    def test_manually_moved_records_also_reuse_old_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, cache, _, _, _ = self.make_cache(root)
            destination = prep / "server_case_completion" / cache.name
            destination.mkdir(parents=True)
            for source in cache.glob("Case*.complete.json"):
                source.rename(destination / source.name)
            with patch.object(prepare, "ROOT", root), patch.object(prepare, "configure_paths"), \
                 patch.object(prepare, "export_fold", side_effect=AssertionError("must reuse manually repaired cache")):
                prepare.prepare("ct_holdG", 5)
            self.assertIs(infer_dataset_class(str(cache)), nnUNetDatasetBlosc2)

    def test_partial_migration_can_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            prep, cache, split, plan_hash, _ = self.make_cache(Path(temp))
            ids = split["train"] + split["val"]
            destination = prepare.completion_directory(prep, cache, ids[:1], plan_hash)
            prepare.completion_directory(prep, cache, ids, plan_hash)
            self.assertIs(infer_dataset_class(str(cache)), nnUNetDatasetBlosc2)
            self.assertEqual(len(list(destination.glob("*.json"))), 2)

    def test_conflicting_record_preserves_all_legacy_files(self):
        with tempfile.TemporaryDirectory() as temp:
            prep, cache, split, plan_hash, _ = self.make_cache(Path(temp))
            ids = split["train"] + split["val"]
            write_json(prep / "server_case_completion" / cache.name / f"{ids[-1]}.complete.json", {"plan_sha256": "different"})
            with self.assertRaisesRegex(ValueError, "冲突"):
                prepare.completion_directory(prep, cache, ids, plan_hash)
            self.assertEqual(len(list(cache.glob("*.complete.json"))), 2)

    def test_invalid_plan_record_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            prep, cache, split, plan_hash, _ = self.make_cache(Path(temp))
            ids = split["train"] + split["val"]
            write_json(cache / f"{ids[-1]}.complete.json", {"plan_sha256": "different"})
            with self.assertRaisesRegex(ValueError, "计划不符"):
                prepare.completion_directory(prep, cache, ids, plan_hash)
            self.assertEqual(len(list(cache.glob("*.complete.json"))), 2)


if __name__ == "__main__":
    unittest.main()
