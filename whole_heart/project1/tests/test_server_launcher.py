"""服务器入口的安全与可恢复性检查，不启动正式训练、不需要读取医学影像。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_src"))
from server_data import resolve_cases, bind_workspace, read_json, write_json, digest
from server_train import select_checkpoint


class ServerDataTests(unittest.TestCase):
    def make_fixture(self, root):
        raw, code, work = root / "原始 数据", root / "code", root / "results"
        raw.mkdir()
        cases = []
        for i, status in [(1, "eligible"), (2, "quarantined")]:
            case = {"case_id": f"Case{i:04d}", "status": status, "group": "A ct_train", "modality": "CT"}
            for kind in ["image", "label"]:
                p = raw / f"{case['case_id']}_{kind}.nii.gz"
                p.write_bytes(f"synthetic-{i}-{kind}".encode())
                case[kind] = f"D:/old/{p.name}"
                case[f"{kind}_sha256"] = digest(p)
            cases.append(case)
        manifest = code / "04_data/manifests"
        write_json(manifest / "case_inventory.json", {"cases": cases})
        write_json(manifest / "splits_v1.json", {"inventory_sha256": "old", "folds": {"example": {"train": ["Case0001"], "val": [], "target": []}}})
        return raw, code, work, cases

    def test_relocation_preserves_splits_quarantine_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, code, work, cases = self.make_fixture(Path(temp))
            before = {p.name: p.read_bytes() for p in raw.iterdir()}
            bind_workspace(code, work, raw, 5)
            mapped = read_json(work / "04_data/manifests/case_inventory.json")
            self.assertTrue(Path(mapped["cases"][0]["image"]).is_absolute())
            self.assertEqual(read_json(work / "04_data/manifests/quarantine.json")[0]["case_id"], "Case0002")
            self.assertEqual(read_json(work / "04_data/manifests/splits_v1.json")["folds"], read_json(code / "04_data/manifests/splits_v1.json")["folds"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in raw.iterdir()})
            bind_workspace(code, work, raw, 5)  # 第二次可重复运行。
            with self.assertRaises(ValueError):
                bind_workspace(code, work, raw, 6)

    def test_missing_duplicate_and_changed_files_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, code, work, cases = self.make_fixture(Path(temp))
            image = raw / "Case0001_image.nii.gz"
            original = image.read_bytes()
            image.write_bytes(b"different")
            with self.assertRaises(ValueError):
                resolve_cases(raw, cases)
            image.write_bytes(original)
            (raw / "duplicate").mkdir()
            (raw / "duplicate" / image.name).write_bytes(original)
            with self.assertRaises(ValueError):
                resolve_cases(raw, cases)
            (raw / "duplicate" / image.name).unlink()
            image.unlink()
            with self.assertRaises(ValueError):
                resolve_cases(raw, cases)

    def test_workspace_cannot_overwrite_source_or_unknown_files(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, code, work, cases = self.make_fixture(Path(temp))
            with self.assertRaises(ValueError):
                bind_workspace(code, raw / "results", raw, 5)
            with self.assertRaises(ValueError):
                bind_workspace(code, code, raw, 5)
            work.mkdir()
            (work / "unrelated.txt").write_text("keep")
            with self.assertRaises(ValueError):
                bind_workspace(code, work, raw, 5)
            self.assertEqual((work / "unrelated.txt").read_text(), "keep")

    def test_checkpoint_priority_never_uses_smoke_or_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ["checkpoint_best.pth", "checkpoint_smoke.pth", "checkpoint_latest.pth.tmp"]:
                (root / name).touch()
            with self.assertRaises(FileNotFoundError):
                select_checkpoint(root)
            (root / "checkpoint_latest.pth").touch()
            self.assertEqual(select_checkpoint(root).name, "checkpoint_latest.pth")
            (root / "checkpoint_final.pth").touch()
            self.assertEqual(select_checkpoint(root).name, "checkpoint_final.pth")


class TrainerGuardTests(unittest.TestCase):
    def test_finite_loss_guard(self):
        from project_trainers import Project1B0
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        with patch.object(nnUNetTrainer, "train_step", return_value={"loss": np.nan}):
            fake = object.__new__(Project1B0)
            with self.assertRaises(FloatingPointError):
                fake.train_step({})

    def test_checkpoint_is_replaced_only_after_successful_write(self):
        from project_trainers import Project1B0
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "latest.pth"
            filename.write_bytes(b"previous")
            fake = object.__new__(Project1B0)
            fake.local_rank, fake.disable_checkpointing = 0, False
            def broken(path):
                Path(path).write_bytes(b"partial")
                raise OSError("simulated interrupted write")
            with patch.object(nnUNetTrainer, "save_checkpoint", side_effect=broken):
                with self.assertRaises(OSError):
                    fake.save_checkpoint(filename)
            self.assertEqual(filename.read_bytes(), b"previous")
            with patch.object(nnUNetTrainer, "save_checkpoint", side_effect=lambda p: Path(p).write_bytes(b"complete")):
                fake.save_checkpoint(filename)
            self.assertEqual(filename.read_bytes(), b"complete")


class ServerExecutionTests(unittest.TestCase):
    def test_export_resume_repairs_partial_copy_without_exporting_target(self):
        import nibabel as nib
        from data_pipeline import export_fold
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, label = root / "image.nii.gz", root / "label.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), image)
            nib.save(nib.Nifti1Image(np.full((4, 4, 4), 500, dtype=np.int16), np.eye(4)), label)
            case = {"case_id": "Case0001", "image": str(image), "label": str(label), "image_sha256": digest(image)}
            split = {"dataset_id": 703, "modality": "CT", "train": ["Case0001"], "val": [], "target": ["Case0002"]}
            out = export_fold(root, "ct_holdG", split, [case])
            copied = out / "imagesTr/Case0001_0000.nii.gz"
            copied.write_bytes(b"interrupted-copy")
            export_fold(root, "ct_holdG", split, [case], resume=True)
            self.assertEqual(digest(copied), digest(image))
            self.assertEqual(len(list((out / "imagesTr").iterdir())), 1)
            self.assertTrue((np.asarray(nib.load(out / "labelsTr/Case0001.nii.gz").dataobj) == 3).all())
            with self.assertRaises(ValueError):
                export_fold(root, "ct_holdG", dict(split, train=[]), [case], resume=True)

    def test_formal_lifecycle_resume_and_changed_environment(self):
        # 模拟训练器生命周期，验证调度与恢复；这里没有运行神经网络长训练。
        import server_train
        import project_trainers
        class FakeTrainer:
            fail_once = True
            calls = 0
            def __init__(self, *args):
                self.network = object()
                self.num_epochs, self.current_epoch = 250, 0
            def initialize(self):
                pass
            def load_checkpoint(self, filename):
                self.current_epoch = 1
            def run_training(self):
                type(self).calls += 1
                out = Path(self.output_folder)
                if type(self).fail_once:
                    type(self).fail_once = False
                    (out / "checkpoint_latest.pth").write_bytes(b"mock-checkpoint")
                    raise KeyboardInterrupt()
                self.current_epoch = self.num_epochs
                (out / "checkpoint_final.pth").write_bytes(b"mock-final")
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            prep = root / "04_data/derived/nnUNet_preprocessed/Dataset703_ct_holdG"
            write_json(root / "04_data/manifests/splits_v1.json", {"folds": {"ct_holdG": {"dataset_id": 703}}})
            write_json(prep / "Project1Plans.json", {})
            write_json(prep / "dataset.json", {})
            (root / "06_configs").mkdir()
            for name in ["pip_freeze_training.txt", "conda_explicit_training.txt"]:
                (root / "06_configs" / name).write_text("fixed")
            stack.enter_context(patch.object(server_train, "ROOT", root))
            stack.enter_context(patch.object(server_train, "configure_paths"))
            stack.enter_context(patch.object(project_trainers, "Project1B0", FakeTrainer))
            stack.enter_context(patch("smoke_train.state_hash", return_value="same-initialization"))
            stack.enter_context(patch("nnunetv2.training.logging.nnunet_logger.MetaLogger"))
            stack.enter_context(patch.object(sys, "argv", ["server_train.py", "ct_holdG", "B0"]))
            with self.assertRaises(KeyboardInterrupt):
                server_train.main()
            record = root / "07_experiments/formal/ct_holdG_B0_seed0/run.json"
            self.assertEqual(read_json(record)["status"], "interrupted")
            with self.assertRaises(FileExistsError):
                server_train.main()
            sys.argv.append("--resume")
            server_train.main()
            result = read_json(record)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_epochs"], 250)
            self.assertTrue(result["history"][-1]["resumed_from"].endswith("checkpoint_latest.pth"))
            server_train.main()  # 已完成的工作不会重复执行。
            self.assertEqual(FakeTrainer.calls, 2)
            (root / "06_configs/pip_freeze_training.txt").write_text("changed")
            with self.assertRaises(ValueError):
                server_train.main()


if __name__ == "__main__":
    unittest.main()
