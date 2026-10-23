import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaoca_seg import cli, pipeline
from aaoca_seg.images import combine, inspect_image
from aaoca_seg.schema import FIELDS, LABELS, export_inventory, read_manifest


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source scans"
        self.source.mkdir()
        self.output = self.root / "results"
        self.image = self.source / "synthetic scan.nii.gz"
        self.save_image(self.image)
        self.cases = [{"case_id": "demo_001", "image": str(self.image)}]
        self.manifest = self.root / "manifest.csv"
        self.write_manifest()
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def save_image(self, path, units="mm", offset=0):
        data = np.arange(9 * 10 * 11, dtype=np.float32).reshape(9, 10, 11) - 600 + offset
        image = nib.Nifti1Image(data, np.diag([0.7, 0.8, 1.0, 1.0]))
        image.header.set_xyzt_units(units)
        image.set_qform(image.affine, 1)
        image.set_sform(image.affine, 1)
        nib.save(image, str(path))

    def write_manifest(self, rows=None):
        rows = rows or [dict(row, spacing_confirmed="true", hu_confirmed="true") for row in self.cases]
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def fake_execute(self, command, log, stop=None):
        self.calls.append(command)
        source = nib.load(command[command.index("-i") + 1])
        destination = Path(command[command.index("-o") + 1])
        task = command[command.index("--task") + 1]
        destination.mkdir(parents=True, exist_ok=True)
        for label_id, name, _, model in LABELS:
            if model != task:
                continue
            data = np.zeros(source.shape, dtype=np.uint8)
            data[label_id % 8, 3, 4] = 1
            data[4, 4, 4] = 1  # Deliberate overlap across independently inferred structures.
            nib.save(nib.Nifti1Image(data, source.affine, source.header), str(destination / f"{name}.nii.gz"))
        Path(log).write_text("synthetic backend\n", encoding="utf-8")

    def batch(self, cases=None, stop=None):
        return pipeline.run_batch(cases or self.cases, self.output, device="cpu", stop_file=stop,
                                  backend={"test_backend": "synthetic-v1"})

    def test_plan_does_not_write_or_run(self):
        with patch.object(pipeline, "execute") as execute:
            result = pipeline.plan(self.cases, self.output)
        self.assertEqual(result["ready"], 1)
        self.assertFalse(self.output.exists())
        execute.assert_not_called()
        self.assertIn(str(self.image), result["cases"][0]["commands"][0])

    def test_unknown_units_rejected(self):
        self.save_image(self.image, "unknown")
        with self.assertRaisesRegex(ValueError, "不是 mm"):
            inspect_image(self.image)

    def test_qform_sform_conflict_rejected(self):
        image = nib.load(str(self.image))
        shifted = image.affine.copy()
        shifted[0, 3] += 1
        image.set_qform(shifted, 1)
        nib.save(image, str(self.image))
        with self.assertRaisesRegex(ValueError, "qform"):
            inspect_image(self.image)

    def test_nan_and_constant_payload_rejected(self):
        for value in (np.nan, 0):
            image = nib.load(str(self.image))
            nib.save(nib.Nifti1Image(np.full(image.shape, value, dtype=np.float32), image.affine, image.header), str(self.image))
            with self.assertRaises(ValueError):
                inspect_image(self.image, full=True)

    def test_manifest_relative_path_and_selection(self):
        self.write_manifest([{"case_id": "demo_001", "image": "source scans/synthetic scan.nii.gz",
                              "spacing_confirmed": "true", "hu_confirmed": "true"}])
        self.assertEqual(read_manifest(self.manifest)[0]["image"], str(self.image))
        for selected in ([], ["missing"]):
            with self.assertRaises(ValueError):
                read_manifest(self.manifest, selected)

    def test_duplicate_and_unsafe_ids_rejected(self):
        row = dict(self.cases[0], spacing_confirmed="true", hu_confirmed="true")
        for rows in ([row, row], [dict(row, case_id="../escape")], [dict(row, case_id="NUL")]):
            self.write_manifest(rows)
            with self.assertRaises(ValueError):
                read_manifest(self.manifest)

    def test_unconfirmed_metadata_rejected(self):
        self.write_manifest([dict(self.cases[0], spacing_confirmed="false", hu_confirmed="true")])
        with self.assertRaises(ValueError):
            read_manifest(self.manifest)

    def test_inventory_filters_aaoca_without_exporting_identity(self):
        record = {"type": "AAOCA", "destination": str(self.image), "copy_status": "copied_verified",
                  "intensity_units_from_conversion": "HU", "units": ["mm", "unknown"],
                  "name": "SYNTHETIC_PERSON", "registered_patient_id": "SYNTHETIC_ID"}
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps([record, dict(record, type="TAPVC")]), encoding="utf-8")
        destination = self.root / "export.csv"
        self.assertEqual(export_inventory(inventory, destination), 1)
        text = destination.read_text(encoding="utf-8-sig")
        self.assertNotIn("SYNTHETIC_PERSON", text)
        self.assertNotIn("SYNTHETIC_ID", text)
        self.assertEqual(len(read_manifest(destination)), 1)
        with self.assertRaises(FileExistsError):
            export_inventory(inventory, destination)

    def test_output_cannot_overlap_source_or_code(self):
        for path in (self.source, self.source / "results", self.root, pipeline.CODE_ROOT / "data"):
            with self.assertRaises(ValueError):
                pipeline.validate_output(path, self.cases)

    def test_command_contract(self):
        cmds = pipeline.commands(self.image, self.output)
        self.assertEqual(len(cmds), 3)
        self.assertEqual(cmds[0][1][-3:], ["pulmonary_vein", "superior_vena_cava", "inferior_vena_cava"])
        for _, command in cmds:
            self.assertNotIn("--fast", command)
            self.assertNotIn("--remove_small_blobs", command)
            self.assertNotIn("--save_lowres", command)
        for device in ("gpu:0;bad", "-1", "gpu:-1"):
            with self.assertRaises(ValueError):
                pipeline.commands(self.image, self.output, device)

    def test_combination_priority_geometry_and_binary_preservation(self):
        with patch.object(pipeline, "execute", side_effect=self.fake_execute):
            result = self.batch()
        self.assertEqual(result["status"], "completed")
        record = json.loads((self.output / "cases/demo_001/latest.json").read_text())
        attempt = self.output / "cases/demo_001" / record["attempt"]
        combined = nib.load(str(attempt / "aaoca_11_labels.nii.gz"))
        self.assertEqual(int(combined.dataobj[4, 4, 4]), 11)
        self.assertEqual(combined.get_data_dtype(), np.dtype("uint8"))
        self.assertTrue(np.allclose(combined.affine, nib.load(str(self.image)).affine))
        self.assertEqual(int(nib.load(str(attempt / "heartchambers_highres/myocardium.nii.gz")).dataobj[4, 4, 4]), 1)
        qc = json.loads((attempt / "qc.json").read_text(encoding="utf-8"))
        self.assertTrue(qc["overlaps"])
        self.assertTrue(qc["review_required"])

    def test_reuse_requires_full_artifact_hashes(self):
        with patch.object(pipeline, "execute", side_effect=self.fake_execute):
            self.batch()
            self.assertEqual(self.batch()["reused"], 1)
            self.assertEqual(len(self.calls), 3)
            record = json.loads((self.output / "cases/demo_001/latest.json").read_text())
            attempt = self.output / "cases/demo_001" / record["attempt"]
            (attempt / "aaoca_11_labels.nii.gz").write_bytes(b"damaged")
            self.assertEqual(self.batch()["completed"], 1)
            self.assertEqual(len(self.calls), 6)
            self.assertTrue(attempt.exists())

    def test_changed_source_invalidates_reuse(self):
        with patch.object(pipeline, "execute", side_effect=self.fake_execute):
            self.batch()
            self.save_image(self.image, offset=10)
            self.assertEqual(self.batch()["completed"], 1)
        self.assertEqual(len(self.calls), 6)

    def test_geometry_mismatch_fails_case(self):
        def shifted(command, log, stop):
            self.fake_execute(command, log, stop)
            output = Path(command[command.index("-o") + 1])
            for path in output.glob("*.nii.gz"):
                original = nib.load(str(path))
                affine = original.affine.copy()
                affine[0, 3] += 0.7
                nib.save(nib.Nifti1Image(np.asanyarray(original.dataobj), affine, original.header), str(path))
        with patch.object(pipeline, "execute", side_effect=shifted):
            result = self.batch()
        self.assertEqual(result["failed"], 1)
        self.assertIn("网格不一致", result["cases"][0]["error"])
        self.assertFalse((self.output / "cases/demo_001/latest.json").exists())

    def test_failure_does_not_block_next_case(self):
        bad = {"case_id": "missing", "image": str(self.source / "missing.nii.gz")}
        with patch.object(pipeline, "execute", side_effect=self.fake_execute):
            result = self.batch([bad, *self.cases])
        self.assertEqual((result["failed"], result["completed"]), (1, 1))
        self.assertEqual(result["status"], "completed_with_errors")

    def test_missing_artifact_declaration_invalidates_reuse(self):
        with patch.object(pipeline, "execute", side_effect=self.fake_execute):
            self.batch()
            latest = self.output / "cases/demo_001/latest.json"
            record = json.loads(latest.read_text())
            record["artifacts"].pop("qc.json")
            latest.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(self.batch()["completed"], 1)

    def test_nonbinary_mask_is_rejected(self):
        with patch.object(pipeline, "execute", side_effect=self.fake_execute):
            self.batch()
        latest = json.loads((self.output / "cases/demo_001/latest.json").read_text())
        attempt = self.output / "cases/demo_001" / latest["attempt"]
        path = attempt / "coronary_arteries/coronary_arteries.nii.gz"
        image = nib.load(str(path))
        nib.save(nib.Nifti1Image(np.full(image.shape, 2, dtype=np.uint8), image.affine, image.header), str(path))
        with self.assertRaisesRegex(ValueError, "二值"):
            combine(self.image, attempt)

    def test_input_change_during_inference_cannot_complete(self):
        def changed(command, log, stop):
            self.fake_execute(command, log, stop)
            if command[command.index("--task") + 1] == "coronary_arteries":
                self.save_image(self.image, offset=20)
        with patch.object(pipeline, "execute", side_effect=changed):
            result = self.batch()
        self.assertEqual(result["failed"], 1)
        self.assertFalse((self.output / "cases/demo_001/latest.json").exists())

    def test_failed_model_keeps_failed_attempt_and_log(self):
        def failed(command, log, stop):
            Path(log).write_text("synthetic model error", encoding="utf-8")
            raise RuntimeError("synthetic failure")
        with patch.object(pipeline, "execute", side_effect=failed):
            result = self.batch()
        self.assertEqual(result["failed"], 1)
        record_path = next((self.output / "cases/demo_001").glob("attempt-*/record.json"))
        self.assertEqual(json.loads(record_path.read_text())["status"], "failed")
        self.assertIn("model_log", result)

    def test_lock_and_stop(self):
        self.output.mkdir()
        with pipeline.run_lock(self.output):
            with self.assertRaises(ValueError):
                self.batch()
        self.assertFalse((self.output / ".run.lock").exists())
        stop = self.root / "request.stop"
        stop.touch()
        with patch.object(pipeline, "execute") as execute:
            result = self.batch(stop=stop)
        execute.assert_not_called()
        self.assertEqual(result["status"], "stopped")
        self.assertFalse((self.output / ".run.lock").exists())

    def test_interruption_does_not_mark_completed(self):
        with patch.object(pipeline, "execute", side_effect=KeyboardInterrupt):
            result = self.batch()
        self.assertEqual(result["status"], "stopped")
        self.assertFalse((self.output / "cases/demo_001/latest.json").exists())
        self.assertFalse((self.output / ".run.lock").exists())

    def test_cli_plan(self):
        with patch("builtins.print"):
            result = cli.main(["plan", "--manifest", str(self.manifest), "--output-root", str(self.output)])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
