import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chd_ct.data import load_volume, read_manifest
from chd_ct.quickstart.datasets import find_manifests, prepare_sample, validate_dataset
from chd_ct.synthetic import make_demo_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_directory_selection_is_unambiguous(tmp_path):
    data = tmp_path / "中文 data"
    manifest = make_demo_dataset(data)
    assert find_manifests(data) == [manifest.resolve()]
    assert find_manifests(manifest) == [manifest.resolve()]
    (data / "fold1.csv").write_text(manifest.read_text(), encoding="utf-8")
    assert len(find_manifests(data)) == 2
    with pytest.raises(ValueError, match="manifest"):
        find_manifests(tmp_path / "missing")


def test_sample_keeps_originals_and_patient_splits(tmp_path):
    manifest = make_demo_dataset(tmp_path / "source")
    before = manifest.read_bytes()
    info = validate_dataset(manifest)
    assert info["initial_complete"]
    result = prepare_sample(manifest, tmp_path / "sample", edge=16)
    rows = read_manifest(result["manifest"])
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert manifest.read_bytes() == before
    assert max(load_volume(rows[0]["image"]).data.shape) <= 16
    assert load_volume(rows[0]["image"]).spacing == pytest.approx((1.2, 1.5, 2.25))
    assert result["inference_split"] == "test"


def test_missing_initial_labels_report_partial_capability(tmp_path):
    manifest = make_demo_dataset(tmp_path / "source")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    for row in rows:
        row["initial_label"] = ""
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    assert validate_dataset(manifest)["initial_complete"] is False


def test_selected_dataset_smoke_subprocess(tmp_path):
    manifest = make_demo_dataset(tmp_path / "用户 data")
    output = tmp_path / "check report"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "start.py"),
            "--dataset",
            str(manifest.parent),
            "--non-interactive",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["steps"]["environment"]["status"] == "passed"
    assert report["steps"]["synthetic_smoke"]["status"] == "passed"
    assert report["steps"]["dataset_smoke"]["status"] == "passed"
    assert (output / "selected/prediction/segmentation.nii.gz").is_file()
    assert (output / "report.txt").is_file()
    # Never overwrite an earlier experiment, including its report.
    again = subprocess.run(run.args, cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert again.returncode != 0
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["status"] == "passed"


def test_failed_environment_stops_with_report(tmp_path, monkeypatch):
    from chd_ct.quickstart import wizard

    monkeypatch.setattr(wizard, "check_environment", lambda *_: {"ok": False, "errors": ["Missing torch"]})
    output = tmp_path / "failure"
    code = wizard.main(["--demo", "--non-interactive", "--output", str(output)])
    assert code == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["steps"]["environment"]["status"] == "failed"
    assert report["steps"]["synthetic_smoke"]["status"] == "not_run"
    assert not (output / "synthetic").exists()


def test_check_only_has_no_training_outputs(tmp_path):
    output = tmp_path / "check_only"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "start.py"),
            "--demo",
            "--mode",
            "check",
            "--non-interactive",
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=45,
    )
    assert run.returncode == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "checked"
    assert report["steps"]["synthetic_smoke"]["status"] == "not_run"
    assert not (output / "synthetic").exists()


def test_partial_data_smoke_has_distinct_exit_code(tmp_path):
    manifest = make_demo_dataset(tmp_path / "source")
    with manifest.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["initial_label"] = ""
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "partial"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "start.py"),
            "--dataset",
            str(manifest),
            "--non-interactive",
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=120,
    )
    assert run.returncode == 2, run.stdout + run.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["steps"]["dataset_smoke"]["inference"] == "all128_baseline_only"
    assert (output / "selected/prediction/baseline.nii.gz").is_file()
    assert not (output / "selected/prediction/diagnosis.json").exists()


def test_multiple_manifests_noninteractive_does_not_guess(tmp_path):
    from chd_ct.quickstart.wizard import main

    manifest = make_demo_dataset(tmp_path / "source")
    (manifest.parent / "fold1.csv").write_bytes(manifest.read_bytes())
    output = tmp_path / "ambiguous"
    assert main(["--dataset", str(manifest.parent), "--non-interactive", "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert "多个" in report["message"]
    assert report["steps"]["synthetic_smoke"]["status"] == "not_run"


def test_no_validation_split_is_rejected(tmp_path):
    manifest = make_demo_dataset(tmp_path / "source")
    manifest.write_text(manifest.read_text().replace(",val,", ",test,"), encoding="utf-8")
    with pytest.raises(ValueError, match="train 和 val"):
        validate_dataset(manifest)
