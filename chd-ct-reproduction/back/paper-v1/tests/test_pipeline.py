from pathlib import Path

import numpy as np
import pytest
import torch

from chd_ct.config import load_config
from chd_ct.inference.pipeline import run_inference
from chd_ct.labels import STAGES
from chd_ct.synthetic import make_demo_dataset
from chd_ct.training.engine import load_checkpoint, train_stage

ROOT = Path(__file__).resolve().parents[1]


def test_eight_stage_training_and_inference(tmp_path):
    manifest = make_demo_dataset(tmp_path / "data")
    config = load_config(ROOT / "configs/smoke.yaml")
    models = tmp_path / "models"
    for stage in STAGES:
        checkpoint = train_stage(config, manifest, stage, models, max_steps=1)
        assert checkpoint.is_file()
        _, record = load_checkpoint(checkpoint, expected_stage=stage)
        assert np.isfinite(record["validation_loss"])
    image = tmp_path / "data/synthetic_03_image.nii.gz"
    with pytest.raises(ValueError, match="Smoke"):
        run_inference(image, [models], tmp_path / "bad")
    pred, initial, volume, metadata = run_inference(image, [models], tmp_path / "out", allow_smoke=True)
    assert pred.shape == volume.data.shape == initial.shape
    assert metadata["profile"] == "smoke"
    assert (tmp_path / "out/segmentation.nii.gz").is_file()
    with pytest.raises(ValueError, match="stage"):
        load_checkpoint(models / "crop64.pt", expected_stage="all64")
    encoder = torch.load(models / "blood2d.pt", weights_only=True)
    encoder["manifest_sha256"] = "wrong-fold"
    torch.save(encoder, models / "blood2d.pt")
    with pytest.raises(ValueError, match="manifest"):
        train_stage(config, manifest, "blood_lstm", models, max_steps=1)
    mixed = torch.load(models / "all64.pt", weights_only=True)
    mixed["manifest_sha256"] = "wrong-fold"
    torch.save(mixed, models / "all64.pt")
    with pytest.raises(ValueError, match="manifest"):
        run_inference(image, [models], tmp_path / "mixed", allow_smoke=True)
    mixed["manifest_sha256"] = record["manifest_sha256"]
    torch.save(mixed, models / "all64.pt")
    record = torch.load(models / "crop64.pt", weights_only=True)
    record["config"]["window"] = [-500, 1500]
    torch.save(record, models / "crop64.pt")
    with pytest.raises(ValueError, match="configurations"):
        run_inference(image, [models], tmp_path / "wrong", allow_smoke=True)


def test_missing_initial_labels_fail(tmp_path):
    manifest = make_demo_dataset(tmp_path / "data")
    config = load_config(ROOT / "configs/smoke.yaml")
    import csv

    from chd_ct.data import read_manifest

    rows = read_manifest(manifest)
    for row in rows:
        row["initial_label"] = ""
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="initial_label"):
        train_stage(config, manifest, "init64", tmp_path / "models", max_steps=1)
