"""Primary CHD data contracts and a complete six-stage software run."""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from chd_ct.imagechd.preprocess import prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_chd(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    shape = (18, 16, 16)
    target = np.zeros(shape, np.uint8)
    for label in range(1, 8):
        target[2 * label : 2 * label + 2, 3:13, 3:13] = label
    target[0, 0, 0] = 10
    affine = np.diag([-1.5, 2.0, 2.5, 1.0])
    affine[0, 3] = 30
    for i in range(4):
        image = np.random.default_rng(i).normal(100, 20, shape).astype(np.float32)
        nib.save(nib.Nifti1Image(image, affine), folder / f"ct_{i}_image.nii.gz")
        nib.save(nib.Nifti1Image(target, affine), folder / f"ct_{i}_label.nii.gz")
    return folder


def test_native_preparation_and_old_cache_reuse(raw_chd, tmp_path):
    from chd_ct.imagechd.common import read_prepared

    native = tmp_path / "native"
    result = prepare(raw_chd, native, size=0)
    assert result["format"] == "imagechd7-v2"
    assert result["resolution"] == "native"
    with np.load(native / result["cases"][0]["cache"]) as cached:
        assert cached["image"].shape == (18, 16, 16)
        assert 255 in cached["target"]
    assert read_prepared(native)[1]["cases"]
    old = tmp_path / "v1"
    prepare(raw_chd, old, size=16)
    data = json.loads((old / "dataset.json").read_text())
    data["format"] = "imagechd7-v1"
    data.pop("resolution", None)
    (old / "dataset.json").write_text(json.dumps(data))
    assert read_prepared(old)[1]["resolution"] == "resized"


def test_blood_target_and_masked_gradient():
    from chd_ct.imagechd.geometry import blood_target
    from chd_ct.imagechd.losses import masked_loss

    target = np.zeros((8, 8, 8), np.uint8)
    target[1:7, 1:7, 1:7] = 6
    target[4, 4, 4] = 255
    target[0, 0, 0] = 5
    blood = blood_target(target)
    assert blood[0, 0, 0] == 0  # myocardium is not blood
    assert blood[4, 4, 4] == 255
    assert blood[4, 4, 3] == 255  # unknown neighbour cannot define a boundary
    logits = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    y = torch.tensor(blood.astype(np.int64))[None]
    loss = masked_loss(logits, y, [1, 2, 16])
    loss.backward()
    assert torch.isfinite(loss)
    assert (logits.grad[:, :, y[0] == 255] == 0).all()


def test_config_rejects_missing_stage_and_even_sequence(tmp_path):
    import yaml

    from chd_ct.imagechd.config import STAGES, load_config

    config = load_config(ROOT / "configs/chd-smoke.yaml")
    assert tuple(config["stages"]) == STAGES
    assert "init64" not in STAGES
    config["stages"]["blood_lstm"]["sequence"] = 4
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="odd|奇数"):
        load_config(path)


def test_six_stages_prediction_without_raw_or_target(raw_chd, tmp_path):
    from chd_ct.imagechd.config import STAGES
    from chd_ct.imagechd.evaluate import evaluate
    from chd_ct.imagechd.predict import predict
    from chd_ct.imagechd.train import train

    cache, models, output = (tmp_path / name for name in ("cache", "models", "prediction"))
    manifest = prepare(raw_chd, cache, size=16)
    report = train(cache, models, ROOT / "configs/chd-smoke.yaml", mode="smoke", device="cpu")
    assert report["status"] == "passed"
    collection = json.loads((models / "models.json").read_text())
    assert collection["status"] == "complete"
    assert list(collection["stages"]) == list(STAGES)
    assert collection["mode"] == "smoke"
    with pytest.raises(ValueError, match="smoke|短测"):
        predict(cache, models, tmp_path / "refused")
    test_row = next(row for row in manifest["cases"] if row["split"] == "test")
    source = nib.load(raw_chd / (test_row["case_id"] + "_image.nii.gz"))
    expected_shape, expected_affine = source.shape, source.affine.copy()
    test_cache = cache / test_row["cache"]
    with np.load(test_cache) as record:
        image, target = record["image"].copy(), record["target"].copy()
    np.savez_compressed(test_cache, image=image)
    for path in raw_chd.iterdir():
        path.unlink()
    predicted = predict(cache, models, output, split="test", allow_smoke=True)
    assert predicted["status"] == "passed"
    result = nib.load(output / (test_row["case_id"] + "_seg.nii.gz"))
    assert result.shape == expected_shape
    np.testing.assert_allclose(result.affine, expected_affine)
    assert set(np.unique(result.dataobj)).issubset(set(range(8)))
    np.savez_compressed(test_cache, image=image, target=target)
    metrics = evaluate(cache, output, tmp_path / "evaluation", split="test")
    assert metrics["cases_evaluated"] == 1
    assert metrics["space"] == "prepared_grid"
    assert metrics["physical_distance_metrics"] is None
    (models / "crop64.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        predict(cache, models, tmp_path / "incomplete", split="test", allow_smoke=True)


def test_dice_excludes_unknown_and_absent_classes():
    from chd_ct.imagechd.evaluate import dice_metrics

    truth = np.array([0, 1, 255])
    prediction = np.array([0, 1, 7])
    score = dice_metrics(prediction, truth)
    assert score["dice"]["LV"] == 1
    assert score["dice"]["PA"] is None
    assert score["mean_foreground_dice"] == 1


def test_primary_help_and_archive_are_isolated():
    result = subprocess.run(
        [sys.executable, "-B", "-S", str(ROOT / "start.py"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "preprocess" in result.stdout and "predict" in result.stdout
    assert (ROOT / "back/paper-v1/src/chd_ct/inference/pipeline.py").is_file()
    for source in (ROOT / "src/chd_ct").rglob("*.py"):
        assert "from back" not in source.read_text(encoding="utf-8")


def test_formal_collection_is_usable_and_mixing_is_rejected(raw_chd, tmp_path):
    import yaml

    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.config import load_config
    from chd_ct.imagechd.train import train

    cache, models = tmp_path / "cache", tmp_path / "models"
    prepare(raw_chd, cache, size=16)
    config = load_config(ROOT / "configs/chd-smoke.yaml")
    config["profile"] = "chd"
    config["augment"] = True
    path = tmp_path / "tiny-formal.yaml"
    path.write_text(yaml.safe_dump(config))
    report = train(cache, models, path, mode="train")
    assert report["status"] == "passed"
    assert read_collection(models)[1]["mode"] == "train"
    checkpoint = models / "all64.pt"
    checkpoint.write_bytes((models / "crop64.pt").read_bytes())
    with pytest.raises(ValueError, match="校验|混用"):
        read_collection(models)


def test_preflight_keeps_model_size_and_marks_short_weights(raw_chd, tmp_path):
    import yaml

    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.config import load_config
    from chd_ct.imagechd.train import train

    cache, models = tmp_path / "cache", tmp_path / "models"
    prepare(raw_chd, cache, size=16)
    config = load_config(ROOT / "configs/chd-smoke.yaml")
    config["stages"]["all128"].update(size=24, base=3, epochs=5)
    path = tmp_path / "preflight.yaml"
    path.write_text(yaml.safe_dump(config))
    train(cache, models, path, mode="preflight")
    record = read_collection(models, allow_smoke=True)[1]
    assert record["config"]["stages"]["all128"]["size"] == 24
    assert record["config"]["stages"]["all128"]["base"] == 3
    assert record["config"]["stages"]["all128"]["epochs"] == 1
    with pytest.raises(ValueError, match="短测"):
        read_collection(models)


def test_training_rejects_patient_leakage_and_prediction_cache(raw_chd, tmp_path):
    from chd_ct.imagechd.train import training_rows

    data = prepare(raw_chd, tmp_path / "prepared", size=16)
    data["cases"][0]["patient_id"] = "shared"
    data["cases"][1]["patient_id"] = "shared"
    data["cases"][0]["split"] = "train"
    data["cases"][1]["split"] = "test"
    with pytest.raises(ValueError, match="跨"):
        training_rows(data)
    data["for_prediction"] = True
    with pytest.raises(ValueError, match="无标签"):
        training_rows(data)


def test_source_unchanged_and_missing_annotations_are_excluded(raw_chd, tmp_path):
    from chd_ct.imagechd.train import file_hash

    sources = {p: file_hash(p) for p in raw_chd.iterdir()}
    label = nib.load(raw_chd / "ct_3_label.nii.gz")
    values = np.asarray(label.dataobj).copy()
    values[values == 5] = 0
    nib.save(nib.Nifti1Image(values, label.affine), raw_chd / "ct_3_label.nii.gz")
    sources[raw_chd / "ct_3_label.nii.gz"] = file_hash(raw_chd / "ct_3_label.nii.gz")
    output = tmp_path / "prepared"
    data = prepare(raw_chd, output, size=16)
    assert len(data["cases"]) == 3
    assert data["excluded"][0]["missing_ids"] == [5]
    assert all(file_hash(p) == digest for p, digest in sources.items())
    with pytest.raises(FileExistsError):
        prepare(raw_chd, output, size=16)


def test_author_scale_requires_cuda_and_limit_cannot_train(raw_chd, tmp_path):
    from chd_ct.imagechd.train import train

    cache = tmp_path / "cache"
    prepare(raw_chd, cache, size=16, limit=3)
    with pytest.raises(ValueError, match="GPU"):
        train(cache, tmp_path / "large", ROOT / "configs/chd-author-unet.yaml", mode="preflight")
    with pytest.raises(ValueError, match="limit"):
        train(cache, tmp_path / "formal", ROOT / "configs/chd.yaml", mode="train")


def test_parallel_stage_workers_and_independent_test(raw_chd, tmp_path, monkeypatch):
    import os

    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.config import STAGES
    from chd_ct.imagechd.stage_queue import run_queue
    from chd_ct.imagechd.test import run_test
    from chd_ct.imagechd.train import train

    cache, serial, parallel = (tmp_path / n for n in ("prepared", "serial", "parallel"))
    prepare(raw_chd, cache, size=16)
    train(cache, serial, ROOT / "configs/chd-smoke.yaml", mode="smoke", device="cpu")
    record = json.loads((serial / "models.json").read_text())
    parallel.mkdir()
    (parallel / "effective-config.json").write_text(json.dumps(record["config"]))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    def command(stage):
        return [
            sys.executable,
            "-m",
            "chd_ct.imagechd.stage_worker",
            "--prepared",
            cache,
            "--output",
            parallel,
            "--stage",
            stage,
            "--mode",
            "smoke",
        ]

    # Real separate CPU workers exercise scheduling/storage without pretending CUDA was tested.
    record["stages"] = run_queue(STAGES, ["test-slot-a", "test-slot-b"], parallel, command, env)
    (parallel / "models.json").write_text(json.dumps(record))
    assert read_collection(parallel, allow_smoke=True)[1]["status"] == "complete"
    for stage in STAGES:
        left = torch.load(serial / (stage + ".pt"), weights_only=True)["state_dict"]
        right = torch.load(parallel / (stage + ".pt"), weights_only=True)["state_dict"]
        assert all(torch.equal(left[key], right[key]) for key in left)
    # A test invocation must not train or preprocess, even indirectly.
    import importlib

    monkeypatch.setattr(
        importlib.import_module("chd_ct.imagechd.train"), "train", lambda *a, **k: pytest.fail("test trained")
    )
    monkeypatch.setattr(
        importlib.import_module("chd_ct.imagechd.preprocess"),
        "prepare",
        lambda *a, **k: pytest.fail("test preprocessed"),
    )
    result = run_test(cache, parallel, tmp_path / "heldout", allow_smoke=True)
    assert result["status"] == "passed" and result["cases_evaluated"] == 1
    assert json.loads((tmp_path / "heldout/test-report.json").read_text())["split"] == "test"
