import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from chd_ct.imagechd.common import IGNORE, read_prepared
from chd_ct.imagechd.predict import predict
from chd_ct.imagechd.preprocess import assign_splits, map_labels, normalize, prepare
from chd_ct.imagechd.train import masked_loss, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    labels = np.zeros((16, 20, 24), np.int16)
    for c in range(1, 8):
        labels[c * 2 - 2 : c * 2, 4:12, 6:16] = c
    labels[0:2, 0:2, 0:2] = 14
    for i in range(6):
        image = np.random.default_rng(i).normal(1500, 200, labels.shape).astype(np.float32)
        affine = np.diag([-0.7, 0.8, 1.2, 1.0])
        affine[:3, 3] = [10, 20, 30]
        target = labels.copy()
        if i == 5:
            target[target == 5] = 0
        nib.save(nib.Nifti1Image(image, affine), folder / f"ct_{1000 + i}_image.nii.gz")
        nib.save(nib.Nifti1Image(target, affine), folder / f"ct_{1000 + i}_label.nii.gz")
    return folder


def test_labels_preserve_native_meaning_and_ignore_unknown():
    a = np.arange(16).reshape(2, 2, 4)
    target, missing, extra = map_labels(a)
    assert np.array_equal(target.flat[:8], np.arange(8))
    assert all(x == IGNORE for x in target.flat[8:])
    assert missing == [] and extra == list(range(8, 16))
    with pytest.raises(ValueError):
        map_labels(np.array([[[1.5]]]))


def test_masked_voxels_have_no_loss_or_gradient():
    torch.manual_seed(0)
    target = torch.arange(8).reshape(1, 2, 2, 2)
    target[0, 0, 0, 0] = IGNORE
    logits = torch.randn(1, 8, 2, 2, 2, requires_grad=True)
    first = masked_loss(logits, target)
    altered = logits.detach().clone()
    altered[0, :, 0, 0, 0] = torch.arange(8) * 100.0
    assert torch.allclose(first, masked_loss(altered, target))
    first.backward()
    assert torch.count_nonzero(logits.grad[0, :, 0, 0, 0]) == 0
    with pytest.raises(ValueError):
        masked_loss(logits, torch.full_like(target, IGNORE))


def test_intensity_is_percentile_based_not_assumed_hu():
    image = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    a, params = normalize(image)
    b, _ = normalize(image + 1024)
    assert np.allclose(a, b) and a.min() == 0 and a.max() == 1
    assert params["high"] > params["low"]
    with pytest.raises(ValueError):
        normalize(np.ones((8, 8, 8)))


def test_preprocess_reusable_splits_and_missing_annotation(raw, tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    result = prepare(raw, tmp_path / "prepared", size=16)
    assert len(result["cases"]) == 5
    assert result["excluded"] == [
        {"case_id": "ct_1005", "reason": "missing_foreground_annotation", "missing_ids": [5]}
    ]
    assert {r["split"] for r in result["cases"]} == {"train", "val", "test"}
    assert all(r["extra_ids_ignored"] == [14] for r in result["cases"])
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    ids = [r["case_id"] for r in result["cases"]]
    assert assign_splits(ids, 42) == assign_splits(list(reversed(ids)), 42)
    with pytest.raises(FileExistsError):
        prepare(raw, tmp_path / "prepared", size=16)


def test_explicit_split_rejects_patient_leakage(tmp_path):
    p = tmp_path / "splits.csv"
    p.write_text("case_id,patient_id,split\na,p,train\nb,p,val\nc,q,test\n")
    with pytest.raises(ValueError, match="患者"):
        assign_splits(["a", "b", "c"], 42, p)


def test_independent_train_and_prediction_without_raw_or_targets(raw, tmp_path):
    prepared = tmp_path / "prepared"
    result = prepare(raw, prepared, size=16)
    test_case = next(r for r in result["cases"] if r["split"] == "test")
    reference = nib.load(raw / (test_case["case_id"] + "_image.nii.gz"))
    shape, affine = reference.shape, reference.affine.copy()
    # Both later steps use prepared caches after the raw input has gone away.
    for p in raw.iterdir():
        p.unlink()
    # A test target must never be read for training, nor for prediction.
    with np.load(prepared / test_case["cache"]) as record:
        image = record["image"].copy()
    np.savez_compressed(prepared / test_case["cache"], image=image)
    report = train(prepared, tmp_path / "model", ROOT / "configs/imagechd7-smoke.yaml", "smoke")
    assert report["status"] == "passed" and report["test_cases_used"] == 0
    with pytest.raises(ValueError, match="短测"):
        predict(prepared, tmp_path / "model/best.pt", tmp_path / "forbidden")
    pred = predict(prepared, tmp_path / "model/best.pt", tmp_path / "pred", split="test", allow_smoke=True)
    output = nib.load(tmp_path / "pred" / pred["cases"][0]["prediction"])
    assert output.shape == shape and np.allclose(output.affine, affine)
    assert set(np.unique(np.asanyarray(output.dataobj))).issubset(set(range(8)))


def test_unlabelled_preparation_cannot_train(raw, tmp_path):
    result = prepare(raw, tmp_path / "predict_input", size=16, for_prediction=True)
    assert len(result["cases"]) == 6 and result["excluded"] == []
    with pytest.raises(ValueError, match="无标签"):
        train(tmp_path / "predict_input", tmp_path / "model", ROOT / "configs/imagechd7-smoke.yaml")
    for row in result["cases"]:
        with np.load(tmp_path / "predict_input" / row["cache"]) as record:
            assert record.files == ["image"]


def test_manifest_rejects_unfinished_or_path_escape(raw, tmp_path):
    folder = tmp_path / "prepared"
    prepare(raw, folder, size=16)
    p = folder / "dataset.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cases"][0]["case_id"] = "../escape"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="路径"):
        read_prepared(folder)
    data["status"] = "failed"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="尚未完成"):
        read_prepared(folder)


def test_existing_conda_is_reused_without_downloading(tmp_path, monkeypatch):
    from chd_ct.server import bootstrap

    existing = tmp_path / "old/miniforge/bin/conda"
    existing.parent.mkdir(parents=True)
    existing.write_text("sentinel")
    root, base = tmp_path / "code", tmp_path / "runtime"
    root.mkdir()
    (root / "pyproject.toml").write_text("test")
    calls = []

    def checked(command, env, log):
        calls.append([str(x) for x in command])
        if "create" in command:
            python = base / "envs/chd_py311/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(bootstrap, "checked", checked)
    monkeypatch.setattr(
        bootstrap.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("不应下载新的 Miniforge")
    )
    python = bootstrap.ensure_python(base, root, {}, tmp_path / "log", str(existing))
    assert python.is_file() and calls[0][0] == str(existing)
    assert existing.read_text() == "sentinel"
    assert not (base / "tools/miniforge3").exists()


@pytest.mark.parametrize(
    "task,mode,expected_script",
    [
        ("preprocess", "smoke", "preprocess_imagechd.py"),
        ("train", "smoke", "train_imagechd.py"),
        ("predict", "smoke", "predict_imagechd.py"),
    ],
)
def test_server_dispatch_keeps_steps_independent(tmp_path, monkeypatch, task, mode, expected_script):
    import signal
    import sys
    from types import SimpleNamespace

    from chd_ct.server import imagechd

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    source = tmp_path / "raw"
    source.mkdir()
    prepared = tmp_path / "prepared"
    if task != "preprocess":
        prepared.mkdir()
        (prepared / "dataset.json").write_text("{}")
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    args = SimpleNamespace(
        task=task,
        mode=mode,
        non_interactive=True,
        runtime=str(tmp_path / "runtime"),
        results=str(tmp_path / "results"),
        prepared=str(prepared),
        dataset=str(source),
        checkpoint=str(checkpoint),
        config=None,
        size=96,
        seed=42,
        limit=None,
        for_prediction=False,
        split_file=None,
        device="cpu",
        gpu="auto",
        python=sys.executable,
        case_id=None,
        split="test",
        allow_smoke=True,
    )
    profile = {
        "runtime": args.runtime,
        "results": args.results,
        "prepared": args.prepared,
        "source_updated": "2026-09-26",
        "threads": 2,
    }
    commands = []
    monkeypatch.setattr(imagechd, "query_gpus", lambda: pytest.fail("CPU任务不能选择GPU"))

    def execute(command, env, log):
        commands.append([str(x) for x in command])
        return 0

    monkeypatch.setattr(imagechd, "run_logged", execute)
    assert imagechd.run(args, profile) == 0
    assert len(commands) == 2
    assert commands[0][1].endswith("start.py") and "--mode" in commands[0]
    assert commands[1][1].endswith(expected_script)
    assert not any("chd_ct.server.training" in c for c in commands)
