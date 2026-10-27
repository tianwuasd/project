"""Coordinate alignment, boundary constraints and scientific compatibility."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import test_chd_main

from chd_ct.imagechd.fusion import refine_with_blood
from chd_ct.imagechd.geometry import blood_target, resize


@pytest.fixture
def raw_chd(tmp_path):
    return test_chd_main.raw_chd.__wrapped__(tmp_path)


@pytest.mark.parametrize("shape", [(6, 7, 14), (17, 8, 5), (13, 11, 9)])
def test_labels_use_image_pixel_centers_without_new_ids(shape):
    source = (np.indices((13, 11, 9)).sum(0) % 8).astype(np.uint8)
    source[2:5, 1:4, 3:7] = 255
    # Independent categorical sample at each align_corners=False pixel center.
    indices = [
        np.minimum(np.floor((np.arange(n) + 0.5) * old / n).astype(int), old - 1)
        for old, n in zip(source.shape, shape)
    ]
    expected = source[np.ix_(*indices)]
    actual = resize(source, shape, labels=True)
    np.testing.assert_array_equal(actual, expected)
    assert set(np.unique(actual)) <= set(np.unique(source))


def test_boundary_exclusion_is_preserved_after_paper_review():
    target = np.zeros((36, 10, 10), np.uint8)
    for label in range(1, 8):
        target[4 * label : 4 * label + 3, 2:8, 2:8] = label
    blood = blood_target(target)
    expected = target.copy()
    expected[(blood == 2) & (target != 5)] = 0
    np.testing.assert_array_equal(refine_with_blood(target, blood), expected)


def test_boundary_labels_are_excluded_and_growth_cannot_cross_boundary():
    seg = np.zeros((10, 3, 3), np.uint8)
    seg[1, 1, 1] = seg[4, 1, 1] = 1
    seg[0, 0, 0], seg[9, 0, 0] = 6, 5
    blood = np.zeros_like(seg)
    blood[1:9, 1, 1] = 1
    blood[4:6, 1, 1] = 2
    result = refine_with_blood(seg, blood)
    assert np.all(result[1:4, 1, 1] == 1)
    assert np.all(result[4:9, 1, 1] == 0)
    assert result[0, 0, 0] == 0 and result[9, 0, 0] == 5


def test_old_native_and_excel_caches_remain_valid_but_old_resized_labels_do_not():
    from chd_ct.imagechd.methods import LABEL_RESIZE, require_aligned_cache

    native = {
        "resolution": "native",
        "size": 0,
        "cases": [{"canonical_shape": [12, 14, 16], "cache_shape": [12, 14, 16]}],
    }
    before = copy.deepcopy(native)
    require_aligned_cache(native)
    assert native == before
    resampled = copy.deepcopy(native)
    row = resampled["cases"][0]
    row["resampling"] = {"resampled_shape": [24, 14, 16]}
    row["cache_shape"] = [24, 14, 16]
    resampled["resolution"] = "resampled"
    require_aligned_cache(resampled)
    resized = copy.deepcopy(resampled)
    resized.update(resolution="resized", size=16)
    resized["cases"][0]["cache_shape"] = [16] * 3
    with pytest.raises(ValueError, match="旧.*缩放|重新预处理"):
        require_aligned_cache(resized)
    resized["label_resize"] = LABEL_RESIZE
    resized["cases"][0]["label_resize"] = LABEL_RESIZE
    require_aligned_cache(resized)
    resized["cases"][0].pop("label_resize")
    with pytest.raises(ValueError, match="重新预处理"):
        require_aligned_cache(resized)
    # An image-only input contains no old, misaligned supervised labels.
    resized["for_prediction"] = True
    require_aligned_cache(resized)


@pytest.mark.parametrize("old_format", ["imagechd7-chd-paper-v3", "imagechd7-threefold-v1"])
def test_old_models_fail_before_weights_are_read(tmp_path, old_format):
    from chd_ct.imagechd.checkpoints import read_collection

    path = tmp_path / "models.json"
    payload = json.dumps({"format": old_format}).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="重新训练|重训"):
        read_collection(tmp_path)
    assert path.read_bytes() == payload


def test_preprocess_resize_contract_records_sampling(raw_chd, tmp_path):
    from chd_ct.imagechd.common import read_prepared
    from chd_ct.imagechd.methods import LABEL_RESIZE, require_aligned_cache
    from chd_ct.imagechd.preprocess import prepare

    output = tmp_path / "cache"
    data = prepare(raw_chd, output, size=16)
    assert data["label_resize"] == LABEL_RESIZE
    assert all(row["label_resize"] == LABEL_RESIZE for row in data["cases"])
    saved = (output / "dataset.json").read_bytes()
    prepare(raw_chd, output, size=16, resume=True)
    assert (output / "dataset.json").read_bytes() == saved
    require_aligned_cache(read_prepared(output)[1])
    contract_path = output / "preprocess-contract.json"
    contract = json.loads(contract_path.read_text())
    contract["options"].pop("label_resize")
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="label_resize"):
        prepare(raw_chd, output, size=16, resume=True)
    assert (output / "dataset.json").read_bytes() == saved


def test_training_tensor_cache_uses_a_new_identity(raw_chd, tmp_path, monkeypatch):
    from chd_ct.imagechd import cpu_cache
    from chd_ct.imagechd.config import load_config
    from chd_ct.imagechd.preprocess import prepare
    from chd_ct.recovery import file_hash

    data = prepare(raw_chd, tmp_path / "prepared")
    config = load_config(Path(__file__).parents[1] / "configs/chd-smoke.yaml")
    row = data["cases"][0]
    current = cpu_cache.FORMAT
    with monkeypatch.context() as patch:
        patch.setattr(cpu_cache, "FORMAT", "imagechd-training-inputs-v1")
        old = cpu_cache.cached_input(
            tmp_path / "inputs", tmp_path / "prepared", row, "all64", config, max_gib=1, reserve_gib=0
        )
    old_hashes = {k: file_hash(v) for k, v in old["paths"].items()}
    assert current != "imagechd-training-inputs-v1"
    new = cpu_cache.cached_input(
        tmp_path / "inputs", tmp_path / "prepared", row, "all64", config, max_gib=1, reserve_gib=0
    )
    assert not new["reused"] and new["paths"] != old["paths"]
    assert old_hashes == {k: file_hash(v) for k, v in old["paths"].items()}


def test_unversioned_current_model_rejected_without_reading_weights(tmp_path):
    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.common import MODEL_FORMAT

    (tmp_path / "models.json").write_text(json.dumps({"format": MODEL_FORMAT}))
    with pytest.raises(ValueError, match="协议.*重新训练"):
        read_collection(tmp_path)


@pytest.mark.parametrize("folds", [1, 3])
def test_old_training_contract_fails_before_tuning_or_overwriting(raw_chd, tmp_path, monkeypatch, folds):
    from chd_ct.imagechd import autotune, train
    from chd_ct.imagechd.config import load_config
    from chd_ct.imagechd.preprocess import prepare

    cache = tmp_path / "prepared"
    prepare(raw_chd, cache, size=16)
    config = load_config(Path(__file__).parents[1] / "configs/chd-smoke.yaml")
    config["cross_validation"]["folds"] = folds
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "models"
    output.mkdir()
    contract = output / "training-contract.json"
    contract.write_text(json.dumps({"recovery_version": 1}))
    old = contract.read_bytes()
    monkeypatch.setattr(autotune, "resolve", lambda *a: pytest.fail("old method entered tuning"))
    with pytest.raises(ValueError, match="协议.*重新训练"):
        train.train(cache, output, config_path, "smoke", resume=True)
    assert contract.read_bytes() == old
    assert not (output / "models.json").exists()
