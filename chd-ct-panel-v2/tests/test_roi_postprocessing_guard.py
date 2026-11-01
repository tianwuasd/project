"""Scientific/recovery boundaries; synthetic CPU arrays and mocked CUDA only."""

import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from chd_ct_v2.imagechd import gpu_memory
from chd_ct_v2.imagechd.postprocessing import (
    apply_policy,
    choose_policy,
    keep_largest,
    limited_growth,
    selection_scores,
)
from chd_ct_v2.imagechd.roi_inputs import row_roi
from chd_ct_v2.recovery import read_json

ROOT = Path(__file__).resolve().parents[1]


def test_growth_preserves_all_seeds_even_when_blood_misses_them():
    vote = np.zeros((9, 9, 9), np.uint8)
    vote[4, 4, 4] = 1
    vote[1, 1, 1] = 5
    assert np.array_equal(limited_growth(vote, np.zeros_like(vote)), vote)
    grown = limited_growth(vote, np.ones_like(vote), 2)
    assert np.array_equal(grown[vote > 0], vote[vote > 0])
    assert grown[4, 4, 6] == 1 and grown[4, 4, 7] == 0


def test_growth_ties_and_boundary_are_not_filled():
    vote = np.zeros((5, 5, 5), np.uint8)
    vote[1, 2, 2], vote[3, 2, 2] = 1, 2
    blood = np.ones_like(vote)
    blood[0, 2, 2] = 2
    grown = limited_growth(vote, blood, 3)
    assert grown[2, 2, 2] == 0 and grown[0, 2, 2] == 0


def records(vote, limited, target):
    result = {"scores": {}, "missing": {}}
    for name, array in (("vote", vote), ("limited", limited)):
        for key, candidate in [(name, array)] + [
            (f"{name}_lcc_{label}", keep_largest(array, label)) for label in range(1, 8)
        ]:
            score, missing = selection_scores(candidate, target)
            result["scores"][key], result["missing"][key] = score.tolist(), missing.tolist()
    return [result]


def test_validation_accepts_beneficial_growth_and_rejects_harm():
    target = np.zeros((5, 5, 5), np.uint8)
    target[2, 2, 2:4] = 1
    vote = target.copy()
    vote[2, 2, 3] = 0
    accepted = choose_policy(records(vote, target, target))
    assert accepted["refinement"] == "limited"
    wrong = target.copy()
    wrong[0, 0, 0] = 2
    rejected = choose_policy(records(vote, wrong, target))
    assert rejected["refinement"] == "vote"  # LV gain cannot justify an RV regression.


def test_component_selection_preserves_valid_multiple_islands():
    target = np.zeros((8, 8, 8), np.uint8)
    target[1:3, 1:3, 1:3] = 1
    target[6, 6, 6] = 1
    policy = choose_policy(records(target, target, target))
    assert policy["largest_components"] == []
    vote = target.copy()
    vote[0, 7, 0] = 2
    # The false island in an absent class cannot be removed by keep-largest;
    # reject rather than pretending the operation fixes it.
    assert 2 not in choose_policy(records(vote, vote, target))["largest_components"]
    target[6, 6, 6] = 0
    policy = choose_policy(records(vote, vote, target))
    assert policy["largest_components"] == [1]
    assert apply_policy(vote, vote, policy)[6, 6, 6] == 0


def test_missing_roi_is_rejected_and_bounds_are_checked():
    with pytest.raises(ValueError, match="预测 ROI"):
        row_roi({}, (8, 8, 8))
    with pytest.raises(ValueError):
        row_roi({"predicted_roi": [[-1, 4], [0, 5], [0, 5]]}, (8, 8, 8))


def fake_cuda(monkeypatch, free_gib=24):
    state = {"free": int(free_gib * gpu_memory.GIB), "fraction": 1.0, "empty": 0, "current": 2, "devices": []}
    cuda = gpu_memory.torch.cuda
    monkeypatch.setattr(cuda, "current_device", lambda: state["current"])

    def indexed(device):
        device = gpu_memory.torch.device(device)
        if device.index is None:
            raise ValueError(f"Expected a torch.device with a specified index, but got: {device}")
        state["devices"].append(str(device))

    def get_fraction(device):
        indexed(device)
        return state["fraction"]

    def set_fraction(value, device):
        indexed(device)
        state["fraction"] = value

    monkeypatch.setattr(cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cuda, "empty_cache", lambda: state.update(empty=state["empty"] + 1))
    monkeypatch.setattr(cuda, "mem_get_info", lambda *_: (state["free"], 24 * gpu_memory.GIB))
    monkeypatch.setattr(cuda, "get_per_process_memory_fraction", get_fraction, raising=False)
    monkeypatch.setattr(cuda, "set_per_process_memory_fraction", set_fraction)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(cuda, "memory_allocated", lambda *_: gpu_memory.GIB)
    monkeypatch.setattr(cuda, "memory_reserved", lambda *_: 2 * gpu_memory.GIB)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda *_: gpu_memory.GIB)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda *_: 2 * gpu_memory.GIB)
    monkeypatch.setattr(gpu_memory, "foreign_processes", lambda _: [])
    return state


def test_soft_budget_allows_reserve_use_and_warns_once(monkeypatch, tmp_path, capsys):
    state = fake_cuda(monkeypatch, 23)
    state["fraction"] = 0.7
    path = tmp_path / "soft.json"
    with gpu_memory.gpu_budget("cuda", path, {}) as guard:
        assert state["fraction"] == 1.0
        assert guard.limit == int(18.4 * gpu_memory.GIB)
        # Cached blocks also count toward allocator capacity, not just tensors.
        state["free"] = 3 * gpu_memory.GIB
        monkeypatch.setattr(gpu_memory.torch.cuda, "memory_allocated", lambda *_: 19 * gpu_memory.GIB)
        monkeypatch.setattr(gpu_memory.torch.cuda, "memory_reserved", lambda *_: 20 * gpu_memory.GIB)
        guard.check()
        guard.check()
        record = read_json(path)
        assert record["budget_mode"] == "soft"
        assert record["soft_budget_exceeded"] is True
        assert record["first_exceeded_reserved_bytes"] == 20 * gpu_memory.GIB
        assert capsys.readouterr().out.count("[gpu-memory] 超过软预算 ") == 1
        state["free"] = 100
        with pytest.raises(RuntimeError, match="余量"):
            guard.check()
    assert state["fraction"] == 0.7


@pytest.mark.parametrize("device, expected", [("cuda", "cuda:2"), (gpu_memory.torch.device("cuda"), "cuda:2"), ("cuda:1", "cuda:1")])
def test_gpu_budget_resolves_process_local_index_once(monkeypatch, tmp_path, device, expected):
    state = fake_cuda(monkeypatch)
    state["fraction"] = 0.7
    path = tmp_path / "indexed.json"
    with gpu_memory.gpu_budget(device, path, {}) as guard:
        assert str(guard.device) == expected
        state["current"] = 3
        guard.check()
    assert state["fraction"] == 0.7
    assert state["devices"] == [expected, expected, expected]
    assert read_json(path)["device"] == expected


def test_gpu_budget_without_fraction_getter(monkeypatch, tmp_path):
    state = fake_cuda(monkeypatch)
    monkeypatch.delattr(gpu_memory.torch.cuda, "get_per_process_memory_fraction")
    with gpu_memory.gpu_budget("cuda", tmp_path / "legacy.json", {}):
        assert state["fraction"] == 1.0
    assert state["devices"] == ["cuda:2", "cuda:2"]
    assert state["fraction"] == 1.0


def test_cpu_budget_never_resolves_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_memory.torch.cuda, "current_device", lambda: pytest.fail("CPU must not initialize CUDA"))
    path = tmp_path / "cpu.json"
    with gpu_memory.gpu_budget("cpu", path, {}) as guard:
        guard.check()
    assert not path.exists()


def test_gpu_soft_budget_restores_previous_cap(monkeypatch, tmp_path):
    state = fake_cuda(monkeypatch)
    path = tmp_path / "gpu.json"
    with gpu_memory.gpu_budget("cuda", path, {}) as guard:
        assert state["fraction"] == 1.0
        guard.check()
    assert state["fraction"] == 1 and state["empty"] == 1
    assert read_json(path)["status"] == "complete"


def test_initial_low_memory_is_recorded(monkeypatch, tmp_path):
    fake_cuda(monkeypatch, 0.25)
    with pytest.raises(RuntimeError, match="显存"):
        with gpu_memory.gpu_budget("cuda", tmp_path / "gpu.json", {}):
            pytest.fail("should not start")
    assert read_json(tmp_path / "gpu.json")["status"] == "failed"


def test_soft_reserve_is_not_an_initial_admission_limit(monkeypatch, tmp_path):
    fake_cuda(monkeypatch, 1.5)
    with gpu_memory.gpu_budget("cuda", tmp_path / "small.json", {}) as guard:
        assert guard.limit == 0
        guard.check()


def test_old_task_config_uses_80_percent_without_mutating_contract(monkeypatch, tmp_path):
    fake_cuda(monkeypatch, 23)
    config = {"gpu_guard": {"fraction": 0.85, "reserve_gib": 2.0, "minimum_free_gib": 0.5}}
    before = copy.deepcopy(config)
    with gpu_memory.gpu_budget("cuda", tmp_path / "old.json", config) as guard:
        assert guard.limit == int(18.4 * gpu_memory.GIB)
        assert guard.record["soft_fraction"] == 0.8
    assert config == before


def test_foreign_process_and_low_free_stop_without_touching_pid(monkeypatch, tmp_path):
    state = fake_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="非本任务"):
        monkeypatch.setattr(gpu_memory, "foreign_processes", lambda _: [12345])
        with gpu_memory.gpu_budget("cuda", tmp_path / "foreign.json", {}):
            pass
    assert state["fraction"] == 1
    monkeypatch.setattr(gpu_memory, "foreign_processes", lambda _: [])
    with pytest.raises(RuntimeError, match="余量"):
        with gpu_memory.gpu_budget("cuda", tmp_path / "low.json", {}) as guard:
            state["free"] = 100
            guard.check()


def test_original_oom_survives_diagnostic_error(monkeypatch, tmp_path):
    state = fake_cuda(monkeypatch)
    with pytest.raises(gpu_memory.torch.OutOfMemoryError, match="original"):
        with gpu_memory.gpu_budget("cuda", tmp_path / "oom.json", {}):
            monkeypatch.setattr(
                gpu_memory.torch.cuda,
                "max_memory_allocated",
                lambda *_: (_ for _ in ()).throw(RuntimeError("stats")),
            )
            raise gpu_memory.torch.OutOfMemoryError("original")
    assert state["fraction"] == 1
    assert read_json(tmp_path / "oom.json")["diagnostic_errors"]


def test_guard_rejects_less_than_two_gib_reserve():
    with pytest.raises(ValueError):
        gpu_memory.budget_bytes(20 * gpu_memory.GIB, 24 * gpu_memory.GIB, reserve_gib=1)


def test_synthetic_pipeline_preserves_variants_and_rejects_policy_tamper(tmp_path):
    import test_chd_main

    from chd_ct_v2.imagechd.checkpoints import read_collection
    from chd_ct_v2.imagechd.predict import predict, valid_record
    from chd_ct_v2.imagechd.preprocess import prepare
    from chd_ct_v2.imagechd.train import train
    from chd_ct_v2.recovery import write_json

    cache, models, predictions = (tmp_path / p for p in ("cache", "models", "predictions"))
    prepare(test_chd_main.raw_chd.__wrapped__(tmp_path), cache, size=16)
    train(cache, models, ROOT / "configs/chd-smoke.yaml", "smoke", "cpu")
    _, collection = read_collection(models, allow_smoke=True)
    assert collection["roi"]["test_cases_used"] == 0
    report = predict(cache, models, predictions, allow_smoke=True, split="val")
    case = report["cases"][0]
    assert valid_record(predictions, case)
    with np.load(predictions / case["grid_prediction"]) as grid:
        assert set(grid.files) == {"prediction", "vote", "limited", "blood"}
        assert np.array_equal(grid["prediction"], grid["vote"])
    broken = copy.deepcopy(case)
    broken.pop("stage_outputs_format")
    assert not valid_record(predictions, broken)
    (predictions / case["stage_predictions"]["limited"]["file"]).unlink()
    assert not valid_record(predictions, case)
    policy = read_json(models / "postprocessing.json")
    policy["policy"]["refinement"] = "limited"
    write_json(models / "postprocessing.json", policy)
    with pytest.raises(ValueError, match="后处理"):
        train(cache, models, ROOT / "configs/chd-smoke.yaml", "smoke", "cpu", resume=True)


def test_formal_calibration_uses_only_validation_and_freezes_sources(tmp_path, monkeypatch):
    import test_chd_main
    import yaml

    from chd_ct_v2.imagechd import calibration
    from chd_ct_v2.imagechd.checkpoints import read_collection
    from chd_ct_v2.imagechd.preprocess import prepare
    from chd_ct_v2.imagechd.train import train
    from chd_ct_v2.recovery import write_json

    cache, models = tmp_path / "cache", tmp_path / "models"
    data = prepare(test_chd_main.raw_chd.__wrapped__(tmp_path), cache, size=16)
    config = yaml.safe_load((ROOT / "configs/chd-smoke.yaml").read_text())
    config["profile"] = "synthetic_calibration_test"
    config_path = tmp_path / "small.yaml"
    config_path.write_text(yaml.safe_dump(config))
    seen = []
    original = calibration.load_case

    def load(directory, row, **kwargs):
        seen.append((row["case_id"], kwargs.get("labels", True)))
        return original(directory, row, **kwargs)

    monkeypatch.setattr(calibration, "load_case", load)
    train(cache, models, config_path, "train", "cpu")
    _, collection = read_collection(models)
    policy = read_json(models / "postprocessing.json")
    expected = {r["case_id"] for r in data["cases"] if r["split"] == "val"}
    assert {case for case, _ in seen} == expected
    assert all(
        [labels for case, labels in seen if case == selected] == [False, True] for selected in expected
    )
    assert policy["test_cases_used"] == 0 and policy["policy"]["calibrated"]
    assert {r["case_id"] for r in policy["cases"]} == expected
    assert calibration.load_policy(models, collection) == policy["policy"]
    ensemble = dict(collection, _members=[{"directory": models, "collection": collection}] * 3)
    assert (
        calibration.load_policy(models, ensemble)["reason"]
        == "external_ensemble_has_no_independent_calibration"
    )
    wrong = copy.deepcopy(collection)
    wrong["stages"]["all64"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="来源"):
        calibration.load_policy(models, wrong)
    roi = read_json(models / "roi.json")
    roi["cases"][next(iter(roi["cases"]))]["roi_slices"][0][1] -= 1
    write_json(models / "roi.json", roi)
    with pytest.raises(ValueError, match="ROI"):
        train(cache, models, config_path, "train", "cpu", resume=True)


def test_foreign_process_detection_excludes_only_verified_workflow_pids(monkeypatch):
    from types import SimpleNamespace

    from chd_ct_v2.server import gpu_hold

    monkeypatch.setattr(
        gpu_memory.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-test")
    )
    monkeypatch.setattr(gpu_hold, "owned_processes", lambda: {123})
    monkeypatch.setattr(
        gpu_memory.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=f"GPU-test, {gpu_memory.os.getpid()}\nGPU-test, 123\nGPU-test, 456\nGPU-other, 789\n"
        ),
    )
    assert gpu_memory.foreign_processes("cuda") == [456]
