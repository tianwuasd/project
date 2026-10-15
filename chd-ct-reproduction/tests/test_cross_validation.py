"""Patient leakage, fold protocol, model pairing and OOF evaluation regressions."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import test_chd_main

from chd_ct.imagechd.folds import apply_fold, make_plan, select_members

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cv_run(tmp_path):
    from chd_ct.imagechd.preprocess import prepare
    from chd_ct.imagechd.train import train

    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    cache, models = tmp_path / "prepared", tmp_path / "models"
    prepare(raw, cache, size=16)
    report = train(cache, models, ROOT / "configs/chd-cv-smoke.yaml", mode="smoke")
    return cache, models, report


def cohort(n=97):
    return {
        "id_policy": "filename_as_patient",
        "cases": [
            {
                "case_id": f"ct_{i:03}",
                "patient_id": f"p{i:03}",
                "split": ("test" if i < 15 else "val" if i < 30 else "train"),
            }
            for i in range(n)
        ],
    }


def test_threefold_patient_separation_coverage_and_counts():
    data = cohort()
    prior = copy.deepcopy(data)
    plan = make_plan(data, 42, "auto")
    assert make_plan(data, 42, "auto") == plan
    assert data == prior
    seen = []
    counts = []
    for fold in plan["folds"]:
        sets = {k: set(v) for k, v in fold["patients"].items()}
        assert not sets["train"] & sets["val"]
        assert not sets["train"] & sets["test"]
        assert not sets["val"] & sets["test"]
        assert len(set.union(*sets.values())) == 97
        seen += fold["patients"]["test"]
        counts.append(tuple(len(fold["cases"][k]) for k in ("train", "val", "test")))
        applied = apply_fold(data, fold)
        assert {r["case_id"] for r in applied["cases"] if r["split"] == "test"} == set(fold["cases"]["test"])
    assert len(seen) == len(set(seen)) == 97
    assert sorted(counts) == [(54, 10, 33), (55, 10, 32), (55, 10, 32)]
    assert make_plan(data, 43, "auto") != plan


def test_explicit_patient_split_preserves_external_holdout():
    data = cohort()
    data["id_policy"] = "explicit_patient_split"
    data["cases"].append({**data["cases"][55], "case_id": "second_scan"})
    plan = make_plan(data, 42, "auto")
    assert plan["development_scope"] == "train_val"
    assert len(plan["external_case_ids"]) == 15
    for fold in plan["folds"]:
        for part in ("train", "val", "test"):
            assert ("ct_055" in fold["cases"][part]) == ("second_scan" in fold["cases"][part])
            assert not set(fold["cases"][part]) & set(plan["external_case_ids"])
    assert make_plan(data, 42, "all")["external_case_ids"] == []


def test_tiny_cohort_and_conflicting_patient_assignments():
    data = cohort(3)
    for fold in make_plan(data, 0, "all")["folds"]:
        assert all(len(v) == 1 for v in fold["cases"].values())
    with pytest.raises(ValueError, match="至少.*3"):
        make_plan(cohort(2), 0, "all")
    data = cohort()
    data["cases"][80]["patient_id"] = data["cases"][0]["patient_id"]
    with pytest.raises(ValueError, match="患者.*跨"):
        make_plan(data, 0, "auto")


def test_known_patients_use_only_their_held_out_model():
    data = cohort()
    plan = make_plan(data, 42, "all")
    members = [{"index": i, "directory": f"fold_{i}", "collection": {}} for i in range(3)]
    collection = {"split_plan": plan, "_members": members}
    for row in data["cases"]:
        chosen, kind = select_members(collection, row)
        assert len(chosen) == 1 and kind == "out_of_fold"
        assert row["patient_id"] in plan["folds"][chosen[0]["index"]]["patients"]["test"]
        # A different scan of the same patient must not switch to the ensemble.
        assert select_members(collection, {**row, "case_id": "different_scan"}) == (chosen, kind)
    assert select_members(collection, {"case_id": "new", "patient_id": "new"}) == (
        members,
        "threefold_ensemble",
    )
    # Relabelling a known case's patient cannot bypass the OOF routing either.
    chosen, kind = select_members(collection, {**data["cases"][0], "patient_id": "renamed"})
    assert len(chosen) == 1 and kind == "out_of_fold"


def test_eighteen_stages_pair_encoders_and_preserve_completed_folds_on_resume(cv_run, monkeypatch):
    from chd_ct.imagechd import train as trainer
    from chd_ct.imagechd.checkpoints import read_collection

    cache, models, report = cv_run
    assert report["status"] == "passed" and report["completed_folds"] == 3
    assert len(list(models.glob("fold_*/*.pt"))) == 36  # best + recovery weights, six per fold
    _, collection = read_collection(models, allow_smoke=True)
    assert len(collection["_members"]) == 3
    for member in collection["_members"]:
        saved = member["collection"]
        assert saved["config"]["fold"]["index"] == member["index"]
        assert all(s["best_epoch"] == 1 for s in saved["stages"].values())
    before = {p: p.read_bytes() for p in models.glob("fold_*/models.json")}
    monkeypatch.setattr(trainer, "train_stage", lambda *a, **k: pytest.fail("Completed folds retrained"))
    resumed = trainer.train(cache, models, ROOT / "configs/chd-cv-smoke.yaml", mode="smoke", resume=True)
    assert resumed["status"] == "passed"
    assert all(p.read_bytes() == content for p, content in before.items())
    # A fold swap remains invalid even after updating the outer file checksum.
    from chd_ct.recovery import file_hash

    record = json.loads((models / "models.json").read_text())
    (models / "fold_0/models.json").write_bytes((models / "fold_1/models.json").read_bytes())
    record["folds"]["fold_0"]["sha256"] = file_hash(models / "fold_0/models.json")
    (models / "models.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="校验|折|配置"):
        read_collection(models, allow_smoke=True)


def test_oof_evaluation_direct_fold_and_unlabelled_ensemble(cv_run, tmp_path):
    from chd_ct.imagechd.common import read_prepared, write_json
    from chd_ct.imagechd.evaluate import evaluate
    from chd_ct.imagechd.predict import predict
    from chd_ct.imagechd.test import run_test

    cache, models, _ = cv_run
    tested = run_test(cache, models, tmp_path / "oof", allow_smoke=True)
    assert tested["split"] == "cv" and tested["cases_evaluated"] == 4
    metrics = json.loads((tmp_path / "oof/evaluation/evaluation.json").read_text())
    assert metrics["protocols"] == ["out_of_fold"] and len(metrics["folds"]) == 3
    weighted = (
        sum(f["mean_case_foreground_dice"] * f["cases_evaluated"] for f in metrics["folds"].values()) / 4
    )
    assert weighted == pytest.approx(metrics["mean_case_foreground_dice"])
    single = run_test(cache, models / "fold_0", tmp_path / "fold", allow_smoke=True)
    plan = json.loads((models / "folds.json").read_text())
    assert single["cases_evaluated"] == len(plan["folds"][0]["cases"]["test"])
    predictions = json.loads((tmp_path / "fold/predictions/prediction-report.json").read_text())
    assert {r["case_id"] for r in predictions["cases"]} == set(plan["folds"][0]["cases"]["test"])

    # A new case has only image data: no cached target or raw NIfTI is available.
    _, data = read_prepared(cache)
    external = tmp_path / "external"
    external.mkdir()
    row = data["cases"][0]
    with np.load(cache / row["cache"], allow_pickle=False) as content:
        np.savez_compressed(external / "new.npz", image=content["image"])
    data.update(
        for_prediction=True, cases=[{**row, "case_id": "new", "patient_id": "new", "cache": "new.npz"}]
    )
    write_json(external / "dataset.json", data)
    report = predict(external, models, tmp_path / "external_prediction", allow_smoke=True)
    assert report["cases"][0]["inference_protocol"] == "threefold_ensemble"
    assert report["cases"][0]["fold_indices"] == [0, 1, 2]

    # Reporting an ensemble score on development patients is rejected.
    report_path = tmp_path / "oof/predictions/prediction-report.json"
    tampered = json.loads(report_path.read_text())
    tampered["cases"][0].update(inference_protocol="threefold_ensemble", fold_indices=[0, 1, 2])
    write_json(report_path, tampered)
    with pytest.raises(ValueError, match="独立评估"):
        evaluate(cache, report_path.parent, tmp_path / "leaky_evaluation")


def test_ensembles_average_probabilities_not_labels_and_keep_fold_pairing(monkeypatch):
    import torch

    from chd_ct.imagechd import predict as inference

    # Two weak class-0 votes must lose to one confident class-1 probability.
    distributions = [[0.51, 0.49, 0.00001], [0.51, 0.49, 0.00001], [0.01, 0.99, 0.00001]]
    members = [{"index": i, "directory": Path(f"fold_{i}"), "collection": {"index": i}} for i in range(3)]
    collection = {"_members": members, "config": {"stages": {"blood_lstm": {"size": 4}}}}
    calls = []

    class FakeModel:
        def __init__(self, index, recurrent=False):
            self.index, self.recurrent = index, recurrent

        def __call__(self, tensor):
            if self.recurrent:
                assert torch.all(tensor == self.index)  # every feature came from its paired encoder
            shape = (4, 4) if self.recurrent else (4, 4, 4)
            return (
                torch.tensor(distributions[self.index])
                .log()
                .reshape(1, 3, *([1] * len(shape)))
                .expand(1, 3, *shape)
            )

    class FakeEncoder:
        def load_state_dict(self, state):
            self.index = state["index"]

        def to(self, device):
            return self

        def eval(self):
            return self

        def features(self, tensor):
            return torch.full_like(tensor, self.index)

    def load(directory, saved, stage, device):
        index = saved["index"]
        assert directory == Path(f"fold_{index}")
        calls.append((stage, index))
        return FakeModel(index, stage == "blood_lstm"), {
            "spec": {"size": 4, "sequence": 3},
            "encoder_spec": {},
            "encoder_state_dict": {"index": index},
        }

    monkeypatch.setattr(inference, "load_stage", load)
    monkeypatch.setattr(inference, "build_model", lambda *args: FakeEncoder())
    image = np.zeros((4, 4, 3), np.float32)
    assert np.all(inference.anatomy(image, "crop64", Path("root"), collection, "cpu") == 1)
    assert np.all(inference.blood_prediction(image, Path("root"), collection, "cpu") == 1)
    assert calls == [(stage, i) for stage in ("crop64", "blood_lstm") for i in range(3)]


def test_partial_fold_resume_and_contract_changes_do_not_retrain_completed_fold(tmp_path, monkeypatch):
    from chd_ct.imagechd import train as trainer
    from chd_ct.imagechd.preprocess import prepare
    from chd_ct.recovery import file_hash, read_json

    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    cache, models = tmp_path / "cache", tmp_path / "models"
    prepare(raw, cache, size=16)
    original = trainer.train_stage
    calls = []

    def interrupt(directory, rows, stage, config, *args, **kwargs):
        index = config["fold"]["index"]
        calls.append((index, stage))
        expected = config["fold"]["cases"]
        assert {r["case_id"] for r in rows["train"]} == set(expected["train"])
        assert {r["case_id"] for r in rows["val"]} == set(expected["val"])
        assert not any(r["case_id"] in expected["test"] for rs in rows.values() for r in rs)
        if index == 1 and stage == "all64":
            raise KeyboardInterrupt("partial second fold")
        return original(directory, rows, stage, config, *args, **kwargs)

    monkeypatch.setattr(trainer, "train_stage", interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cache, models, ROOT / "configs/chd-cv-smoke.yaml", mode="smoke")
    assert read_json(models / "report.json")["completed_folds"] == 1
    preserved = file_hash(models / "fold_0/models.json")
    # An invalid resume must leave every durable record unchanged.
    before = {p: p.read_bytes() for p in models.rglob("*.json")}
    import yaml

    from chd_ct.imagechd.config import load_config

    config = load_config(ROOT / "configs/chd-cv-smoke.yaml")
    config["seed"] += 1
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="改变"):
        trainer.train(cache, models, changed, mode="smoke", resume=True)
    assert all(p.read_bytes() == content for p, content in before.items())

    def finish(directory, rows, stage, config, *args, **kwargs):
        assert config["fold"]["index"] != 0
        return original(directory, rows, stage, config, *args, **kwargs)

    monkeypatch.setattr(trainer, "train_stage", finish)
    assert (
        trainer.train(cache, models, ROOT / "configs/chd-cv-smoke.yaml", mode="smoke", resume=True)["status"]
        == "passed"
    )
    assert file_hash(models / "fold_0/models.json") == preserved

    # Interrupt validation again, before the first fold has been accepted. All
    # three committed hashes must survive so a later resume cannot ignore them.
    from chd_ct.imagechd import checkpoints

    original_read = checkpoints.read_collection

    def stop(*args, **kwargs):
        raise KeyboardInterrupt("resume validation interrupted")

    prior_refs = read_json(models / "models.json")["folds"]
    monkeypatch.setattr(checkpoints, "read_collection", stop)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cache, models, ROOT / "configs/chd-cv-smoke.yaml", mode="smoke", resume=True)
    assert read_json(models / "models.json")["folds"] == prior_refs
    monkeypatch.setattr(checkpoints, "read_collection", original_read)
    assert (
        trainer.train(cache, models, ROOT / "configs/chd-cv-smoke.yaml", mode="smoke", resume=True)["status"]
        == "passed"
    )


def test_stage_worker_uses_frozen_fold_instead_of_cache_split(tmp_path, monkeypatch):
    from chd_ct.imagechd import stage_worker
    from chd_ct.imagechd.common import read_prepared, write_json
    from chd_ct.imagechd.config import effective_config, load_config
    from chd_ct.imagechd.preprocess import prepare

    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    cache, output = tmp_path / "cache", tmp_path / "worker"
    output.mkdir()
    prepare(raw, cache, size=16)
    _, data = read_prepared(cache)
    fold = make_plan(data, 42)["folds"][1]
    config = effective_config(load_config(ROOT / "configs/chd-cv-smoke.yaml"), "smoke", "cpu")
    config["fold"] = fold
    write_json(output / "effective-config.json", config)

    def stage(directory, rows, name, config, *args):
        assert all(
            {r["case_id"] for r in rows[part]} == set(fold["cases"][part]) for part in ("train", "val")
        )
        return {"fold": fold["index"]}

    monkeypatch.setattr(stage_worker, "train_stage", stage)
    assert (
        stage_worker.main(
            ["--prepared", str(cache), "--output", str(output), "--stage", "crop64", "--mode", "smoke"]
        )
        == 0
    )
    assert json.loads((output / "crop64.result.json").read_text())["fold"] == 1


def test_source_config_changes_during_run_do_not_change_later_folds(tmp_path, monkeypatch):
    import yaml

    from chd_ct.imagechd import train as trainer
    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.config import load_config
    from chd_ct.imagechd.preprocess import prepare

    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    cache, models = tmp_path / "cache", tmp_path / "models"
    prepare(raw, cache, size=16)
    config = load_config(ROOT / "configs/chd-cv-smoke.yaml")
    path = tmp_path / "mutable.yaml"
    path.write_text(yaml.safe_dump(config))
    initial_seed = config["seed"]
    original = trainer.train_stage
    seen = []

    def change_source(directory, rows, stage, frozen, *args, **kwargs):
        seen.append((frozen["fold"]["index"], frozen["seed"]))
        config["seed"] = initial_seed + 1
        path.write_text(yaml.safe_dump(config))
        return original(directory, rows, stage, frozen, *args, **kwargs)

    monkeypatch.setattr(trainer, "train_stage", change_source)
    result = trainer.train(cache, models, path, mode="smoke")
    assert result["status"] == "passed"
    assert seen == [(i, initial_seed) for i in range(3) for _ in range(6)]
    assert read_collection(models, allow_smoke=True)[1]["config"]["seed"] == initial_seed


def test_threefold_and_smoke_presets_are_packaged_identically():
    from chd_ct.imagechd.config import load_config

    for name, folds in (("chd.yaml", 3), ("chd-cv-smoke.yaml", 3), ("chd-smoke.yaml", 1)):
        config = load_config(ROOT / "configs" / name)
        assert config == load_config(ROOT / "src/chd_ct/presets" / name)
        assert config["cross_validation"] == {"folds": folds, "development_scope": "auto"}
