"""Real diagnosis-index conventions and immutable legacy-label recovery."""

import copy
import json

import pytest


@pytest.mark.parametrize(
    "index,expected",
    [(1, "ct_1001"), (178, "ct_1178"), (1001, "ct_1001"), (1178, "ct_1178"), ("1001.0", "ct_1001")],
)
def test_diagnosis_index_preserves_full_case_number(tmp_path, index, expected):
    from chd_ct.diagnosis.labels import import_labels

    source = tmp_path / "truth.csv"
    source.write_text(f"index,ASD\n{index},1\n", encoding="utf-8")
    result = import_labels(source, tmp_path / "labels")
    assert result["cases"][0]["case_id"] == expected
    assert result["id_policy"] != "ImageCHD index + 1000"


@pytest.mark.parametrize("index", ["-1", "1.5", "nan", "inf", "", "ct_1001"])
def test_invalid_numeric_index_is_rejected(tmp_path, index):
    from chd_ct.diagnosis.labels import import_labels

    source = tmp_path / "truth.csv"
    source.write_text(f"index,ASD\n{index},1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index"):
        import_labels(source, tmp_path / "labels")


def test_index_alias_collision_is_rejected(tmp_path):
    from chd_ct.diagnosis.labels import import_labels

    source = tmp_path / "truth.csv"
    source.write_text("index,ASD\n1,1\n1001,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        import_labels(source, tmp_path / "labels")


def test_explicit_case_id_is_never_shifted(tmp_path):
    from chd_ct.diagnosis.labels import import_labels

    source = tmp_path / "truth.csv"
    source.write_text("case_id,index,ASD\nct_2001,1001,1\n", encoding="utf-8")
    result = import_labels(source, tmp_path / "labels")
    assert result["cases"][0]["case_id"] == "ct_2001"
    assert result["id_policy"] == "case_id"


def test_legacy_alignment_is_complete_and_does_not_mutate_truth(capsys):
    from chd_ct.diagnosis.labels import align_labels

    truth = {
        "id_policy": "ImageCHD index + 1000",
        "cases": [
            {"case_id": "ct_2001", "labels": {"ASD": 1}},
            {"case_id": "ct_2002", "labels": {"ASD": 0}},
            {"case_id": "ct_2178", "labels": {"ASD": None}},
        ],
    }
    before = copy.deepcopy(truth)
    indexed, audit = align_labels(truth, ["ct_1001", "ct_1178"])
    assert indexed["ct_1001"]["labels"]["ASD"] == 1
    assert indexed["ct_1178"]["labels"]["ASD"] is None
    assert audit["mode"] == "legacy_index_offset_recovery"
    assert audit["matched_cases"] == 2
    assert audit["mapping"]["ct_2178"] == "ct_1178"
    assert truth == before
    assert "旧版" in capsys.readouterr().out


@pytest.mark.parametrize(
    "policy,ids,expected",
    [
        ("case_id", ["ct_2001"], ["ct_1001"]),
        ("ImageCHD index + 1000", ["ct_2001"], ["ct_1001", "ct_1002"]),
        ("ImageCHD index + 1000", ["ct_1001", "ct_2002"], ["ct_1001", "ct_1002"]),
        ("ImageCHD index + 1000", ["ct_2001", "ct_3001"], ["ct_1001"]),
    ],
)
def test_legacy_alignment_rejects_unproven_or_partial_mapping(policy, ids, expected):
    from chd_ct.diagnosis.labels import align_labels

    truth = {"id_policy": policy, "cases": [{"case_id": case} for case in ids]}
    with pytest.raises(ValueError, match="未覆盖"):
        align_labels(truth, expected)


def test_existing_correct_legacy_labels_are_used_directly(capsys):
    from chd_ct.diagnosis.labels import align_labels

    truth = {"id_policy": "ImageCHD index + 1000", "cases": [{"case_id": "ct_1001"}]}
    indexed, audit = align_labels(truth, ["ct_1001"])
    assert "ct_1001" in indexed and audit["mode"] == "exact"
    assert not capsys.readouterr().out


def test_real_style_indices_and_legacy_artifact_train_and_evaluate_identically(tmp_path):
    from chd_ct.diagnosis.classifier import train_classifier
    from chd_ct.diagnosis.demo import run_demo
    from chd_ct.diagnosis.evaluate import evaluate_diagnosis
    from chd_ct.diagnosis.io import file_hash
    from chd_ct.diagnosis.labels import import_labels
    from chd_ct.diagnosis.predict import diagnose

    root = tmp_path / "demo"
    run_demo(root)
    features = json.loads((root / "features.json").read_text(encoding="utf-8"))
    truth = json.loads((root / "labels/labels.json").read_text(encoding="utf-8"))
    source = tmp_path / "truth.csv"
    lines = ["index,ASD"]
    truth_index = {r["case_id"]: r for r in truth["cases"]}
    for i, row in enumerate(features["cases"], 1001):
        lines.append(f"{i},{truth_index[row['case_id']]['labels']['ASD']}")
        row["case_id"] = row["patient_id"] = f"ct_{i}"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps(features), encoding="utf-8")
    fixed = import_labels(source, tmp_path / "correct-labels")
    legacy = copy.deepcopy(fixed)
    legacy["id_policy"] = "ImageCHD index + 1000"
    for row in legacy["cases"]:
        row["case_id"] = "ct_" + str(int(row["case_id"][3:]) + 1000)
    legacy_path = tmp_path / "legacy-labels.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    original_bytes, original_mtime = legacy_path.read_bytes(), legacy_path.stat().st_mtime_ns
    models, metrics = [], []
    for name, labels in (("fixed", tmp_path / "correct-labels"), ("legacy", legacy_path)):
        model_dir = tmp_path / (name + "-model")
        trained = train_classifier(feature_path, labels, model_dir, allow_smoke=True)
        assert trained["status"] == "passed"
        pred_dir = tmp_path / (name + "-predictions")
        diagnose(feature_path, pred_dir, classifier=model_dir, allow_smoke=True)
        evaluated = evaluate_diagnosis(feature_path, labels, pred_dir, tmp_path / (name + "-evaluation"))
        assert evaluated["status"] == "passed" and evaluated["cases_evaluated"] == 4
        models.append(json.loads((model_dir / "classifier.json").read_text(encoding="utf-8")))
        metrics.append(evaluated["classifier"])
    assert models[0]["trees"] == models[1]["trees"]
    assert metrics[0] == metrics[1]
    assert models[1]["labels_alignment"]["mode"] == "legacy_index_offset_recovery"
    assert models[1]["labels_sha256"] == file_hash(legacy_path)
    assert legacy_path.read_bytes() == original_bytes
    assert legacy_path.stat().st_mtime_ns == original_mtime
