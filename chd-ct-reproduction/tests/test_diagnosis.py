"""Contracts for independent interpretable diagnosis."""

import json

import numpy as np
import pytest


def test_labels_preserve_explicit_zero_and_track_blank_assumption(tmp_path):
    from chd_ct.diagnosis.labels import import_labels

    source = tmp_path / "truth.csv"
    source.write_text("case_id,ASD,VSD\nct_1001,1,\nct_1002,0,1\n", encoding="utf-8")
    report = import_labels(source, tmp_path / "labels")
    assert report["blank_policy"] == "negative"
    assert report["cases"][0]["labels"]["VSD"] == 0
    assert "VSD" in report["cases"][0]["assumed_negative"]
    assert "ASD" not in report["cases"][1]["assumed_negative"]
    unknown = import_labels(source, tmp_path / "unknown", blank_policy="unknown")
    assert unknown["cases"][0]["labels"]["VSD"] is None
    source.write_text("case_id,ASD\nct_1001,1\nct_1001,0\n")
    with pytest.raises(ValueError, match="重复"):
        import_labels(source, tmp_path / "duplicate")


def test_anatomy_features_do_not_invent_missing_structures():
    from chd_ct.diagnosis.features import measure

    mask = np.zeros((12, 12, 12), np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    result = measure(mask)
    assert result["values"]["fraction_LV"] == 1
    assert result["values"]["fraction_RV"] == 0
    assert result["values"]["contact_LV_RV"] is None
    assert result["rule_features"]["conn_PV_RA"] is None
    assert result["rule_features"]["conn_LV_RV"] is None
    mask[0, 0, 0] = 8
    with pytest.raises(ValueError):
        measure(mask)


def test_rules_unknown_is_not_a_negative():
    from chd_ct.diagnosis.rules import evaluate_rules

    config = {"rules": {"A": {"all": [{"feature": "x", "op": "eq", "value": True}]}}, "thresholds": {}}
    assert evaluate_rules({}, config)["A"]["status"] == "indeterminate"
    assert evaluate_rules({"x": True}, config)["A"]["status"] == "positive"
    assert evaluate_rules({"x": False}, config)["A"]["status"] == "negative"


def test_synthetic_diagnosis_end_to_end(tmp_path):
    from chd_ct.diagnosis.demo import run_demo

    result = run_demo(tmp_path / "demo")
    assert result["status"] == "passed"
    assert result["trained_diseases"] >= 1
    report = json.loads((tmp_path / "demo/prediction/diagnosis.json").read_text(encoding="utf-8"))
    finding = report["cases"][0]["classifier"]["ASD"]
    assert finding["path"]
    assert finding["leaf"]["samples"] >= 2
    assert finding["probability"] == finding["leaf"]["positive"] / finding["leaf"]["samples"]
    metrics = json.loads((tmp_path / "demo/evaluation/diagnosis-metrics.json").read_text(encoding="utf-8"))
    assert metrics["cases_evaluated"] == 4
    assert metrics["classifier"]["ASD"]["accuracy"] == 1


@pytest.fixture
def demo_artifacts(tmp_path):
    from chd_ct.diagnosis.demo import run_demo

    root = tmp_path / "sample"
    run_demo(root)
    return root


def test_training_ignores_test_truth_and_feature_values(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.classifier import train_classifier

    root = demo_artifacts
    features = json.loads((root / "features.json").read_text(encoding="utf-8"))
    truth = json.loads((root / "labels/labels.json").read_text(encoding="utf-8"))
    test_ids = {row["case_id"] for row in features["cases"] if row["split"] == "test"}
    for row in features["cases"]:
        if row["case_id"] in test_ids:
            row["values"] = dict.fromkeys(row["values"], 1e10)
    for row in truth["cases"]:
        if row["case_id"] in test_ids:
            row["labels"]["ASD"] = 1 - row["labels"]["ASD"]
    changed_features = tmp_path / "changed-features.json"
    changed_truth = tmp_path / "changed-truth.json"
    changed_features.write_text(json.dumps(features))
    changed_truth.write_text(json.dumps(truth))
    train_classifier(changed_features, changed_truth, tmp_path / "changed-model", allow_smoke=True)
    original = json.loads((root / "training/classifier.json").read_text(encoding="utf-8"))
    changed = json.loads((tmp_path / "changed-model/classifier.json").read_text(encoding="utf-8"))
    assert changed["trees"] == original["trees"]
    assert changed["skipped"] == original["skipped"]
    assert changed["development_patients"] == original["development_patients"]


def test_exported_tree_matches_sklearn_and_missing_branch_abstains():
    from sklearn.tree import DecisionTreeClassifier

    from chd_ct.diagnosis.classifier import explain_tree, export_tree
    from chd_ct.diagnosis.features import FEATURE_NAMES

    x = np.zeros((8, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = np.arange(8)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=42).fit(x, y)
    record = {"nodes": export_tree(tree, x, y)}
    for row in x:
        result = explain_tree(record, dict(zip(FEATURE_NAMES, row)))
        assert int(result["status"] == "positive") == tree.predict(row[None])[0]
        assert result["probability"] == tree.predict_proba(row[None])[0, 1]
    values = dict(zip(FEATURE_NAMES, x[0]))
    values[FEATURE_NAMES[0]] = None
    result = explain_tree(record, values)
    assert result["status"] == "indeterminate" and result["leaf"] is None


def test_no_positive_class_cannot_produce_a_trained_classifier(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.classifier import train_classifier

    root = demo_artifacts
    truth = json.loads((root / "labels/labels.json").read_text(encoding="utf-8"))
    for row in truth["cases"]:
        row["labels"] = dict.fromkeys(row["labels"], 0)
    source = tmp_path / "all-negative.json"
    source.write_text(json.dumps(truth))
    result = train_classifier(root / "features.json", source, tmp_path / "untrainable", allow_smoke=True)
    assert result["status"] == "not_trainable"
    assert not (tmp_path / "untrainable/classifier.json").exists()


def test_patient_split_overlap_and_corrupt_tree_rejected(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.classifier import read_classifier
    from chd_ct.diagnosis.features import read_features

    root = demo_artifacts
    data = json.loads((root / "features.json").read_text(encoding="utf-8"))
    data["cases"][-1]["patient_id"] = data["cases"][0]["patient_id"]
    source = tmp_path / "bad-feature.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="跨数据划分"):
        read_features(source)
    model = json.loads((root / "training/classifier.json").read_text(encoding="utf-8"))
    model["trees"]["ASD"]["nodes"][0]["left"] = 0
    source.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="循环"):
        read_classifier(source)


def test_evaluation_rejects_model_development_case(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.evaluate import evaluate_diagnosis, metrics

    root = demo_artifacts
    pred = json.loads((root / "prediction/diagnosis.json").read_text(encoding="utf-8"))
    pred["development_cases"].append(pred["cases"][0]["case_id"])
    source = tmp_path / "bad-prediction.json"
    source.write_text(json.dumps(pred))
    with pytest.raises(ValueError, match="泄漏"):
        evaluate_diagnosis(root / "features.json", root / "labels", source, tmp_path / "eval")
    result = metrics(
        [(1, {"status": "positive"}), (0, {"status": "indeterminate"}), (None, {"status": "positive"})]
    )
    assert result["coverage"] == 0.5 and result["accuracy"] == 1
    assert result["correct_fraction_all_known"] == 0.5 and result["unknown_truth"] == 1


def test_feature_extraction_reads_only_prediction_not_target(tmp_path):
    from chd_ct.diagnosis.features import extract
    from chd_ct.diagnosis.io import file_hash
    from chd_ct.imagechd.common import FORMAT, IGNORE, LABELS, NORMALIZATION

    prepared, predictions = tmp_path / "prepared", tmp_path / "predictions"
    prepared.mkdir()
    predictions.mkdir()
    # This file exists for cache provenance but is deliberately not a readable NPZ.
    (prepared / "unreadable.npz").write_text("must never open image or target")
    manifest = {
        "format": FORMAT,
        "status": "prepared",
        "labels": list(LABELS),
        "normalization": NORMALIZATION,
        "ignore_index": IGNORE,
        "resolution": "native",
        "cases": [
            {
                "case_id": "ct_1001",
                "patient_id": "p1",
                "split": "train",
                "cache": "unreadable.npz",
                "cache_shape": [8] * 3,
                "canonical_shape": [8] * 3,
                "native_shape": [8] * 3,
                "native_affine": np.eye(4).tolist(),
            }
        ],
    }
    (prepared / "dataset.json").write_text(json.dumps(manifest))
    mask = np.zeros((8, 8, 8), np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    np.savez(predictions / "pred.npz", prediction=mask)
    report = {
        "status": "passed",
        "prepared_sha256": file_hash(prepared / "dataset.json"),
        "models_sha256": "model-test",
        "model_mode": "train",
        "cases": [{"case_id": "ct_1001", "grid_prediction": "pred.npz"}],
    }
    (predictions / "prediction-report.json").write_text(json.dumps(report))
    result = extract(prepared, predictions, tmp_path / "features")
    assert result["cases"][0]["values"]["fraction_LV"] == 1
    assert result["segmentation_mode"] == "train"


def test_rule_inference_requires_reviewed_evidence(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.predict import diagnose

    root = demo_artifacts
    result = diagnose(root / "features.json", tmp_path / "no-evidence", method="rules", allow_smoke=True)
    assert result["cases"][0]["rules"]["ASD"]["status"] == "indeterminate"
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "synthetic_16",
                        "source": "synthetic fixture review",
                        "features": {"conn_LA_RA": True, "single_atrium_confirmed": False},
                    }
                ]
            }
        )
    )
    result = diagnose(
        root / "features.json", tmp_path / "reviewed", method="rules", allow_smoke=True, evidence=evidence
    )
    assert result["cases"][0]["rules"]["ASD"]["status"] == "positive"


def test_diagnosis_cli_untrainable_and_existing_output(demo_artifacts):
    from chd_ct.diagnosis.cli import main

    assert main(["demo", "--output", str(demo_artifacts)]) == 1


@pytest.mark.parametrize(
    "features",
    [
        {"pa_atresia_confirmed": 2},
        {"conn_LA_RA": -1},
        {"n_island_AO": -1},
        {"n_island_AO": 0.2},
        {"ao_override_fraction": 1.2},
        {"unknown_key": True},
    ],
)
def test_invalid_clinical_evidence_cannot_become_negative(features):
    from chd_ct.diagnosis.rules import validate_evidence

    with pytest.raises(ValueError):
        validate_evidence(features)


def test_formal_training_requires_segmentation_split_provenance(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.classifier import train_classifier

    root = demo_artifacts
    features = json.loads((root / "features.json").read_text(encoding="utf-8"))
    features["segmentation_mode"] = "train"
    features["segmentation_prepared_sha256"] = "another-fixed-patient-split"
    source = tmp_path / "changed-split.json"
    source.write_text(json.dumps(features))
    with pytest.raises(ValueError, match="同一预处理清单"):
        train_classifier(source, root / "labels", tmp_path / "model")


def test_metrics_disclose_unaligned_rule_diseases(demo_artifacts):
    result = json.loads((demo_artifacts / "evaluation/diagnosis-metrics.json").read_text(encoding="utf-8"))
    assert "PuA" in result["rules_alignment"]["rule_diseases_without_matching_truth"]
    assert "PA" in result["rules_alignment"]["label_diseases_without_rule"]
    assert "ASD" in result["rules_alignment"]["matched_diseases"]


def test_software_classifier_stays_marked_as_software(demo_artifacts, tmp_path):
    from chd_ct.diagnosis.predict import diagnose

    root = demo_artifacts
    features = json.loads((root / "features.json").read_text(encoding="utf-8"))
    features["segmentation_mode"] = "train"
    source = tmp_path / "formal-looking-features.json"
    source.write_text(json.dumps(features))
    with pytest.raises(ValueError, match="软件短测分类器"):
        diagnose(source, tmp_path / "prediction", classifier=root / "training")


def test_html_labels_prior_only_tree_and_escapes_case():
    from chd_ct.diagnosis.predict import render_report

    report = {
        "classifier_blank_policy": "negative",
        "cases": [
            {
                "case_id": "<script>alert(1)</script>",
                "rules": {},
                "classifier": {
                    "ASD": {
                        "status": "positive",
                        "path": [],
                        "leaf": {"id": 0, "samples": 6, "positive": 4},
                        "probability": 4 / 6,
                        "reason": "no_split_training_prior",
                    }
                },
            }
        ],
    }
    page = render_report(report)
    assert "训练先验" in page and "没有个体解剖分支依据" in page
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_installed_diagnosis_layout_has_rules_preset(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    site = tmp_path / "site-packages"
    shutil.copytree(root / "src/chd_ct", site / "chd_ct", ignore=shutil.ignore_patterns("__pycache__"))
    outside = tmp_path / "outside"
    outside.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "chd_ct.diagnosis.cli", "demo", "--output", str(outside / "demo")],
        cwd=outside,
        env=dict(os.environ, PYTHONPATH=str(site), PYTHONIOENCODING="utf-8"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((outside / "demo/prediction/diagnosis.json").read_text(encoding="utf-8"))
    assert report["effective_rules"]["name"] == "independent_candidate_rules_v2"


def test_sparse_xlsx_blank_cells_and_formula_rejection(tmp_path):
    from zipfile import ZipFile

    from chd_ct.diagnosis.labels import import_labels

    source = tmp_path / "sparse.xlsx"
    workbook = '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>'
    relationships = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
    sheet = '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>index</t></is></c><c r="B1" t="inlineStr"><is><t>ASD</t></is></c><c r="C1" t="inlineStr"><is><t>VSD</t></is></c></row><row r="2"><c r="A2"><v>1</v></c><c r="C2"><v>1</v></c></row></sheetData></worksheet>'

    def make(text):
        with ZipFile(source, "w") as z:
            z.writestr("xl/workbook.xml", workbook)
            z.writestr("xl/_rels/workbook.xml.rels", relationships)
            z.writestr("xl/worksheets/sheet1.xml", text)

    make(sheet)
    result = import_labels(source, tmp_path / "imported")
    row = result["cases"][0]
    assert row["case_id"] == "ct_1001" and row["labels"]["ASD"] == 0 and row["labels"]["VSD"] == 1
    assert row["assumed_negative"] == ["ASD"]
    make(sheet.replace("<v>1</v></c></row>", "<f>1+0</f><v>1</v></c></row>"))
    with pytest.raises(ValueError, match="公式"):
        import_labels(source, tmp_path / "formula")
