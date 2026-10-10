"""Held-out disease metrics with missing truth and abstention reported explicitly."""

import numpy as np
from sklearn.metrics import roc_auc_score

from .features import read_features
from .io import DISEASES, case_index, file_hash, new_output, read_artifact, save
from .labels import read_labels


def metrics(pairs):
    known = [(truth, pred) for truth, pred in pairs if truth is not None]
    decided = [(t, p) for t, p in known if p.get("status") in {"positive", "negative"}]
    tp = sum(t == 1 and p["status"] == "positive" for t, p in decided)
    tn = sum(t == 0 and p["status"] == "negative" for t, p in decided)
    fp = sum(t == 0 and p["status"] == "positive" for t, p in decided)
    fn = sum(t == 1 and p["status"] == "negative" for t, p in decided)
    scores = [(t, p["probability"]) for t, p in decided if p.get("probability") is not None]

    def ratio(a, b):
        return a / b if b else None

    return {
        "known": len(known),
        "unknown_truth": len(pairs) - len(known),
        "decided": len(decided),
        "abstained": len(known) - len(decided),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "coverage": ratio(len(decided), len(known)),
        "accuracy": ratio(tp + tn, len(decided)),
        "correct_fraction_all_known": ratio(tp + tn, len(known)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "auroc": float(roc_auc_score([t for t, _ in scores], [p for _, p in scores]))
        if {t for t, _ in scores} == {0, 1}
        else None,
    }


def evaluate_diagnosis(features, labels, predictions, output, split="test"):
    fpath, data = read_features(features)
    lpath, truth = read_labels(labels)
    ppath, predicted = read_artifact(predictions, "diagnosis.json", "chd-diagnosis-predictions-v1")
    if predicted["features_sha256"] != file_hash(fpath):
        raise ValueError("诊断结果不属于此特征文件")
    rows = [r for r in data["cases"] if r["split"] == split]
    if not rows:
        raise ValueError("所选评估集合为空")
    pindex, lindex = case_index(predicted["cases"], splits=True), case_index(truth["cases"])
    if split == "test" and (
        set(r["patient_id"] for r in rows) & set(predicted["development_patients"])
        or set(r["case_id"] for r in rows) & set(predicted["development_cases"])
    ):
        raise ValueError("测试病例/患者参与过诊断模型开发，拒绝泄漏评估")
    for row in rows:
        case = row["case_id"]
        if case not in pindex or case not in lindex:
            raise ValueError("所选集合缺少诊断预测或标签")
        if pindex[case]["patient_id"] != row["patient_id"] or pindex[case]["split"] != row["split"]:
            raise ValueError("诊断结果的患者/划分信息与特征不一致")
        for group in ("classifier", "rules"):
            for finding in pindex[case][group].values():
                if finding.get("status") not in {"positive", "negative", "indeterminate"}:
                    raise ValueError("非法诊断状态")
                prob = finding.get("probability")
                if prob is not None and (not np.isfinite(prob) or not 0 <= prob <= 1):
                    raise ValueError("非法诊断概率")
    report = {
        "format": "chd-diagnosis-metrics-v1",
        "status": "passed",
        "split": split,
        "cases_evaluated": len(rows),
        "blank_policy": truth["blank_policy"],
        "training_blank_policy": predicted.get("classifier_blank_policy"),
        "labels_sha256": file_hash(lpath),
        "predictions_sha256": file_hash(ppath),
        "research_only": True,
        "segmentation_mode": predicted["segmentation_mode"],
        "metric_note": "accuracy/sensitivity/specificity exclude abstentions; coverage and correct_fraction_all_known expose them. One patient-disease pair is one observation. No paper top-two convention.",
        "assumed_negative_cells": sum(len(lindex[r["case_id"]].get("assumed_negative", [])) for r in rows),
    }
    rule_diseases = set().union(*(set(pindex[r["case_id"]]["rules"]) for r in rows))
    report["rules_alignment"] = {
        "policy": "exact native-code match only; no unverified clinical aliases",
        "matched_diseases": sorted(rule_diseases & set(DISEASES)),
        "rule_diseases_without_matching_truth": sorted(rule_diseases - set(DISEASES)),
        "label_diseases_without_rule": sorted(set(DISEASES) - rule_diseases),
    }
    report["software_only"] = predicted.get("software_only", False)
    for group in ("classifier", "rules"):
        report[group], all_pairs = {}, []
        for disease in DISEASES:
            pairs = [
                (
                    lindex[r["case_id"]]["labels"][disease],
                    pindex[r["case_id"]][group].get(disease, {"status": "indeterminate"}),
                )
                for r in rows
            ]
            report[group][disease] = metrics(pairs)
            all_pairs.extend(pairs)
        report[group + "_micro"] = metrics(all_pairs)
    output = new_output(output)
    save(output / "diagnosis-metrics.json", report)
    return report
