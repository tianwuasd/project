"""Synthetic diagnosis-only integration check; never measures clinical accuracy."""

import csv

import numpy as np

from .classifier import train_classifier
from .evaluate import evaluate_diagnosis
from .features import FEATURE_NAMES, FEATURE_SCHEMA, measure
from .io import new_output, save
from .labels import import_labels
from .predict import diagnose


def run_demo(output):
    output = new_output(output)
    features = {
        "format": "chd-diagnosis-features-v1",
        "status": "passed",
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "prepared_sha256": "synthetic-only",
        "segmentation_sha256": "synthetic-only",
        "segmentation_mode": "synthetic",
        "cases": [],
    }
    truth = []
    for i in range(20):
        mask = np.zeros((24, 24, 24), np.uint8)
        for label in range(1, 8):
            mask[2 * label : 2 * label + 2, 3 : 8 + (4 if label == 1 and i % 2 else 0), 3:9] = label
        case = f"synthetic_{i}"
        features["cases"].append(
            {
                "case_id": case,
                "patient_id": case,
                "split": "train" if i < 12 else "val" if i < 16 else "test",
                **measure(mask),
            }
        )
        truth.append({"case_id": case, "ASD": i % 2, "VSD": 1 - i % 2})
    save(output / "features.json", features)
    with (output / "truth.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "ASD", "VSD"])
        writer.writeheader()
        writer.writerows(truth)
    import_labels(output / "truth.csv", output / "labels")
    trained = train_classifier(
        output / "features.json", output / "labels", output / "training", allow_smoke=True
    )
    diagnose(
        output / "features.json", output / "prediction", classifier=output / "training", allow_smoke=True
    )
    evaluate_diagnosis(
        output / "features.json", output / "labels", output / "prediction", output / "evaluation"
    )
    result = {
        "status": "passed",
        "synthetic_only": True,
        "trained_diseases": trained["trained_diseases"],
        "note": "Diagnosis software check, not clinical performance.",
    }
    save(output / "demo-report.json", result)
    return result
