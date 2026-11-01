"""Explicit denominators; no absent-class Dice inflation or concealed abstention."""

import numpy as np

from .data import validate_labels
from .labels import DISEASES


def segmentation_metrics(prediction, target, classes=11):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    validate_labels(prediction, classes)
    validate_labels(target, classes)
    dice = {}
    for c in range(classes):
        p, t = prediction == c, target == c
        denominator = int(p.sum() + t.sum())
        dice[str(c)] = 2 * int((p & t).sum()) / denominator if denominator else None
    present = [v for k, v in dice.items() if k != "0" and v is not None]
    return {
        "dice": dice,
        "mean_foreground_dice": float(np.mean(present)) if present else None,
        "foreground_classes_in_mean": len(present),
    }


def multilabel_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (
        truth.shape != prediction.shape
        or truth.ndim != 2
        or truth.shape[1] != len(DISEASES)
        or not len(truth)
    ):
        raise ValueError("Expected nonempty matching [patients,17] matrices")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [-1, 0, 1]).all():
        raise ValueError("Truth must be 0/1; predictions -1 (unknown), 0 or 1")
    decided = prediction >= 0
    correct = (truth == prediction) & decided
    per_class = {}
    for i, name in enumerate(DISEASES):
        d, t, p = decided[:, i], truth[:, i], prediction[:, i]
        tp, tn = int(((t == 1) & (p == 1)).sum()), int(((t == 0) & (p == 0)).sum())
        positives, negatives = int(((t == 1) & d).sum()), int(((t == 0) & d).sum())
        per_class[name] = {
            "sensitivity_on_decided": tp / positives if positives else None,
            "specificity_on_decided": tn / negatives if negatives else None,
            "decided": int(d.sum()),
            "unknown": int((~d).sum()),
        }
    return {
        "coverage": float(decided.mean()),
        "micro_accuracy_on_decided": float(correct.sum() / decided.sum()) if decided.any() else None,
        "correct_fraction_all_slots": float(correct.mean()),
        "exact_match_including_unknown_as_incorrect": float(correct.all(axis=1).mean()),
        "per_class": per_class,
        "note": "These metrics do not reproduce the paper's nine-group clinical aggregation.",
    }
