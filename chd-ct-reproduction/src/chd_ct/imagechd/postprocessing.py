"""Seed-preserving candidates and frozen validation-selected component policies."""

import numpy as np
from scipy import ndimage as ndi

from .geometry import BLOOD_IDS

POLICY_FORMAT = "imagechd-validation-postprocessing-v1"


def limited_growth(segmentation, blood, steps=2):
    if segmentation.shape != blood.shape or not np.isin(segmentation, range(8)).all():
        raise ValueError("有限补全输入无效")
    if not np.isin(blood, range(3)).all() or type(steps) is not int or not 0 <= steps <= 16:
        raise ValueError("有限补全血池或步数无效")
    result = segmentation.copy()
    structure = ndi.generate_binary_structure(3, 1)
    for _ in range(steps):
        available = (result == 0) & (blood == 1)
        proposals = np.zeros_like(result)
        counts = np.zeros(result.shape, np.uint8)
        for label in BLOOD_IDS:
            frontier = ndi.binary_dilation(result == label, structure=structure) & available
            proposals[frontier] = label
            counts[frontier] += 1
        changed = counts == 1
        if not changed.any():
            break
        result[changed] = proposals[changed]
    return result


def keep_largest(segmentation, label):
    result = segmentation.copy()
    components, count = ndi.label(segmentation == label, ndi.generate_binary_structure(3, 1))
    if count > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        result[(components != 0) & (components != sizes.argmax())] = 0
    return result


def selection_scores(prediction, target):
    if (
        prediction.shape != target.shape
        or not np.isin(prediction, range(8)).all()
        or not np.isin(target, [*range(8), 255]).all()
    ):
        raise ValueError("验证输入尺寸或标签无效")
    valid = target != 255
    if not valid.any():
        raise ValueError("验证病例无有效标签")
    scores, missing = [], []
    for label in range(1, 8):
        p, t = (prediction == label) & valid, (target == label) & valid
        denominator = int(p.sum() + t.sum())
        scores.append(2 * int((p & t).sum()) / denominator if denominator else 1.0)
        missing.append(bool(t.any() and not p.any()))
    return np.asarray(scores), np.asarray(missing, dtype=int)


def choose_policy(records, steps=2):
    if not records:
        raise ValueError("后处理选择需要非空验证集")

    def mean(name):
        return np.mean([r["scores"][name] for r in records], axis=0)

    def missing(name):
        return np.sum([r["missing"][name] for r in records], axis=0)

    raw, grown = mean("vote"), mean("limited")
    use_growth = bool(
        np.mean(grown) > np.mean(raw) + 1e-8
        and np.all(grown >= raw)
        and np.all(missing("limited") <= missing("vote"))
    )
    base = "limited" if use_growth else "vote"
    labels, trials = [], {}
    for label in range(1, 8):
        name = f"{base}_lcc_{label}"
        better = bool(
            mean(name)[label - 1] > mean(base)[label - 1] + 1e-8
            and missing(name)[label - 1] <= missing(base)[label - 1]
        )
        trials[str(label)] = {
            "accepted": better,
            "before": float(mean(base)[label - 1]),
            "after": float(mean(name)[label - 1]),
        }
        if better:
            labels.append(label)
    return {
        "format": POLICY_FORMAT,
        "refinement": base,
        "max_growth_steps": steps,
        "largest_components": labels,
        "calibrated": True,
        "selection_rule": "strict mean improvement; no class mean decrease; no extra empty organs",
        "empty_class_rule": "both absent=1, fixed validation denominator",
        "growth_accepted": use_growth,
        "component_trials": trials,
        "vote_class_means": raw.tolist(),
        "limited_class_means": grown.tolist(),
    }


def apply_policy(vote, limited, policy):
    if policy.get("format") != POLICY_FORMAT or policy.get("refinement") not in {"vote", "limited"}:
        raise ValueError("后处理策略无效")
    result = (limited if policy["refinement"] == "limited" else vote).copy()
    labels = policy.get("largest_components")
    if not isinstance(labels, list) or any(
        type(label) is not int or label not in range(1, 8) for label in labels
    ):
        raise ValueError("连通区域策略无效")
    for label in labels:
        result = keep_largest(result, label)
    return result


def baseline_policy(steps=2, reason="not_calibrated"):
    return {
        "format": POLICY_FORMAT,
        "refinement": "vote",
        "max_growth_steps": steps,
        "largest_components": [],
        "calibrated": False,
        "reason": reason,
    }
