"""Four anatomy votes and boundary-constrained blood refinement, using seven IDs."""

import numpy as np
from scipy import ndimage as ndi

from .geometry import BLOOD_IDS


def weighted_vote(predictions, weights=(1, 1, 2, 2)):
    if len(predictions) != len(weights) or not predictions:
        raise ValueError("投票数量不匹配")
    scores = np.zeros((8, *predictions[0].shape), np.float32)
    for prediction, weight in zip(predictions, weights):
        if prediction.shape != predictions[0].shape or not np.isin(prediction, range(8)).all() or weight <= 0:
            raise ValueError("投票标签/形状/权重无效")
        for label in range(8):
            scores[label] += weight * (prediction == label)
    return scores.argmax(0).astype(np.uint8)


def refine_with_blood(segmentation, blood):
    if (
        segmentation.shape != blood.shape
        or not np.isin(segmentation, range(8)).all()
        or not np.isin(blood, range(3)).all()
    ):
        raise ValueError("融合输入无效")
    result = segmentation.copy()
    inside = blood == 1
    result[np.isin(result, BLOOD_IDS) & ~inside] = 0
    available = inside & (result == 0)
    structure = ndi.generate_binary_structure(3, 1)
    while available.any():
        proposals = np.zeros_like(result)
        for label in BLOOD_IDS:
            frontier = (
                ndi.binary_dilation(result == label, structure=structure) & available & (proposals == 0)
            )
            proposals[frontier] = label
        changed = proposals > 0
        if not changed.any():
            break
        result[changed] = proposals[changed]
        available[changed] = False
    return result
