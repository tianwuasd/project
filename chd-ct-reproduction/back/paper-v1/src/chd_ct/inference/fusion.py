import numpy as np
from scipy import ndimage as ndi

from ..data import validate_labels
from ..labels import BLOOD_IDS


def weighted_vote(predictions, kinds, weights=(1, 1, 2, 2, 2, 2)):
    """Initial-vessel labels vote compatibly for both AO/PA; ties use lowest class ID."""
    if len(predictions) != len(kinds) or len(kinds) != len(weights):
        raise ValueError("Prediction, kind and weight counts differ")
    shape = predictions[0].shape
    scores = np.zeros((11, *shape), dtype=np.float32)
    for pred, kind, weight in zip(predictions, kinds, weights):
        if pred.shape != shape or kind not in ("crop", "all", "init") or weight <= 0:
            raise ValueError("Invalid voting input")
        validate_labels(pred, 7 if kind == "init" else 11)
        if kind == "init":
            mapping = {0: (0,), 1: (1,), 2: (2,), 3: (3,), 4: (4,), 5: (7,), 6: (5, 6)}
        else:
            mapping = {i: (i,) for i in range(11)}
        for source, targets in mapping.items():
            for destination in targets:
                scores[destination] += weight * (pred == source)
    return scores.argmax(axis=0).astype(np.uint8)


def refine_with_blood(segmentation, blood):
    """Geodesic voxel growth inside blood interior; never crosses the predicted boundary.

    Small components are retained. The paper's domain-dependent reassignment is not specified.
    """
    if segmentation.shape != blood.shape:
        raise ValueError("Blood and anatomy grids differ")
    validate_labels(segmentation, 11)
    validate_labels(blood, 3)
    result = segmentation.copy()
    inside = blood == 1
    result[np.isin(result, BLOOD_IDS) & ~inside] = 0
    available = inside & (result == 0)
    structure = ndi.generate_binary_structure(3, 1)
    # Synchronous wavefront gives deterministic one-voxel geodesic expansion.
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
