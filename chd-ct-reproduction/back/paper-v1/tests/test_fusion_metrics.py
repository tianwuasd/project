import numpy as np
import pytest

from chd_ct.evaluation import multilabel_metrics, segmentation_metrics
from chd_ct.inference.fusion import refine_with_blood, weighted_vote


def test_initial_class_compatibility_does_not_become_pa():
    whole = np.full((4, 4, 4), 5, dtype=np.uint8)
    initial = np.full_like(whole, 6)
    pred = weighted_vote(
        [whole, whole, whole, initial, whole, initial], ["crop", "crop", "all", "init", "all", "init"]
    )
    assert np.all(pred == 5)


def test_region_growth_preserves_myocardium_and_boundary():
    seg = np.zeros((9, 9, 9), dtype=np.uint8)
    seg[1, 1, 1] = 1
    seg[7, 7, 7] = 7
    blood = np.zeros_like(seg)
    blood[1:4, 1:4, 1:4] = 1
    blood[4, :, :] = 2
    out = refine_with_blood(seg, blood)
    assert out[3, 3, 3] == 1
    assert out[7, 7, 7] == 7
    assert not out[4].any()


def test_undefined_metrics_are_not_perfect_scores():
    z = np.zeros((2, 2, 2), dtype=np.uint8)
    result = segmentation_metrics(z, z, classes=3)
    assert result["dice"]["1"] is None
    assert result["mean_foreground_dice"] is None
    diagnosis = multilabel_metrics(np.zeros((2, 17)), np.full((2, 17), -1))
    assert diagnosis["coverage"] == 0
    assert diagnosis["micro_accuracy_on_decided"] is None
    with pytest.raises(ValueError):
        multilabel_metrics(np.zeros((2, 17)), np.zeros((3, 17)))
