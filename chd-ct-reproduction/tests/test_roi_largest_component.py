"""ROI geometry and method boundaries; no real training or CUDA required."""

import copy

import numpy as np
import pytest

from chd_ct.imagechd import methods, roi_inputs


def roi(crops, margin_fraction=0):
    return roi_inputs.roi_from_crops(crops, margin_fraction)


def pairs():
    a = np.zeros((20, 20, 20), np.uint8)
    a[7:12, 8:13, 6:11] = 1
    b = a.copy()
    a[0, 0, 0] = 7
    b[-1, -1, -1] = 6
    return a, b


def test_remote_islands_do_not_enlarge_box_and_labels_are_unchanged():
    a, b = pairs()
    originals = a.copy(), b.copy()
    box, info = roi([a, b], 0.15)
    assert box == (slice(6, 13), slice(7, 14), slice(5, 12))
    assert info['raw_roi_slices'] == [[0, 20]] * 3
    assert info['roi_slices'] == [[6, 13], [7, 14], [5, 12]]
    assert info['roi_components']['crop64']['removed_voxels'] == 1
    assert all(np.array_equal(x, y) for x, y in zip((a, b), originals))


def test_each_branch_is_cleaned_before_union():
    a = np.zeros((30, 30, 30), np.uint8)
    b = a.copy()
    a[3:6, 3:6, 3:6] = 1
    b[3:6, 3:6, 3:6] = 2
    a[20:22, 20:23, 20:24] = 7  # 24 voxels each, 36 in union > true 27.
    b[21:23, 20:23, 20:24] = 7
    box, _ = roi([a, b])
    assert box == (slice(3, 6),) * 3


def test_diagonal_multiclass_structure_is_one_component():
    a = np.zeros((8, 8, 8), np.uint8)
    a[2, 2, 2], a[3, 3, 3], a[4, 4, 4] = 1, 5, 7
    a[0, 7, 0] = 2
    box, info = roi([a, np.zeros_like(a)])
    assert box == (slice(2, 5),) * 3
    assert info['roi_components']['crop64']['kept_voxels'] == 3


def test_ties_keep_one_deterministic_component():
    a = np.zeros((8, 8, 8), np.uint8)
    a[1, 1, 1] = a[6, 6, 6] = 1
    box, info = roi([a, a])
    assert box == (slice(1, 2),) * 3
    assert info['roi_components']['crop64']['largest_tie_count'] == 2


@pytest.mark.parametrize('empty_index', [0, 1])
def test_single_empty_branch_uses_nonempty_branch(empty_index):
    a, _ = pairs()
    crops = [a, a.copy()]
    crops[empty_index] = np.zeros_like(a)
    box, info = roi(crops)
    assert box == (slice(7, 12), slice(8, 13), slice(6, 11))
    assert not info['empty_roi_fallback']


def test_both_empty_fallback_is_explicit():
    a = np.zeros((4, 5, 6), np.uint8)
    box, info = roi([a, a])
    assert box == (slice(0, 4), slice(0, 5), slice(0, 6))
    assert info['empty_roi_fallback'] and info['full_volume_roi']
    assert info['roi_warning']


def test_large_real_component_and_padding_clip_to_image():
    a = np.ones((4, 5, 6), np.uint8)
    box, info = roi([a, a], 0.15)
    assert box == (slice(0, 4), slice(0, 5), slice(0, 6))
    assert not info['empty_roi_fallback'] and info['full_volume_roi']


@pytest.mark.parametrize('margin', [-0.1, 1.5, True, float('nan'), float('inf')])
def test_invalid_margins_fail_clearly(margin):
    with pytest.raises(ValueError, match='边距'):
        roi(pairs(), margin)


def test_old_processing_is_rejected_and_current_method_is_shared():
    old = copy.deepcopy(methods.PROCESSING)
    old['roi'] = 'fold-localizer-predicted-roi-v1'
    with pytest.raises(ValueError, match='ROI'):
        methods.require_current_processing({'processing_method': old})
    assert methods.PROCESSING['roi'] == roi_inputs.ROI_METHOD


def test_locate_returns_original_multiclass_predictions(monkeypatch):
    from chd_ct.imagechd import predict

    a, b = pairs()
    monkeypatch.setattr(predict, 'anatomy', lambda image, stage, *args: a if stage == 'crop64' else b)
    crops, box, info = roi_inputs.locate(np.zeros(a.shape), None, {'config': {'roi_margin_fraction': 0.15}}, 'cpu')
    assert crops[0][0, 0, 0] == 7 and crops[1][-1, -1, -1] == 6
    assert box == (slice(6, 13), slice(7, 14), slice(5, 12))
    assert info['roi_method'] == roi_inputs.ROI_METHOD


def test_relative_margin_uses_each_axis_and_rounds_up():
    a = np.zeros((40, 50, 60), np.uint8)
    a[10:20, 12:32, 15:45] = 1
    box, info = roi([a, a], 0.15)
    assert box == (slice(8, 22), slice(9, 35), slice(10, 50))
    assert info['roi_margin_voxels'] == [2, 3, 5]
    assert info['roi_margin_fraction'] == 0.15


def test_relative_padding_is_applied_after_union():
    a = np.zeros((30, 30, 30), np.uint8)
    b = a.copy()
    a[5:10, 10:15, 10:15] = 1
    b[15:20, 10:15, 10:15] = 2
    box, _ = roi([a, b], 0.15)
    assert box == (slice(2, 23), slice(9, 16), slice(9, 16))


@pytest.mark.parametrize('fraction', [-0.1, True, float('nan'), 1.5])
def test_config_rejects_invalid_relative_margin(tmp_path, fraction):
    from pathlib import Path

    import yaml

    from chd_ct.imagechd.config import load_config
    config = yaml.safe_load(Path('configs/chd-smoke.yaml').read_text(encoding='utf-8'))
    config['roi_margin_fraction'] = fraction
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    with pytest.raises(ValueError, match='边距'):
        load_config(path)
