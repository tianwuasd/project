import numpy as np
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape


def restore(probabilities, entry, plan):
    props = entry["properties"]
    transform = plan["transpose_forward"]
    original_spacing = np.asarray(entry["spacing_xyz"])[::-1][transform]
    resized = resample_data_or_seg_to_shape(
        probabilities, props["shape_after_cropping_and_before_resampling"],
        plan["configurations"]["3d_fullres"]["spacing"], original_spacing,
        is_seg=False, order=1, order_z=0, force_separate_z=None)
    cropped = resized.argmax(0).astype(np.uint8)
    expanded = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
    box = tuple(slice(int(begin), int(end)) for begin, end in props["bbox_used_for_cropping"])
    if expanded[box].shape != cropped.shape:
        raise ValueError("预测恢复的裁剪框与尺寸不一致")
    expanded[box] = cropped
    xyz = expanded.transpose(plan["transpose_backward"]).transpose(2, 1, 0)
    native = np.zeros(entry["source_shape_xyz"], dtype=np.uint8)
    roi = tuple(slice(begin, end) for begin, end in entry["roi_xyz"])
    if native[roi].shape != xyz.shape:
        raise ValueError("预测恢复的原图子区域尺寸不一致")
    native[roi] = xyz
    return native
