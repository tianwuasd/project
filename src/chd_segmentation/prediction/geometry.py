import numpy as np
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape


def restore_source(probabilities, geometry):
    plan = geometry.get('source_plan')
    if geometry.get('medsam_native') and plan is None:
        xyz = probabilities.argmax(0).astype(np.uint8).transpose(2, 1, 0)
    else:
        if plan is None:
            raise ValueError('源空间恢复缺少实际来源source_plan')
        props = geometry['properties']
        forward = plan['transpose_forward']
        original_spacing = np.asarray(geometry['spacing_xyz'])[::-1][forward]
        processed_spacing = geometry.get('processed_spacing', plan['configurations']['3d_fullres']['spacing'])
        resized = resample_data_or_seg_to_shape(probabilities,
            props['shape_after_cropping_and_before_resampling'], processed_spacing, original_spacing,
            is_seg=False, order=1, order_z=0, force_separate_z=None)
        cropped = resized.argmax(0).astype(np.uint8)
        expanded = np.zeros(props['shape_before_cropping'], dtype=np.uint8)
        box = tuple(slice(int(low), int(high)) for low, high in props['bbox_used_for_cropping'])
        if expanded[box].shape != cropped.shape:
            raise ValueError('实际来源的裁剪框与恢复尺寸不一致')
        expanded[box] = cropped
        xyz = expanded.transpose(plan['transpose_backward']).transpose(2, 1, 0)
    result = np.zeros(geometry['source_shape_xyz'], dtype=np.uint8)
    region = tuple(slice(int(low), int(high)) for low, high in geometry['roi_xyz'])
    if result[region].shape != xyz.shape:
        raise ValueError('实际来源ROI与原图恢复尺寸不一致')
    result[region] = xyz
    return result


def map_prompts(prompts, geometry, input_space):
    if input_space in ('native_dhw', 'input_dhw'):
        return prompts
    if input_space != 'source_xyz':
        raise ValueError('提示空间必须为source_xyz或input_dhw')
    plan = geometry.get('source_plan')
    native = geometry.get('medsam_native') and plan is None
    forward = [0, 1, 2] if native else plan['transpose_forward']
    roi_start = np.asarray([pair[0] for pair in geometry['roi_xyz']])
    crop = np.zeros(3) if native else np.asarray([pair[0] for pair in geometry['properties']['bbox_used_for_cropping']])
    before = np.asarray(geometry['image'].shape[1:]) if native else np.asarray(
        geometry['properties']['shape_after_cropping_and_before_resampling'])
    scale = np.asarray(geometry['image'].shape[1:]) / before
    output = []
    for item in prompts:
        x0, y0, x1, y1 = item['box_xyxy']
        corners = np.asarray([[x, y, item['frame']] for x in (x0, x1) for y in (y0, y1)])
        points = (corners-roi_start)[:, ::-1][:, forward]
        points = (points-crop) * scale
        # 当前框提示用输入切片上的轴平行矩形；切片轴改变时无法由单框表达。
        if not np.allclose(points[:, 0], points[0, 0]):
            raise ValueError('源提示切片经轴变换后跨多个输入帧，请提供input_dhw提示')
        low, high = points.min(0), points.max(0)
        output.append(dict(item, frame=int(round(low[0])), space='native_dhw',
                           box_xyxy=[float(low[2]), float(low[1]), float(high[2]), float(high[1])]))
    return output
