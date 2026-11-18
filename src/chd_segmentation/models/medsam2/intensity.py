from math import isfinite


def resolve_intensity(config, source):
    semantics = config.get('intensity_semantics', source.get('semantics'))
    mode = config.get('intensity_mode', 'auto')
    if mode == 'auto':
        if semantics == 'windowed_uint8':
            mode = 'windowed_uint8'
        elif semantics in ('hu', 'clipped_hu', 'zscore_clipped_hu'):
            mode = 'source_window'
        else:
            raise ValueError('MedSAM2缺少明确input_source.intensity_semantics；可显式选择normalized_percentile实验')
    if mode not in ('source_window', 'windowed_uint8', 'normalized_percentile'):
        raise ValueError('MedSAM2强度模式无效：'+str(mode))
    result = dict(mode=mode, semantics=semantics, experimental=mode == 'normalized_percentile')
    for key in ('mean', 'std', 'clip_range', 'gray_window', 'windowed_range'):
        result[key] = config.get(key, source.get(key))
    if mode == 'source_window':
        if semantics not in ('hu', 'clipped_hu', 'zscore_clipped_hu'):
            raise ValueError('source_window需要显式HU或已裁剪HU语义')
        window = result['gray_window']
        if not isinstance(window, (list, tuple)) or len(window) != 2 or not all(isfinite(v) for v in window) or window[0] >= window[1]:
            raise ValueError('MedSAM2缺少递增input_source.gray_window')
        if semantics == 'zscore_clipped_hu' and (
                result['mean'] is None or result['std'] is None or not isfinite(result['mean']) or
                not isfinite(result['std']) or result['std'] <= 0):
            raise ValueError('MedSAM2撤销来源线性标准化需要mean/std；截断信息无法恢复')
    if mode == 'normalized_percentile':
        values = config.get('percentiles', [.5, 99.5])
        if len(values) != 2 or not all(isfinite(v) for v in values) or not 0 <= values[0] < values[1] <= 100:
            raise ValueError('MedSAM2百分位实验参数无效')
        result['percentiles'] = list(values)
    return result


def convert_image(image, policy):
    import numpy as np
    from time import perf_counter
    began = perf_counter()
    if policy['mode'] == 'windowed_uint8':
        if image.dtype != np.uint8:
            raise ValueError('windowed_uint8需要实际uint8图像')
        return image, dict(policy, seconds=perf_counter()-began)
    values = np.asarray(image, dtype=np.float32)
    if policy['mode'] == 'normalized_percentile':
        low, high = np.percentile(values, policy['percentiles'])
        selected = np.clip(values, low, high)
    else:
        if policy['semantics'] == 'zscore_clipped_hu':
            values = values * policy['std'] + policy['mean']
        selected = np.clip(values, *policy['gray_window'])
        low, high = float(selected.min()), float(selected.max())
    if not np.isfinite([low, high]).all():
        raise ValueError('MedSAM2强度转换遇到非有限灰度')
    result = np.zeros(values.shape, dtype=np.uint8) if high <= low else (
        (selected-low)/(high-low)*255.).astype(np.uint8)
    return result, dict(policy, effective_range=[float(low), float(high)], seconds=perf_counter()-began)
