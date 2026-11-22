from pathlib import Path
import math

import yaml

from ...config import DEFAULTS, _merge, validate_resources


def load_config(path):
    path = Path(path).resolve(strict=True)
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    required = {'model', 'weights', 'dependencies', 'manifest', 'cohort', 'data_root',
                'work_root', 'registry_root', 'orientation_evidence', 'intensity_evidence'}
    defaults = dict(prompt='aorta', gray_window=[-160., 240.], intensity_semantics='hu',
                    source_mean=None, source_std=None, resources={},
                    estimated_ram_bytes=7 * 2**29, estimated_vram_bytes=5 * 2**30,
                    ram_limit_bytes=8 * 2**30, vram_limit_bytes=7 * 2**30,
                    worker_timeout_seconds=600)
    if not isinstance(raw, dict) or not required <= raw.keys() or set(raw) - required - defaults.keys():
        raise ValueError('SAM3配置缺少必要字段或包含未知字段')
    config = {**defaults, **raw}
    if config['model'] != 'sam3' or not isinstance(config['prompt'], str) or not config['prompt'].strip():
        raise ValueError('需要SAM3原模型及明确文本提示')
    for key in ('orientation_evidence', 'intensity_evidence'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError('需要显式几何及强度证据')
    if config['intensity_semantics'] not in ('hu', 'windowed_uint8', 'zscore_clipped_hu'):
        raise ValueError('不支持的输入强度语义')
    if config['intensity_semantics'] == 'zscore_clipped_hu' and (
            any(type(config[k]) not in (int, float) or not math.isfinite(config[k])
                for k in ('source_mean', 'source_std')) or config['source_std'] <= 0):
        raise ValueError('撤销标准化需要明确mean和正std')
    window = config['gray_window']
    if len(window) != 2 or not all(math.isfinite(v) for v in window) or window[0] >= window[1]:
        raise ValueError('灰度窗需要递增的有限上下界')
    for key in ('estimated_ram_bytes', 'estimated_vram_bytes', 'ram_limit_bytes',
                'vram_limit_bytes', 'worker_timeout_seconds'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('预算必须为正整数：' + key)
    if (config['estimated_ram_bytes'] > config['ram_limit_bytes'] > 0 or
            config['ram_limit_bytes'] > 8 * 2**30 or config['vram_limit_bytes'] > 7 * 2**30 or
            config['estimated_vram_bytes'] > config['vram_limit_bytes'] or
            config['worker_timeout_seconds'] > 600):
        raise ValueError('配置超过有限切片检查预算')
    for key in ('weights', 'dependencies', 'manifest', 'cohort', 'data_root', 'work_root', 'registry_root'):
        config[key] = str((path.parent / config[key]).resolve())
    resources = _merge(DEFAULTS['resources'], config['resources'])
    resources.update(device='cuda', gpu_count=1, cpu_processes=1, workers=0,
                     min_gpus=None, batch_layout=[1], precision='bf16', memory_policy='host_fraction')
    if (resources['ram_fraction'] > .8 or resources['memory_limit_bytes'] > 100_000_000_000 or
            resources['wait_seconds'] is None or not 0 < resources['wait_seconds'] <= 180 or
            resources['vram_reserve_bytes'] < 2**30):
        raise ValueError('资源配置超过本次许可范围')
    validate_resources(resources, 1)
    resources['max_wait'] = resources['wait_seconds']
    config['resources'] = resources
    return config
