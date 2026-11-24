from pathlib import Path
import math

import yaml

from ...config import DEFAULTS, _merge, validate_resources


PROMPTS = {
    1: 'left ventricle', 2: 'right ventricle', 3: 'left atrium', 4: 'right atrium',
    5: 'aorta', 6: 'pulmonary artery', 7: 'myocardium', 8: 'superior vena cava',
    9: 'inferior vena cava', 10: 'pulmonary veins',
}


def load_config(path):
    path = Path(path).resolve(strict=True)
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    required = {
        'model', 'manifest', 'data_root', 'cohort', 'work_root', 'registry_root', 'python',
        'official_root', 'base_weights', 'lora_weights', 'official_config', 'bpe_path',
        'orientation_evidence', 'spacing_policy', 'geometry_policy', 'intensity_evidence',
    }
    defaults = dict(
        device='cuda', precision='fp32', resolution=1008, gray_window=[-160., 240.],
        intensity_semantics='hu', source_mean=None, source_std=None, prompts=PROMPTS,
        detection_threshold=.5, nms_iou_threshold=.5, coverage='all', slices=None,
        fixed_ram_bytes=8 * 2**30, workspace_bytes=2**30, estimated_vram_bytes=8 * 2**30,
        resources={}, heartbeat_seconds=15, worker_timeout_seconds=900, capacity_probe_bytes=0)
    if not isinstance(raw, dict) or not required <= raw.keys() or set(raw) - required - defaults.keys():
        raise ValueError('MedSAM3配置缺少必需字段或包含未知字段')
    config = {**defaults, **raw}
    if type(config['capacity_probe_bytes']) is not int or config['capacity_probe_bytes'] < 0:
        raise ValueError('首次容量检查显存上限需要非负整数')
    if config['model'] != 'medsam3' or config['resolution'] != 1008:
        raise ValueError('MedSAM3固定使用所选官方1008图像配方')
    if config['device'] not in ('cpu','cuda') or config['precision'] not in ('fp32','bf16'):
        raise ValueError('设备和精度必须显式支持，禁止自动替换')
    if config['device'] == 'cpu' and config['precision'] != 'fp32':
        raise ValueError('显式CPU配方使用fp32')
    if config['geometry_policy'] != 'shared_index_excel' or config['spacing_policy'] != 'author_excel':
        raise ValueError('CHD1282需要明确共享数组索引及作者间距合同')
    if any(not isinstance(config[k], str) or not config[k].strip()
           for k in ('orientation_evidence', 'intensity_evidence')):
        raise ValueError('必须记录RAS研究约定及强度假设的证据')
    if config['intensity_semantics'] not in ('hu', 'windowed_uint8', 'zscore_clipped_hu'):
        raise ValueError('MedSAM3输入强度语义无效')
    if config['intensity_semantics'] == 'zscore_clipped_hu' and (
            any(type(config[k]) not in (int, float) or not math.isfinite(config[k])
                for k in ('source_mean', 'source_std')) or config['source_std'] <= 0):
        raise ValueError('撤销标准化需要明确有限mean及正std')
    window = config['gray_window']
    if len(window) != 2 or not all(math.isfinite(v) for v in window) or window[0] >= window[1]:
        raise ValueError('CT窗需要两个递增有限数值')
    if config['prompts'] != PROMPTS:
        raise ValueError('本配方固定CHD1282十类医学文本，不读取标签生成提示')
    for key in ('detection_threshold', 'nms_iou_threshold'):
        if not 0 < float(config[key]) < 1:
            raise ValueError('检测及NMS阈值必须位于0与1之间')
    if config['coverage'] not in ('all', 'selected'):
        raise ValueError('coverage必须是all或selected')
    if config['coverage'] == 'selected' and (not isinstance(config['slices'], dict) or not config['slices']):
        raise ValueError('部分覆盖需要逐例预先指定切片索引')
    if config['coverage'] == 'all' and config['slices'] is not None:
        raise ValueError('整例模式不能包含切片子集')
    for key in ('fixed_ram_bytes', 'workspace_bytes', 'estimated_vram_bytes',
                'heartbeat_seconds', 'worker_timeout_seconds'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('预算及超时需要正整数：' + key)
    path_fields = required - {
        'model', 'orientation_evidence', 'spacing_policy', 'geometry_policy', 'intensity_evidence'}
    for key in path_fields:
        config[key] = str((path.parent / Path(config[key])).resolve())
    resources = _merge(DEFAULTS['resources'], config['resources'])
    resources.update(device=config['device'], gpu_count=1, cpu_processes=1, min_gpus=None,
                     workers=0, batch_layout=[1], precision=config['precision'])
    if resources['ram_fraction'] > .8 or resources['memory_limit_bytes'] > 100_000_000_000:
        raise ValueError('MedSAM3不继承技术测试内存豁免')
    validate_resources(resources, 1)
    resources['max_wait'] = resources['wait_seconds']
    config['resources'] = resources
    return config
