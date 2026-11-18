from copy import deepcopy
import math
from pathlib import Path

from ..config import DEFAULTS, _merge, validate_resources


PREDICTION_MODELS = ('totalsegmentator', 'medsam3')
MEDSAM3_COMMIT = '84f43118b5b67b50abad81d7c946681ebb97dcda'


def options(raw, path, overrides=None):
    defaults = dict(model=None, run_dir=None, work_root=None, output=None, device='cuda',
        threads=2, execution={}, logging=dict(heartbeat_seconds=15, case_details=True),
        input_source=dict(path=None, intensity_semantics='hu'), backend={})
    if set(raw) - set(defaults):
        raise ValueError('预测模型配置字段无效：' + ','.join(sorted(set(raw)-set(defaults))))
    config = {**defaults, **deepcopy(raw)}
    for key, value in (overrides or {}).items():
        if key not in defaults:
            raise ValueError('此模型仅支持预测，不接受参数：' + key)
        config[key] = {**config[key], **value} if isinstance(value, dict) else value
    if config['model'] not in PREDICTION_MODELS:
        raise ValueError('未知独立预测模型')
    for key in ('run_dir', 'work_root', 'output'):
        if not config[key]:
            raise ValueError('必须指定' + key)
        config[key] = str((Path(path).resolve().parent / config[key]).resolve())
    if config['device'] not in ('cpu', 'cuda') or type(config['threads']) is not int or config['threads'] < 1:
        raise ValueError('设备必须为cpu/cuda，线程数必须为正整数')
    source = config['input_source']
    if set(source) - {'path', 'intensity_semantics'} or source.get('intensity_semantics', 'hu') != 'hu':
        raise ValueError('预测按manifest读取原始HU影像，不能把标准化数组声明为HU')
    if source.get('path'):
        source['path'] = str((Path(path).resolve().parent/source['path']).resolve())
    resources = _merge(DEFAULTS['resources'], config['execution'])
    specified = config['execution']
    if 'wait_seconds' in specified and 'max_wait' in specified and specified['wait_seconds']!=specified['max_wait']:
        raise ValueError('显式wait_seconds与max_wait不同')
    if 'wait_seconds' in specified and 'max_wait' not in specified:
        resources['max_wait'] = specified['wait_seconds']
    resources.update(device=config['device'], cpu_threads=config['threads'])
    if resources['gpu_count'] != 1 or resources['cpu_processes'] != 1:
        raise ValueError('独立预测模型每个任务使用单个工作进程和一张显卡')
    if resources['ram_fraction'] > .8 or resources['cpu_fraction'] > .8 or resources['memory_limit_bytes'] > 100_000_000_000:
        raise ValueError('资源上限不能超过整机80%及100GB')
    validate_resources(resources, 1)
    config['execution'] = resources
    logging = config['logging']
    if not isinstance(logging,dict) or set(logging)-{'heartbeat_seconds','case_details'}:
        raise ValueError('日志字段无效')
    config['logging'] = {**defaults['logging'],**logging}
    heartbeat = config['logging']['heartbeat_seconds']
    if type(heartbeat) not in (int,float) or not math.isfinite(heartbeat) or heartbeat<=0 or type(config['logging']['case_details']) is not bool:
        raise ValueError('日志心跳与病例细节设置无效')
    backend = config['backend']
    paths = ('weights_dir',) if config['model']=='totalsegmentator' else ('official_root','base_weights','lora_weights','tokenizer','lora_config')
    allowed = set(paths) | {'model_memory_bytes'}
    allowed |= {'task','version','fast'} if config['model']=='totalsegmentator' else {'source_commit','gray_window','slice_direction','class_prompts','detection_threshold','nms_iou_threshold'}
    if set(backend)-allowed or not set(paths) <= backend.keys():
        raise ValueError('官方后端字段无效或缺少本地资源路径')
    for key in paths:
        backend[key] = str((Path(path).resolve().parent/backend[key]).resolve())
    memory = backend.get('model_memory_bytes')
    if type(memory) is not int or memory < 1:
        raise ValueError('需要明确模型驻留内存估计model_memory_bytes')
    if config['model']=='totalsegmentator':
        if (backend.get('task'),backend.get('version'),backend.get('fast')) != ('heartchambers_highres','2.18.0',False):
            raise ValueError('Total固定2.18.0 heartchambers_highres fast=false')
    else:
        if backend.get('source_commit') != MEDSAM3_COMMIT or backend.get('slice_direction') != 'axial':
            raise ValueError('MedSAM3需要指定作者固定源码及axial切片')
        window = backend.get('gray_window')
        if not isinstance(window,list) or len(window)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in window) or window[0]>=window[1]:
            raise ValueError('CT窗必须为有限递增HU范围')
        for key in ('detection_threshold','nms_iou_threshold'):
            if type(backend.get(key)) not in (int,float) or not 0 <= backend[key] <= 1:
                raise ValueError('检测及NMS阈值必须在0至1之间')
        if not isinstance(backend.get('class_prompts'),dict) or not backend['class_prompts'] or any(not isinstance(v,str) or not v.strip() for v in backend['class_prompts'].values()):
            raise ValueError('需要非空解剖名称文本提示')
    return config
