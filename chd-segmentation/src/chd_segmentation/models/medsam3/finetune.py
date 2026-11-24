from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time

import yaml

from ...acceleration import ResourceSession
from ...artifacts.store import digest, file_digest, operation_lock, read_json, require_output
from .config import load_config
from .environment import inspect_environment
from .runtime import request
from .weights import dependencies


def settings(path, prepared=None):
    path = Path(path).resolve(strict=True)
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    defaults = dict(prepared=None, classes=list(range(1, 11)), workers=1, prefetch=2,
                    updates=1000, validate_every=20, learning_rate=.00005, seed=42,
                    warmup_steps=100, max_seconds=14400, ram_limit_bytes=16*2**30,
                    vram_limit_bytes=16*2**30, batch=1)
    required = {'base_config', 'run_dir', 'train_cases', 'val_cases'}
    if not isinstance(raw, dict) or not required <= raw.keys() or set(raw)-required-defaults.keys():
        raise ValueError('医学LoRA训练配置字段无效')
    training = {**defaults, **raw}
    for name in ('updates', 'validate_every', 'prefetch', 'max_seconds', 'ram_limit_bytes', 'vram_limit_bytes', 'batch'):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError('训练预算必须为正整数：'+name)
    if type(training['workers']) is not int or training['workers'] < 0 or training['prefetch'] < training['workers']:
        raise ValueError('CPU读取线程与预取深度无效')
    if (type(training['warmup_steps']) is not int or not 0 <= training['warmup_steps'] < training['updates'] or
            type(training['seed']) is not int or not 0 <= training['seed'] < 2**32 or
            not 0 < float(training['learning_rate']) < 1):
        raise ValueError('学习率、warmup或随机种子无效')
    training['learning_rate'] = float(training['learning_rate'])
    if (not training['classes'] or any(type(k) is not int or k not in range(1, 11) for k in training['classes']) or
            len(set(training['classes'])) != len(training['classes'])):
        raise ValueError('需要不重复的CHD1282已登记类别')
    if any(not isinstance(training[k], list) or not training[k] for k in ('train_cases', 'val_cases')):
        raise ValueError('需要显式训练和验证病例名单')
    config = load_config(path.parent/training['base_config'])
    from ...config import validate_resources
    validate_resources(config['resources'],training['batch'])
    if config['device'] != 'cuda':
        raise ValueError('官方MedSAM3训练损失使用Triton，需要显式CUDA设备')
    world = config['resources']['gpu_count']
    if world*(training['workers']+1) > config['resources']['cpu_threads']:
        raise ValueError('CPU预算需同时覆盖读取线程和主线程')
    if (training['ram_limit_bytes'] > 100_000_000_000 or
            world*(config['fixed_ram_bytes']+config['workspace_bytes']) > training['ram_limit_bytes']):
        raise ValueError('原始工作集与训练内存上限不相容')
    training['run_dir'] = str((path.parent/training['run_dir']).resolve())
    training['prepared'] = str(Path(prepared).resolve(strict=True)) if prepared else (
        str((path.parent/training['prepared']).resolve(strict=True)) if training['prepared'] else None)
    require_output(training['run_dir'], config['work_root'],
                   [config['data_root'], config['official_root'], Path(__file__).parents[3]])
    config['training'] = training
    return config


def check(config):
    keys = ('python', 'base_weights', 'lora_weights', 'official_config', 'bpe_path', 'manifest', 'cohort')
    missing = [key for key in keys if not Path(config[key]).is_file()]
    if missing:
        return dict(ready=False, missing=missing, model_loaded=False)
    from .training_data import entries
    sources = entries(config, config['training'])
    record = dependencies(config)
    environment = inspect_environment(config['python'])
    return dict(ready=True, model_loaded=False, sources=sources, dependencies=record, environment=environment)


def training_contract(config,checked):
    implementation = {name:file_digest(Path(__file__).with_name(name+'.py')) for name in
        ('adapter','config','input','training_data','training_worker','finetune','distributed','launcher')}
    implementation['gradient_sync'] = file_digest(Path(__file__).parents[2]/'acceleration'/'distributed.py')
    return dict(sources=checked['sources'],dependencies=checked['dependencies'],implementation=implementation,
        training={k:v for k,v in config['training'].items() if k not in ('run_dir','prepared','max_seconds')},
        science={k:config[k] for k in ('precision','gray_window','intensity_semantics','source_mean','source_std',
            'prompts','geometry_policy','spacing_policy','orientation_evidence','intensity_evidence')})


def run(config, operation, resume=False, stop_after=None, checkpoint='best', deadline=None):
    from .training_worker import execute
    from .launcher import launch
    from .distributed import topology
    if operation not in ('train', 'validate') or checkpoint not in ('best', 'latest'):
        raise ValueError('医学LoRA操作或检查点无效')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('本次新增更新次数必须为正整数')
    checked = check(config)
    if not checked['ready']:
        raise FileNotFoundError('医学LoRA训练材料未齐备：'+str(checked['missing']))
    training = config['training']
    cutoff = time.time()+training['max_seconds']
    if deadline is not None:
        if not time.time() < deadline < float('inf'):
            raise ValueError('固定截止必须为未来有限Unix时间戳')
        cutoff = min(cutoff, deadline)
    root = Path(training['run_dir'])
    if operation == 'train' and not resume and root.exists():
        raise FileExistsError('新训练目录已存在；续训需显式--resume')
    if operation == 'validate' or resume:
        (root/(('latest' if resume else checkpoint)+'.pt')).resolve(strict=True)
    signature = digest(training_contract(config,checked))
    resources_config = deepcopy(config)
    remaining = cutoff-time.time()
    resources_config['resources']['max_wait'] = min(remaining, config['resources']['max_wait']) if config['resources']['max_wait'] is not None else remaining
    req = replace(request(resources_config, root), stage='medsam3_lora_'+operation)
    payload = dict(config=config, topology=topology(config), sources=checked['sources'], environment=checked['environment'],
        dependencies=checked['dependencies'], identity=signature, operation=operation, resume=resume,
        stop_after=stop_after, checkpoint=checkpoint, deadline=cutoff)
    with operation_lock(root), ResourceSession(req) as session:
        launch(config,checked['environment'],session,execute,(payload,session.handle,session.allocation.gpus),
            payload['topology']['world'],cutoff)
    return read_json(root/('validation.json' if operation == 'validate' else 'training.json'))
