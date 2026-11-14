from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from time import perf_counter

from filelock import FileLock
import nibabel as nib
import numpy as np
import psutil
import torch
import yaml

from ..artifacts.store import digest, file_digest, write_json
from ..data.dataset import CaseDataset
from ..preprocessing.prepare import load_prepared
from ..prediction.restore import restore


def options(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    defaults = dict(model=None, run_dir=None, device='cpu', precision='fp32', threads=2,
                    seed=20261114, updates=2, accumulation=4, learning_rate=0.0002,
                    patch=128, local_size=128, frames=8, resolution=512,
                    pretrained=None, parents={}, groups=None, technical=False,
                    official_root=None, dependency_root=None, freeze_image_encoder=False,
                    max_candidates=24, warmup_updates=2000,method='M07',key_references=None,
                    validate_every=1000)
    if not isinstance(raw, dict) or set(raw)-set(defaults):
        raise ValueError('模型配置字段无效')
    config = {**defaults, **raw}
    if config['model'] not in ('medsam2', 'q_route1') or not config['run_dir']:
        raise ValueError('必须指定模型与独立运行目录')
    for key in ('updates', 'accumulation', 'threads', 'patch', 'local_size', 'frames', 'resolution', 'max_candidates'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('模型配置需要正整数：'+key)
    if config['precision'] not in ('fp32','bf16'):
        raise ValueError('模型精度必须是fp32或bf16')
    if config['technical'] and (config['updates'] > 2 or config['accumulation'] > 4):
        raise ValueError('技术验证限定每阶段最多两次更新、每次最多四个输入')
    if not config['technical'] and (config['patch'] != 128 or config['local_size'] != 128) and config['model']=='q_route1':
        raise ValueError('Q正式任务固定128读取尺寸')
    if config['model']=='medsam2' and config['resolution'] != 512:
        raise ValueError('MedSAM2当前配方固定官方512分辨率')
    if config['max_candidates'] > 24:
        raise ValueError('Q候选数量上限为24')
    if config['model']=='q_route1' and config['method'] not in ('M02','M07'):
        raise ValueError('Q统一训练入口支持M02同监督对照及M07完整求解')
    if config['model']=='q_route1' and not config['technical'] and config['accumulation']!=4:
        raise ValueError('Q正式训练每次更新需要四次梯度累积')
    if type(config['validate_every']) is not int or config['validate_every']<1:
        raise ValueError('验证间隔必须为正整数')
    if not np.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise ValueError('学习率必须为有限正数')
    root = Path(path).resolve().parent
    for key in ('run_dir','pretrained','groups','official_root','dependency_root','key_references'):
        if config[key] is not None:
            config[key] = str((root/Path(config[key])).resolve())
    config['parents'] = {key:str((root/Path(value)).resolve(strict=True)) for key,value in config['parents'].items()}
    return config


def resources(config):
    torch.set_num_threads(config['threads'])
    device = torch.device(config['device'])
    if device.type not in ('cpu','cuda'):
        raise ValueError('模型仅支持显式CPU或CUDA设备')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('请求的CUDA设备不可用')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    if config['precision']=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16需要支持该精度的CUDA设备')
    return device


def adapter(config, prepared, stage, initialize=True):
    if config['model']=='q_route1':
        from .q_route1.adapter import QAdapter
        return QAdapter(config,prepared,stage,initialize=initialize)
    from .medsam2.adapter import MedSAM2Adapter
    return MedSAM2Adapter(config,prepared,stage,initialize=initialize)


def source_identity(config):
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)):file_digest(path) for path in sorted(root.rglob('*.py'))}


def training_names(config, prepared, stage):
    if config['technical']:
        if prepared['scope']!='technical' or len(prepared['manifest']['cases'])>2:
            raise ValueError('技术验证必须使用最多两例的技术准备描述')
        return [row['case_id'] for row in prepared['manifest']['cases']]
    if prepared['scope']!='formal':
        raise ValueError('正式模型训练需要正式准备描述')
    if config['model']=='q_route1':
        if not config['groups']:
            raise ValueError('Q正式训练需要显式A/B/C/D分组')
        groups = yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))
        if set(groups)!=set('ABCD') or any(not isinstance(groups[key],list) or not groups[key] for key in groups):
            raise ValueError('Q分组必须包含非空A/B/C/D病例名单')
        rows = {row['case_id']:row for row in prepared['manifest']['cases']}
        seen = set()
        for role in 'ABCD':
            patients = {rows[name]['patient'] for name in groups[role]}
            if seen & patients or not all(patients):
                raise ValueError('Q患者分组重叠或缺少患者身份')
            seen |= patients
        fit_ids = set(prepared['plan'].get('fit_ids',[]))
        if not fit_ids or not fit_ids <= set(groups['A']):
            raise ValueError('Q统计拟合必须限于A组')
        return groups['B' if stage=='local' else 'A']
    names = [row['case_id'] for row in prepared['manifest']['cases'] if 'train' in row['roles']]
    if not names:
        raise ValueError('没有训练病例')
    return names


def dataset_for(prepared, with_label):
    return CaseDataset(prepared,dict(reserve_ram_bytes=4*2**30,cache_bytes=512*2**20),with_label=with_label)


def load_state(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def save_state(path, state):
    pending = path.with_suffix('.pending')
    with pending.open('wb') as stream:
        torch.save(state,stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending,path)


def model_identity(config,prepared,stage,initialization_sha256=None):
    scientific = {key:value for key,value in config.items() if key not in ('run_dir','threads','dependency_root','official_root')}
    return digest(dict(config=scientific,prepared=prepared['identity'],stage=stage,code=source_identity(config),
        parents={key:file_digest(value) for key,value in config['parents'].items()},
        pretrained=initialization_digest(config,initialization_sha256),
        groups=file_digest(config['groups']) if config['groups'] else None,
        key_references=file_digest(config['key_references']) if config['key_references'] else None))


def initialization_digest(config,previous=None):
    if config['pretrained'] is None:
        return None
    if Path(config['pretrained']).is_file():
        return file_digest(config['pretrained'])
    if previous is None:
        raise FileNotFoundError('初始化权重缺失且检查点未保存其摘要')
    return previous


def train(config, prepared_path, stage, resume=False, stop_after=None):
    started = perf_counter()
    device = resources(config)
    prepared = load_prepared(prepared_path)
    names = training_names(config,prepared,stage)
    root = Path(config['run_dir'])/stage
    if root.exists() and not resume:
        raise FileExistsError('阶段目录已存在，使用resume或新目录')
    if resume and not (root/'latest.pt').is_file():
        raise FileNotFoundError('阶段缺少完整检查点')
    if stop_after is not None and not 1 <= stop_after <= config['updates']:
        raise ValueError('stop-after超出预算')
    if prepared['manifest']['data_root'] == str(root) or Path(prepared['manifest']['data_root']) in root.parents:
        raise ValueError('训练输出不能位于原始数据目录内')
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    previous = load_state(root/'latest.pt') if resume else None
    initial_sha = initialization_digest(config,previous['initialization_sha256'] if previous else None)
    bundle = adapter(config,prepared,stage,initialize=not resume)
    model = bundle.model.to(device)
    model.train()
    optimizer = bundle.optimizer()
    identity = model_identity(config,prepared,stage,initial_sha)
    dataset = dataset_for(prepared,True)
    bundle.prepare_training(dataset,names)
    update,history,best = 0,[],None
    if resume:
        state = previous
        if state['identity']!=identity:
            raise ValueError('恢复的配置、数据、源码或上游身份不同')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        rng.bit_generator.state = state['data_rng']
        if device.type=='cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        bundle.restore_extra(state['extra'])
        update,history,best = state['update'],state['history'],state['best']
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(root/'stage.lock',timeout=0):
        while update < (config['updates'] if stop_after is None else stop_after):
            mark = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            bundle.schedule(optimizer,update)
            loss_sum = 0.
            for _ in range(config['accumulation']):
                entry = bundle.sample(dataset,names,rng)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                    loss = bundle.loss(entry,rng,update)
                if not torch.isfinite(loss):
                    raise FloatingPointError('模型训练损失非有限')
                (loss/config['accumulation']).backward()
                loss_sum += float(loss.detach())/config['accumulation']
                del loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            update += 1
            history.append(dict(update=update,loss=loss_sum,gradient_norm=float(norm),seconds=perf_counter()-mark))
            improved = False
            if not config['technical'] and (update%config['validate_every']==0 or update==config['updates']):
                validation_names = (yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))['C']
                    if config['model']=='q_route1' else [row['case_id'] for row in prepared['manifest']['cases'] if 'validation' in row['roles']])
                if not validation_names:
                    raise ValueError('正式选模需要独立验证病例')
                model.eval()
                with torch.no_grad():
                    value = bundle.validate(dataset,validation_names)
                model.train()
                improved = best is None or value<best
                if improved:
                    best = value
                history[-1]['validation_loss'] = value
            state = dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                update=update,history=history,config=config,prepared_identity=prepared['identity'],stage=stage,
                spec=bundle.spec,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                data_rng=rng.bit_generator.state,cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                extra=bundle.extra_state(),training_cases=names,scope=prepared['scope'],best=best)
            state.update(prepared=prepared,initialization_sha256=initial_sha)
            save_state(root/'latest.pt',state)
            if improved:
                save_state(root/'best.pt',state)
            print(json.dumps(dict(model=config['model'],stage=stage,**history[-1]),ensure_ascii=False),flush=True)
        summary = dict(model=config['model'],stage=stage,scope=prepared['scope'],update=update,history=history,
                       checkpoint=str(root/'latest.pt'),checkpoint_sha256=file_digest(root/'latest.pt'),
                       seconds=perf_counter()-started,rss_bytes=psutil.Process().memory_info().rss,
                       peak_vram_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
        write_json(root/'summary.json',summary)
    dataset.clear()
    return summary


def predict(config, prepared_path, checkpoint, output, case_ids=None, prompts=None):
    started = perf_counter()
    resources(config)
    prepared = load_prepared(prepared_path)
    state = load_state(checkpoint)
    if state['config']['model']!=config['model'] or state['scope']!=prepared['scope']:
        raise ValueError('预测模型或范围与检查点不同')
    if not config['technical'] and state['scope']!='formal':
        raise ValueError('正式预测禁止技术检查点')
    trained = state['prepared']
    if prepared['dataset']!=trained['dataset'] or digest(prepared['plan'])!=digest(trained['plan']):
        raise ValueError('新预测输入必须使用训练时的同一数据集语义与拟合计划')
    if state['identity']!=model_identity(config,trained,state['stage'],state['initialization_sha256']):
        raise ValueError('预测配置、源码或上游检查点与训练身份不同')
    bundle = adapter(config,trained,state['stage'],initialize=False)
    bundle.model.load_state_dict(state['model'],strict=True)
    bundle.model.to(config['device']).eval()
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError('预测目录已存在')
    if Path(prepared['manifest']['data_root']) in root.parents:
        raise ValueError('预测输出不能位于原始数据目录内')
    rows = prepared['manifest']['cases']
    names = [row['case_id'] for row in rows] if case_ids is None else case_ids
    if not names or not set(names)<={row['case_id'] for row in rows}:
        raise ValueError('预测病例未登记')
    dataset = dataset_for(prepared,False)
    root.mkdir(parents=True)
    results = []
    for name in names:
        entry = dataset.get(name)
        mark = perf_counter()
        with torch.inference_mode():
            probability,diagnostics = bundle.predict(entry,prompts)
        native = restore(probability,entry,prepared['plan'])
        path = root/(name+'.nii.gz')
        image = nib.Nifti1Image(native,np.asarray(entry['source_affine']))
        image.header.set_xyzt_units(entry['source_units'])
        nib.save(image,path)
        check = nib.load(path)
        if check.shape!=tuple(entry['source_shape_xyz']) or not np.array_equal(np.asanyarray(check.dataobj),native):
            raise ValueError('预测保存后读回不一致')
        results.append(dict(case_id=name,file=str(path),sha256=file_digest(path),diagnostics=diagnostics,
                            seconds=perf_counter()-mark,shape=list(check.shape),timing=entry['timing']))
        dataset.clear()
    summary = dict(model=config['model'],scope=prepared['scope'],checkpoint_sha256=file_digest(checkpoint),
                   cases=results,seconds=perf_counter()-started)
    write_json(root/'summary.json',summary)
    return summary
