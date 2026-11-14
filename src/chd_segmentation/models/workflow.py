from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from time import perf_counter

from filelock import FileLock
import numpy as np
import psutil
import torch
import yaml

from ..artifacts.store import digest, file_digest, read_json, write_json
from ..data.dataset import CaseDataset
from ..preprocessing.prepare import load_prepared


def options(path,overrides=None):
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    raw = json.loads(content) if path.suffix.lower()=='.json' else yaml.safe_load(content)
    if overrides:
        overrides = deepcopy(overrides)
        if 'global_batch' in overrides:
            overrides['accumulation' if raw['model']=='q_route1' else 'batch'] = overrides.pop('global_batch')
        if 'execution' in overrides:
            raw['execution'] = {**raw.get('execution',{}),**overrides.pop('execution')}
        raw.update(overrides)
    defaults = dict(model=None, run_dir=None, device='cpu', precision='fp32', threads=2,
                    seed=20261114, updates=2, accumulation=4, learning_rate=0.0002,
                    patch=128, local_size=128, frames=8, resolution=512,
                    pretrained=None, parents={}, groups=None, technical=False,
                    official_root=None, dependency_root=None, freeze_image_encoder=False,
                    max_candidates=24, warmup_updates=2000,method='M07',key_references=None,
                    validate_every=1000,batch=2,workers=15,epochs=None,gray_window=None,
                    work_root=None,execution={},logging=dict(heartbeat_seconds=15,case_details=True))
    if not isinstance(raw, dict) or set(raw)-set(defaults):
        raise ValueError('模型配置字段无效')
    config = {**defaults, **raw}
    if isinstance(config['logging'],dict):
        config['logging'] = {**defaults['logging'],**config['logging']}
    if config['model']=='medsam2':
        if 'learning_rate' not in raw:
            config['learning_rate'] = .00005
        if 'seed' not in raw:
            config['seed'] = 123
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
    if config['model']=='medsam2' and (type(config['batch']) is not int or config['batch']<1 or
        type(config['workers']) is not int or config['workers']<0 or config['epochs'] is not None and
        (type(config['epochs']) is not int or config['epochs']<1)):
        raise ValueError('MedSAM2的batch、worker和epoch数量无效')
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
    logging = config['logging']
    if not isinstance(logging,dict) or set(logging)!= {'heartbeat_seconds','case_details'} or \
            type(logging['heartbeat_seconds']) not in (int,float) or not np.isfinite(logging['heartbeat_seconds']) or \
            logging['heartbeat_seconds']<0 or type(logging['case_details']) is not bool:
        raise ValueError('模型日志心跳需要有限非负秒数，病例细节开关需要布尔值')
    root = Path(path).resolve().parent
    for key in ('run_dir','pretrained','groups','official_root','dependency_root','key_references','work_root'):
        if config[key] is not None:
            config[key] = str((root/Path(config[key])).resolve())
    config['parents'] = {key:str((root/Path(value)).resolve(strict=True)) for key,value in config['parents'].items()}
    from .execution import execution_config
    execution_config(config)
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


def validated_q_groups(config,prepared):
    if not config['groups']:
        raise ValueError('Q正式训练需要显式A/B/C/D分组')
    groups = yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))
    if not isinstance(groups,dict) or set(groups)!=set('ABCD') or any(
            not isinstance(groups[key],list) or not groups[key] for key in groups):
        raise ValueError('Q分组必须包含非空A/B/C/D病例名单')
    rows = {row['case_id']:row for row in prepared['manifest']['cases']}
    seen_patients,seen_cases = set(),set()
    for role in 'ABCD':
        names = groups[role]
        if any(not isinstance(name,str) for name in names) or len(set(names))!=len(names) or \
                not set(names)<=rows.keys() or seen_cases & set(names):
            raise ValueError('Q分组病例重复、重叠或未登记')
        patients = {rows[name].get('patient') for name in names}
        if seen_patients & patients or not all(patients):
            raise ValueError('Q患者分组重叠或缺少患者身份')
        seen_cases.update(names)
        seen_patients.update(patients)
    if prepared['manifest'].get('split_source',{}).get('format')=='q1282' or any('q_group' in row for row in rows.values()):
        expected = {key:set() for key in 'ABCD'}
        source_roles = dict(A={'train'},B={'train'},C={'development'},
                            D={'source_test','target_primary','target_secondary'})
        for name,row in rows.items():
            group = row.get('q_group')
            if group not in expected or row.get('source_role') not in source_roles[group]:
                raise ValueError('Q1282准备描述缺少冻结组别或来源角色')
            expected[group].add(name)
        if any(set(groups[key])!=expected[key] for key in 'ABCD'):
            raise ValueError('Q1282外部分组与准备描述的冻结名单不一致')
    fit_ids = set(prepared['plan'].get('fit_ids',[]))
    if not fit_ids or not fit_ids <= set(groups['A']):
        raise ValueError('Q统计拟合必须限于A组')
    return groups


def training_names(config, prepared, stage):
    if config['technical']:
        if prepared['scope']!='technical' or len(prepared['manifest']['cases'])>2:
            raise ValueError('技术验证必须使用最多两例的技术准备描述')
        return [row['case_id'] for row in prepared['manifest']['cases']]
    if prepared['scope']!='formal':
        raise ValueError('正式模型训练需要正式准备描述')
    if config['model']=='q_route1':
        groups = validated_q_groups(config,prepared)
        return groups['B' if stage=='local' else 'A']
    names = [row['case_id'] for row in prepared['manifest']['cases'] if 'train' in row['roles']]
    if not names:
        raise ValueError('没有训练病例')
    return names


def dataset_for(prepared, with_label, resources=None, log_dir=None, logging=None):
    if resources is None:
        resources = dict(reserve_ram_bytes=4*2**30,cache_bytes=512*2**20)
    return CaseDataset(prepared,resources,with_label=with_label,log_dir=log_dir,logging=logging)


def load_state(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def save_state(path, state):
    pending = path.with_suffix('.pending')
    with pending.open('wb') as stream:
        torch.save(state,stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending,path)


def model_identity(config,prepared,stage,initialization_sha256=None,code=None):
    scientific = {key:value for key,value in config.items() if key not in
                  ('run_dir','threads','dependency_root','official_root','execution','work_root','device','logging')}
    return digest(dict(config=scientific,prepared=prepared['identity'],stage=stage,
        code=source_identity(config) if code is None else code,
        parents={key:file_digest(value) for key,value in config['parents'].items()},
        pretrained=initialization_digest(config,initialization_sha256),
        groups=file_digest(config['groups']) if config['groups'] else None,
        key_references=file_digest(config['key_references']) if config['key_references'] else None))


def checked_identity(config,prepared,stage,initialization_sha256=None,previous=None):
    current = model_identity(config,prepared,stage,initialization_sha256)
    if previous is None or previous==current:
        return current
    audit = read_json(Path(__file__).with_name('code_compatibility.json'))
    if audit['schema']!=1:
        raise ValueError('模型代码兼容审计格式无效')
    target = digest({name.replace('\\','/'):value for name,value in source_identity(config).items()})
    for edge in audit['edges']:
        if edge['target']!=target or config['model'] not in edge['models']:
            continue
        # 两种路径分隔符覆盖原检查点所在平台，完整文件摘要仍须匹配审计记录。
        for separator in ('/','\\'):
            code = {name.replace('/',separator):value for name,value in edge['source_files'].items()}
            if previous==model_identity(config,prepared,stage,initialization_sha256,code):
                return previous
    raise ValueError('模型配置、数据、源码或上游检查点身份不同')


def initialization_digest(config,previous=None):
    if config['pretrained'] is None:
        return None
    if Path(config['pretrained']).is_file():
        return file_digest(config['pretrained'])
    if previous is None:
        raise FileNotFoundError('初始化权重缺失且检查点未保存其摘要')
    return previous


def train(config, prepared_path, stage, resume=False, stop_after=None):
    if config['model']=='medsam2':
        from .medsam2.official import train_official
        return train_official(config,prepared_path,stage,resume,stop_after)
    from .execution import admitted,run_workers
    prepared = load_prepared(prepared_path)
    training_names(config,prepared,stage)
    root = Path(config['run_dir'])/stage
    if root.resolve().is_relative_to(Path(prepared['manifest']['data_root']).resolve()):
        raise ValueError('训练输出不能位于原始数据目录内')
    if stop_after is not None and not 1<=stop_after<=config['updates']:
        raise ValueError('stop-after超出预算')
    root.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(root.parent/(stage+'.lock'),timeout=0):
        if not resume:
            from .startup import check_start
            check_start(root,config['model'],prepared['identity'],stage)
        if resume and not (root/'latest.pt').is_file():
            raise FileNotFoundError('阶段缺少完整检查点')
        topology = read_json(root/'execution-topology.json') if resume else None
        with admitted(config,prepared,stage,topology) as (common,allocation):
            return run_workers(dict(config=config,prepared=prepared,stage=stage,
                resume=resume,stop_after=stop_after),common,allocation)


def _train_q_rank(payload,rank,common,allocation,gate,device):
    from .execution import all_states,synchronize_gradients,pressure,rank_measurement
    from ..training.resources import record_peaks
    config,prepared,stage = payload['config'],payload['prepared'],payload['stage']
    resume,stop_after = payload['resume'],payload['stop_after']
    started = perf_counter()
    names = training_names(config,prepared,stage)
    root = Path(config['run_dir'])/stage
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed,rank]))
    previous = load_state(root/'latest.pt') if resume else None
    if previous:
        from ..training.checkpoint import require_resume_device
        require_resume_device(previous['config']['device'],device)
    initial_sha = initialization_digest(config,previous['initialization_sha256'] if previous else None)
    bundle = adapter(dict(config,device=str(device),seed=seed+rank),prepared,stage,initialize=not resume)
    model = bundle.model.to(device)
    model.train()
    optimizer = bundle.optimizer()
    identity = checked_identity(config,prepared,stage,initial_sha,previous['identity'] if previous else None)
    class AdmittedCases(CaseDataset):
        position = 0
        mode = 'candidates'

        def get(self,name):
            if self.mode=='candidates':
                gate.ready.wait(rank,f'q-candidate:{self.position}:{name}',device)
            with gate.build(rank,gate.batch.value,self.position):
                gate.reserve(rank,name)
                entry = super().get(name)
            self.position += 1
            return entry
    dataset = AdmittedCases(prepared,dict(reserve_ram_bytes=common['resources']['reserve_ram_bytes'],
        cache_bytes=allocation['supply']['cache_bytes'][rank]),with_label=True,
        log_dir=root,logging=config['logging'])
    gate.start(rank,'q-candidates',[len(names) if stage=='local' else 0]*allocation['world'],device)
    bundle.prepare_training(dataset,names)
    gate.finish(rank,'q-candidates',device)
    update,history,best = 0,[],None
    restored = False
    if resume:
        state = previous
        if state['identity']!=identity:
            raise ValueError('恢复的配置、数据、源码或上游身份不同')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        from ..training.checkpoint import verify_optimizer_device
        verify_optimizer_device(optimizer)
        if len(state['rank_states'])!=allocation['world'] or state['batch_layout']!=allocation['shares']:
            raise ValueError('Q恢复需要相同进程数量与批量份额')
        local_state = state['rank_states'][rank]
        torch.set_rng_state(local_state['torch_rng'])
        random.setstate(local_state['python_rng'])
        np.random.set_state(local_state['numpy_rng'])
        rng.bit_generator.state = local_state['data_rng']
        if device.type=='cuda':
            torch.cuda.set_rng_state(local_state['cuda_rng'],device)
        bundle.restore_extra(local_state['extra'])
        from ..training.checkpoint import tensors_equal
        restored = (tensors_equal(model.state_dict(),state['model']) and
            tensors_equal(optimizer.state_dict(),state['optimizer']) and
            torch.equal(torch.get_rng_state(),local_state['torch_rng']) and
            random.getstate()==local_state['python_rng'] and
            np.random.get_state()[0]==local_state['numpy_rng'][0] and
            np.array_equal(np.random.get_state()[1],local_state['numpy_rng'][1]) and
            np.random.get_state()[2:]==local_state['numpy_rng'][2:] and
            (device.type!='cuda' or torch.equal(torch.cuda.get_rng_state(device),local_state['cuda_rng'])) and
            tensors_equal(bundle.extra_state(),local_state['extra']) and
            rng.bit_generator.state==local_state['data_rng'])
        if not restored:
            raise ValueError('Q模型、优化器或逐进程随机状态未完整恢复')
        update,history,best = state['update'],state['history'],state['best']
    reason = None
    peak_vram = 0
    replicas_equal = None
    with torch.enable_grad():
        while update < (config['updates'] if stop_after is None else stop_after):
            mark = perf_counter()
            gate.start(rank,f'q-update:{update}',allocation['shares'],device)
            dataset.mode,dataset.position = 'training',0
            if pressure(common,device):
                reason = 'resource_pressure'
                break
            optimizer.zero_grad(set_to_none=True)
            bundle.schedule(optimizer,update)
            loss_sum = 0.
            for micro in range(allocation['shares'][rank]):
                sample_started = perf_counter()
                entry = bundle.sample(dataset,names,rng)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                    loss = bundle.loss(entry,rng,update)
                if not torch.isfinite(loss):
                    raise FloatingPointError('模型训练损失非有限')
                (loss/config['accumulation']).backward()
                loss_sum += float(loss.detach())/config['accumulation']
                del loss,entry
                from ..observability.events import event
                event(root,'model_sample_consumed',rank=rank,batch=update,position=micro,
                    seconds=perf_counter()-sample_started,cuda_completion_checked_at='batch_boundary')
            gate.finish(rank,f'q-update:{update}',device)
            communication_started = perf_counter()
            synchronize_gradients(model)
            loss_sum = sum(all_states(loss_sum))
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            event(root,'model_gradient_communication',rank=rank,batch=update,
                seconds=perf_counter()-communication_started,all_ranks_data_consumed=True)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if config['technical']:
                import hashlib
                checksum = hashlib.sha256()
                for value in model.state_dict().values():
                    checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                gate.ready.wait(rank,f'q-parameters:{update}',device)
                replicas_equal = len(set(all_states(checksum.hexdigest())))==1
                if not replicas_equal:
                    raise ValueError('Q各进程更新后的模型参数不一致')
            if device.type=='cuda':
                peak_vram = max(peak_vram,rank_measurement(device,allocation,rank)['peak_vram_bytes'])
            update += 1
            history.append(dict(update=update,loss=loss_sum,gradient_norm=float(norm),seconds=perf_counter()-mark))
            improved = False
            if not config['technical'] and (update%config['validate_every']==0 or update==config['updates']):
                validation_names = (validated_q_groups(config,prepared)['C']
                    if config['model']=='q_route1' else [row['case_id'] for row in prepared['manifest']['cases'] if 'validation' in row['roles']])
                if not validation_names:
                    raise ValueError('正式选模需要独立验证病例')
                model.eval()
                gate.start(rank,f'q-validation:{update}',[len(validation_names)]+[0]*(allocation['world']-1),device)
                dataset.mode,dataset.position = 'validation',0
                with torch.no_grad():
                    value = bundle.validate(dataset,validation_names) if rank==0 else None
                gate.finish(rank,f'q-validation:{update}',device)
                value = all_states(value)[0]
                model.train()
                improved = best is None or value<best
                if improved:
                    best = value
                history[-1]['validation_loss'] = value
            local_state = dict(torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                data_rng=rng.bit_generator.state,cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                extra=bundle.extra_state())
            gate.ready.wait(rank,f'q-checkpoint-state:{update}',device)
            rank_states = all_states(local_state)
            state = dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rank_states=rank_states,
                batch_layout=allocation['shares'],
                update=update,history=history,config=config,prepared_identity=prepared['identity'],stage=stage,
                spec=bundle.spec,training_cases=names,scope=prepared['scope'],best=best)
            state.update(prepared=prepared,initialization_sha256=initial_sha)
            if rank==0:
                save_state(root/'latest.pt',state)
                if improved:
                    save_state(root/'best.pt',state)
                from ..observability.events import event
                event(root,'model_update_completed',model=config['model'],stage=stage,**history[-1])
            gate.ready.wait(rank,f'q-checkpoint-written:{update}',device)
        measurement = rank_measurement(device,allocation,rank)
        if device.type=='cuda':
            measurement['peak_vram_bytes'] = max(measurement['peak_vram_bytes'],peak_vram)
        gate.ready.wait(rank,f'q-summary:{update}',device)
        measurements = all_states(measurement)
        summary = dict(model=config['model'],stage=stage,scope=prepared['scope'],update=update,history=history,
                       checkpoint=str(root/'latest.pt'),
                       seconds=perf_counter()-started,rss_bytes=psutil.Process().memory_info().rss,
                       peak_vram_bytes=max(m['peak_vram_bytes'] for m in measurements) if device.type=='cuda' else None,
                       reason=reason,world_size=allocation['world'],batch_layout=allocation['shares'],
                       ranks=measurements,restored_exact=all(all_states(restored)) if resume else None,
                       replicas_equal=replicas_equal)
        if rank==0:
            if not (root/'latest.pt').is_file():
                raise MemoryError('Q在第一次更新前资源不足，未生成训练检查点')
            summary['checkpoint_sha256'] = file_digest(root/'latest.pt')
            write_json(root/'summary.json',summary)
            if update==(config['updates'] if stop_after is None else stop_after):
                record_peaks(common,allocation['spec'],allocation['gpus'],allocation['shares'],
                    [m['peak_vram_bytes'] for m in measurements] if allocation['gpus'] else [])
    dataset.clear()
    return summary


from .prediction import predict
