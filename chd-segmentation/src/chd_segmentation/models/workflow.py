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
from ..preprocessing.prepare import load_prepared
from ..training.attempts import settle_attempts


def options(path,overrides=None):
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    raw = json.loads(content) if path.suffix.lower()=='.json' else yaml.safe_load(content)
    if isinstance(raw, dict) and raw.get('model') in ('totalsegmentator','medsam3'):
        from .standalone_config import options as prediction_options
        return prediction_options(raw,path,overrides)
    if overrides:
        overrides = deepcopy(overrides)
        if 'global_batch' in overrides:
            overrides['accumulation' if raw['model']=='q_route1' else 'batch'] = overrides.pop('global_batch')
        if 'execution' in overrides:
            raw['execution'] = {**raw.get('execution',{}),**overrides.pop('execution')}
        for key in ('early_stopping','input_source'):
            if key in overrides:
                raw[key] = {**raw.get(key,{}), **overrides.pop(key)}
        raw.update(overrides)
    defaults = dict(model=None, run_dir=None, device='cpu', precision='fp32', threads=2,
                    seed=20261114, updates=None, accumulation=4, learning_rate=0.0002,
                    patch=128, local_size=128, frames=8, resolution=512,
                    pretrained=None, parents={}, groups=None, technical=False,
                    official_root=None, dependency_root=None, freeze_image_encoder=False,
                    max_candidates=24, warmup_updates=2000,method='M07',key_references=None,
                    validate_every=1000,batch=2,workers=15,epochs=None,gray_window=None,
                    work_root=None,execution={},logging=dict(heartbeat_seconds=15,case_details=True),
                    input_source={}, technical_roles={}, attempt_ledger=None, early_stopping={},
                    checkpoint_dir=None, updates_per_epoch=None, save_every_updates=250,
                    save_every_seconds=300.,distributed_backend=None)
    if not isinstance(raw, dict) or set(raw)-set(defaults):
        raise ValueError('模型配置字段无效')
    config = {**defaults, **raw}
    if config['technical'] and config['updates'] is None:
        config['updates'] = 2
    from ..training.stopping import selection_options
    config['early_stopping'] = selection_options(config['early_stopping'], config['model'])
    if isinstance(config['logging'],dict):
        config['logging'] = {**defaults['logging'],**config['logging']}
    if config['model']=='medsam2':
        if 'learning_rate' not in raw:
            config['learning_rate'] = .00005
        if 'seed' not in raw:
            config['seed'] = 123
    if config['model'] not in ('medsam2', 'q_route1') or not config['run_dir']:
        raise ValueError('必须指定模型与独立运行目录')
    if config['distributed_backend'] not in (None,'gloo','nccl') or (
            config['model']!='medsam2' and config['distributed_backend'] is not None):
        raise ValueError('MedSAM2可显式选择gloo或nccl；其他模型保持现有后端策略')
    for key in ('updates', 'accumulation', 'threads', 'patch', 'local_size', 'frames', 'resolution', 'max_candidates'):
        if key == 'updates' and config[key] is None:
            continue
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
    for key in ('run_dir','pretrained','groups','official_root','dependency_root','key_references','work_root',
                'attempt_ledger','checkpoint_dir'):
        if config[key] is not None:
            config[key] = str((root/Path(config[key])).resolve())
    config['parents'] = {key:str((root/Path(value)).resolve()) for key,value in config['parents'].items()}
    if not isinstance(config['input_source'],dict):
        raise ValueError('input_source必须为来源配置对象')
    if config['input_source'].get('path') is not None:
        config['input_source'] = dict(config['input_source'],path=str((root/Path(config['input_source']['path'])).resolve()))
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


def adapter(config, prepared, stage, initialize=True, parent_states=None):
    from .contracts import require_capability
    require_capability(config['model'],'train')
    if config['model']=='q_route1':
        from .q_route1.adapter import QAdapter
        return QAdapter(config,prepared,stage,initialize=initialize,parent_states=parent_states)
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
    frozen = prepared['manifest'].get('split_source',{}).get('format')=='q1282' or any('q_group' in row for row in rows.values())
    unit = prepared['manifest'].get('split_source',{}).get('unit',
        'case' if prepared['dataset']=='imagechd' else 'patient')
    if unit not in ('case','patient') or frozen and unit != 'patient':
        raise ValueError('Q划分单位无效或与1282冻结患者协议不符')
    seen_patients,seen_cases = set(),set()
    for role in 'ABCD':
        names = groups[role]
        if any(not isinstance(name,str) for name in names) or len(set(names))!=len(names) or \
                not set(names)<=rows.keys() or seen_cases & set(names):
            raise ValueError('Q分组病例重复、重叠或未登记')
        patients = {rows[name].get('patient') for name in names} if unit=='patient' else set()
        if unit=='patient' and (seen_patients & patients or not all(patients)):
            raise ValueError('Q患者分组重叠或缺少患者身份')
        seen_cases.update(names)
        seen_patients.update(patients)
    if frozen:
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
    return groups


def training_names(config, prepared, stage):
    if config['technical']:
        if prepared['scope']!='technical' or len(prepared['manifest']['cases'])>2:
            raise ValueError('技术验证必须使用最多两例的技术准备描述')
        from .roles import technical_names
        return technical_names(config.get('technical_roles', {}), stage, 'train',
                               [row['case_id'] for row in prepared['manifest']['cases']])
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
    from ..data.dataset import CaseDataset
    if resources is None:
        resources = dict(reserve_ram_bytes=4*2**30,cache_bytes=512*2**20)
    return CaseDataset(prepared,resources,with_label=with_label,log_dir=log_dir,logging=logging)


def selected_prepared(prepared, names):
    names = set(names)
    rows = prepared['manifest']['cases']
    if not names or not names <= {row['case_id'] for row in rows}:
        raise ValueError('当前消费名单不属于实验描述')
    return dict(prepared,manifest=dict(prepared['manifest'],cases=[row for row in rows if row['case_id'] in names]))


def stage_case_names(config, prepared, stage):
    names = training_names(config,prepared,stage)
    if config['technical']:
        from .roles import technical_names
        validation = technical_names(config['technical_roles'],stage,'validation',
            [row['case_id'] for row in prepared['manifest']['cases']])
    elif config['model']=='q_route1':
        validation = validated_q_groups(config,prepared)['C']
    else:
        validation = [row['case_id'] for row in prepared['manifest']['cases'] if 'validation' in row['roles']]
    if not validation:
        raise ValueError('训练选模需要独立验证病例')
    return list(dict.fromkeys(names+validation))


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
                  ('run_dir','threads','dependency_root','official_root','execution','work_root','device','logging','distributed_backend')}
    return digest(dict(config=scientific, dataset=prepared['dataset'], stage=stage))


def checked_identity(config,prepared,stage,initialization_sha256=None,previous=None):
    return previous if previous is not None else model_identity(config,prepared,stage)


def initialization_digest(config,previous=None):
    # 历史摘要仅作为来源记录；实际初始化直接由框架读取配置路径。
    return previous


def train(config, prepared_path, stage, resume=False, stop_after=None):
    from .contracts import require_capability
    require_capability(config['model'],'train')
    from ..training.stopping import resolve_budget
    config = dict(config, **resolve_budget(config, config['model'], stage))
    config.pop('unit', None)
    if config['model']=='medsam2':
        from .medsam2.official import train_official
        return train_official(config,prepared_path,stage,resume,stop_after)
    from .execution import admitted,run_workers
    from .offline import model_storage_lease
    prepared = load_prepared(prepared_path)
    training_names(config,prepared,stage)
    consumed = stage_case_names(config,prepared,stage)
    if config.get('input_source',{}).get('path'):
        from ..data.input_binding import bind_prepared_source
        prepared = bind_prepared_source(prepared,config['input_source']['path'],case_ids=consumed)
    root = Path(config['run_dir'])/stage
    if root.resolve().is_relative_to(Path(prepared['manifest']['data_root']).resolve()):
        raise ValueError('训练输出不能位于原始数据目录内')
    if stop_after is not None and not 1<=stop_after<=config['updates']:
        raise ValueError('stop-after超出预算')
    root.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(root.parent/(stage+'.lock'),timeout=0), model_storage_lease(config,selected_prepared(prepared,consumed),stage):
        if not resume:
            from .startup import check_start
            check_start(root,config['model'],prepared['identity'],stage)
        if resume and not (root/'latest.pt').is_file():
            raise FileNotFoundError('阶段缺少完整检查点')
        topology = read_json(root/'execution-topology.json') if resume else None
        with admitted(config,prepared,stage,topology) as (common,allocation):
            return run_workers(dict(config=config,prepared=prepared,stage=stage,
                resume=resume,stop_after=stop_after),common,allocation)


@settle_attempts
def _train_q_rank(payload,rank,common,allocation,gate,device,memory):
    from ..data.dataset import CaseDataset
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
        if (previous['stage']!=stage or previous['config']['model']!=config['model'] or
            previous['prepared']['dataset']!=prepared['dataset'] or previous['scope']!=prepared['scope'] or
            previous['prepared']['dataset_json']!=prepared['dataset_json'] or previous['training_cases']!=names):
            raise ValueError('Q恢复的模型阶段、数据集类别或当前训练名单不同')
        from ..training.checkpoint import require_resume_device
        require_resume_device(previous['config']['device'],device)
    initial_sha = initialization_digest(config,previous['initialization_sha256'] if previous else None)
    source_changed = False
    if previous is not None:
        from .contracts import require_resume_contract
        source_changed = require_resume_contract(previous['config'],previous['prepared'],config,prepared,stage)
    bundle = adapter(dict(config,device=str(device),seed=seed+rank),prepared,stage,initialize=not resume,
        parent_states=previous.get('parent_models') if previous else None)
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
            gate.mark_ready(rank)
            gate.deliver(rank)
            self.position += 1
            return entry
    dataset = AdmittedCases(selected_prepared(prepared,stage_case_names(config,prepared,stage)),dict(reserve_ram_bytes=common['resources']['reserve_ram_bytes'],
        cache_bytes=allocation['supply']['cache_bytes'][rank]),with_label=True,
        log_dir=root,logging=config['logging'])
    from ..observability.progress import Progress
    from ..observability.events import event
    feedback = Progress(dict(common,run_dir=str(root)),'train',total=config['updates'],unit='updates') if rank==0 else None
    if rank==0:
        feedback.update(done=previous['update'] if previous else 0,phase='准备候选')
        feedback.snapshot(force=True)
        event(root,'model_candidates_started',stage=stage,cases=len(names))
    candidates_started = perf_counter()
    gate.start(rank,'q-candidates',[len(names) if stage=='local' else 0]*allocation['world'],device)
    bundle.prepare_training(dataset,names)
    gate.finish(rank,'q-candidates',device)
    if rank==0:
        event(root,'model_candidates_completed',stage=stage,seconds=perf_counter()-candidates_started)
    update,history,best = 0,[],None
    selection = None
    from ..training.stopping import selection_options, update_selection
    stopping = selection_options(config.get('early_stopping', {}), config['model'])
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
        if not source_changed:
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
            (source_changed or tensors_equal(bundle.extra_state(),local_state['extra'])) and
            rng.bit_generator.state==local_state['data_rng'])
        if not restored:
            raise ValueError('Q模型、优化器或逐进程随机状态未完整恢复')
        if source_changed:
            restored = False
        update,history,best = state['update'],state['history'],state['best']
        if source_changed:
            best = None
        selection = None if source_changed else state.get('selection')
    reason = None
    saved_at = perf_counter()
    peak_vram = 0
    vram_complete = True
    replicas_equal = None
    resumed_update = update
    if rank == 0:
        feedback.update(done=update, new=0, step=update,phase='训练更新')
        feedback.snapshot(force=True)
    with torch.enable_grad():
        while update < (config['updates'] if stop_after is None else stop_after):
            mark = perf_counter()
            from ..training.attempts import claim_attempt, remaining_attempts
            remaining = remaining_attempts(config, stage) if rank == 0 else None
            remaining = all_states(remaining)[0]
            if config['technical'] and remaining == 0:
                reason = 'attempt_limit'
                break
            attempt = claim_attempt(config, stage) if rank == 0 else None
            attempt = all_states(attempt)[0]
            gate.start(rank,f'q-update:{update}',allocation['shares'],device)
            dataset.mode,dataset.position = 'training',0
            optimizer.zero_grad(set_to_none=True)
            if pressure(common,device,gate.ready,rank,f'q-update:{update}',memory,release=dataset.clear):
                reason = 'resource_pressure'
                break
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
            if rank == 0:
                from ..training.attempts import finish_attempt
                finish_attempt(config, stage, attempt)
                from ..observability.events import event
                event(root,'model_optimizer_completed',model=config['model'],stage=stage,
                    attempt=attempt,update=update+1,successful=True)
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
                observed = rank_measurement(device,allocation,rank)
                if observed['peak_vram_bytes'] is None:
                    vram_complete = False
                else:
                    peak_vram = max(peak_vram,observed['peak_vram_bytes'])
            update += 1
            if rank==0:
                feedback.update(done=update,new=update-resumed_update,step=update,phase='优化器更新完成')
                feedback.snapshot(force=True)
            history.append(dict(update=update,loss=loss_sum,gradient_norm=float(norm),seconds=perf_counter()-mark))
            improved = False
            if update%config['validate_every']==0 or update==config['updates'] or update==stop_after:
                if config['technical']:
                    from .roles import technical_names
                    validation_names = technical_names(config['technical_roles'], stage, 'validation', names + [
                        row['case_id'] for row in prepared['manifest']['cases']])
                else:
                    validation_names = validated_q_groups(config,prepared)['C']
                if not validation_names:
                    raise ValueError('正式选模需要独立验证病例')
                model.eval()
                validation_started = perf_counter()
                if rank==0:
                    feedback.update(phase='独立验证')
                    feedback.snapshot(force=True)
                    event(root,'model_validation_started',stage=stage,update=update,cases=len(validation_names))
                gate.start(rank,f'q-validation:{update}',[len(validation_names)]+[0]*(allocation['world']-1),device)
                dataset.mode,dataset.position = 'validation',0
                with torch.no_grad():
                    value = bundle.validate(dataset,validation_names) if rank==0 else None
                gate.finish(rank,f'q-validation:{update}',device)
                value = all_states(value)[0]
                model.train()
                selection = update_selection(selection, value, update, stopping)
                improved, best = selection['improved'], selection['best']
                history[-1]['validation_loss'] = value
                if rank==0:
                    event(root,'model_validation_completed',stage=stage,update=update,loss=value,
                        seconds=perf_counter()-validation_started)
            due = (config['technical'] or improved or selection and selection['stop'] or
                   update == (config['updates'] if stop_after is None else stop_after) or
                   update % config.get('save_every_updates', 250) == 0 or
                   perf_counter()-saved_at >= config.get('save_every_seconds', 300.)) if rank == 0 else None
            due = all_states(due)[0]
            local_state = dict(torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                data_rng=rng.bit_generator.state,cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                extra=bundle.extra_state())
            gate.ready.wait(rank,f'q-checkpoint-state:{update}',device)
            rank_states = all_states(local_state)
            state = dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rank_states=rank_states,
                batch_layout=allocation['shares'],
                update=update,history=history,config=config,prepared_identity=prepared['identity'],stage=stage,
                selection_source_changed=source_changed,
                spec=bundle.spec,training_cases=names,scope=prepared['scope'],best=best,
                selection=selection, attempt=attempt)
            state.update(prepared=prepared,initialization_sha256=initial_sha)
            if stage=='local':
                state['parent_models'] = bundle.parent_models
            if rank==0 and due:
                feedback.update(phase='保存检查点')
                feedback.snapshot(force=True)
                checkpoint_started = perf_counter()
                event(root,'model_checkpoint_started',stage=stage,update=update,best_improved=improved)
                save_state(root/'latest.pt',state)
                if improved:
                    save_state(root/'best.pt',state)
                from ..observability.events import event
                event(root,'model_update_completed',model=config['model'],stage=stage,**history[-1])
                event(root,'model_checkpoint_completed',stage=stage,update=update,
                    seconds=perf_counter()-checkpoint_started)
            if due:
                saved_at = perf_counter()
            gate.ready.wait(rank,f'q-checkpoint-written:{update}',device)
            if rank == 0:
                feedback.update(done=update, new=update-resumed_update, step=update,
                    pending=config['updates']-update, active=0,phase='当前更新已保存')
                feedback.snapshot(force=True)
            if selection and selection['stop']:
                reason = 'early_stopping'
                break
        measurement = rank_measurement(device,allocation,rank)
        if device.type=='cuda':
            if not vram_complete or measurement['peak_vram_bytes'] is None:
                measurement.update(peak_vram_bytes=None,vram_complete=False)
            else:
                measurement['peak_vram_bytes'] = max(measurement['peak_vram_bytes'],peak_vram)
        gate.ready.wait(rank,f'q-summary:{update}',device)
        measurements = all_states(measurement)
        from ..acceleration.measurements import complete_peak
        summary = dict(model=config['model'],stage=stage,scope=prepared['scope'],update=update,history=history,
                       checkpoint=str(root/'latest.pt'),
                       seconds=perf_counter()-started,rss_bytes=psutil.Process().memory_info().rss,
                       peak_vram_bytes=complete_peak(measurements),
                       reason=reason,world_size=allocation['world'],batch_layout=allocation['shares'],
                       ranks=measurements,restored_exact=all(all_states(restored)) if resume else None,
                       replicas_equal=replicas_equal)
        if rank==0:
            if not (root/'latest.pt').is_file():
                raise MemoryError('Q在第一次更新前资源不足，未生成训练检查点')
            write_json(root/'summary.json',summary)
            if update==(config['updates'] if stop_after is None else stop_after):
                from .execution import record_model_peaks
                record_model_peaks(common,allocation['spec'],allocation['gpus'],allocation['shares'],measurements)
    dataset.clear()
    if rank == 0:
        feedback.close()
    return summary


from .prediction import predict
