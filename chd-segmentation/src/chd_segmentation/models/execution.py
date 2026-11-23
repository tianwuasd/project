from contextlib import ExitStack,contextmanager,redirect_stdout,redirect_stderr
from dataclasses import asdict
from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import socket
import sys

import psutil
import torch
import torch.distributed as dist
import yaml

from ..artifacts.store import digest,file_digest,read_json,write_json
from ..config import DEFAULTS,_merge,validate_resources
from ..observability.events import Stage,event
from ..observability.progress import console_progress
from ..training.memory_guard import MemoryGuard
from ..training.resources import resource_lease,layout,register_member,close_members,host_capacity,process_vram,record_peaks


def execution_config(config):
    resources = _merge(DEFAULTS['resources'],config['execution'])
    device = torch.device(config['device'])
    resources['device'] = device.type
    if device.type=='cuda' and device.index is not None and 'gpu_ids' not in config['execution']:
        if resources['gpu_count']!=1:
            raise ValueError('cuda:index用于单卡；多卡请使用device: cuda及execution.gpu_count或gpu_ids')
        resources['gpu_ids'] = [device.index]
    batch = config['batch'] if config['model']=='medsam2' else config['accumulation']
    if config['model']=='q_route1':
        if resources['workers'] not in ('auto',0):
            raise ValueError('Q在各训练进程内读取病例，execution.workers需为0或auto')
        resources['workers'] = 0
    elif 'workers' not in config['execution']:
        resources['workers'] = config['workers']
    validate_resources(resources,batch)
    resources['technical_cpu'] = bool(config['technical'] and device.type == 'cpu' and sys.platform == 'win32')
    world = resources['gpu_count'] if device.type=='cuda' else resources['cpu_processes']
    shares = layout(batch,world,resources['batch_layout'])
    if config['model']=='medsam2' and len(set(shares))!=1:
        raise ValueError('MedSAM2官方DistributedSampler要求全局batch可均分到各卡')
    patch = [config['frames'],config['resolution'],config['resolution']] if config['model']=='medsam2' else [config['patch']]*3
    return dict(model=config['model'],resources=resources,run_dir=config['run_dir'],
        input_source=config.get('input_source',{}),
        scope='technical' if config['technical'] else 'formal',pretrained=config['pretrained'],
        logging=config.get('logging',DEFAULTS['logging']),
        work_root=config['work_root'] or str(Path(config['run_dir']).parent),
        technical=dict(roi_xyz=None),training=dict(batch=batch,patch=patch,precision=config['precision']))


def resource_spec(config,prepared,stage):
    return dict(resource_model=config['model'],stage=stage,patch=config['patch'],local_size=config['local_size'],
        frames=config['frames'],resolution=config['resolution'],technical=config['technical'],
        method=config['method'],prepared=prepared['identity'],parents=config['parents'])


def training_budget(config,prepared,stage):
    from .workflow import stage_case_names, training_names, validated_q_groups
    names = set(stage_case_names(config,prepared,stage))
    budget = deepcopy(prepared)
    train = set(training_names(config,prepared,stage))
    if config['technical']:
        from .roles import technical_names
        validation = set(technical_names(config['technical_roles'],stage,'validation',list(names)))
    elif config['model']=='q_route1':
        validation = set(validated_q_groups(config,prepared)['C'])
    else:
        validation = {row['case_id'] for row in prepared['manifest']['cases'] if 'validation' in row['roles']}
    budget['q_validation_names'] = sorted(validation)
    budget['manifest']['cases'] = [dict(row,roles=(['train'] if row['case_id'] in train else []) +
                                                       (['validation'] if row['case_id'] in validation else []))
                                  for row in prepared['manifest']['cases']
                                  if row['case_id'] in names]
    return budget


@contextmanager
def admitted(config,prepared,stage,topology=None):
    common = execution_config(config)
    common['run_dir'] = str(Path(config['run_dir'])/'admission'/stage)
    common['technical'] = prepared['technical']
    common['model_stage'] = stage
    if stage=='prediction':
        common['training']['batch'] = 1
        common['training']['precision'] = 'fp32'
    if topology is not None:
        requested_layout = common['resources']['batch_layout']
        if requested_layout is not None and requested_layout!=topology['shares']:
            raise ValueError('模型恢复需要检查点原批量份额')
        common['model_topology'] = topology
        common['resources']['batch_layout'] = layout(common['training']['batch'],topology['world'],topology['shares'])
        common['resources'].update(gpu_count=topology['world'],cpu_processes=topology['world'],
            min_gpus=topology['world'] if common['resources']['device']=='cuda' else None)
        if common['resources']['gpu_ids'] is not None and len(common['resources']['gpu_ids'])!=topology['world']:
            raise ValueError('恢复时指定显卡数量必须与模型检查点一致')
    root = Path(common['work_root']).resolve()
    source = Path(prepared['manifest']['data_root']).resolve()
    if root==source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('资源登记目录与原始数据重叠')
    spec = resource_spec(config,prepared,stage)
    budget = prepared
    if stage!='prediction':
        budget = training_budget(config,prepared,stage)
    from ..acceleration import ResourceSession, RuntimeRequest, ResourceDemand
    from ..training.budget import cpu_plan
    world = common['resources']['gpu_count'] if common['resources']['device'] == 'cuda' else common['resources']['cpu_processes']
    supply = cpu_plan(common, budget, world, stage='prediction' if stage=='prediction' else 'train')
    amount = supply['ram_bytes'] if supply['admitted'] else supply.get('minimum_bytes', 0)
    if amount <= 0:
        raise RuntimeError('模型供数无法建立有效工作集预算：'+supply.get('reason', '未知原因'))
    from ..training.session import supply_demand,supply_layout,supply_planner
    shares = layout(common['training']['batch'],world,common['resources']['batch_layout'])
    demand = supply_demand(supply,world,common['resources'],shares)
    request = RuntimeRequest(str(Path(config['run_dir']).resolve()), stage, root/'resources',
        common['run_dir'], dict(common['resources'],gpu_spec=spec,max_wait=common['resources']['wait_seconds'],
            precision=common['training']['precision'],previous_world=topology['world'] if topology is not None else None), demand)
    with ResourceSession(request,layout=supply_layout(supply),planner=supply_planner(common,budget,
            'prediction' if stage=='prediction' else 'train',supply['case_estimates'])) as session:
        applied = session.allocation
        common['resource_handle'] = session.handle
        common['resources'] = dict(common['resources'],batch_layout=applied.shares,
            cpu_processes=applied.world,
            gpu_count=applied.world if applied.gpus else common['resources']['gpu_count'],
            min_gpus=applied.world if applied.gpus else common['resources']['min_gpus'],
            gpu_ids=[gpu['uuid'] for gpu in applied.gpus] if applied.gpus else common['resources']['gpu_ids'])
        allocation = dict(owner=session.handle.owner, gpus=session.allocation.gpus,
            world=applied.world,supply=applied.supply,shares=applied.shares,
            resource_demand=asdict(session.applied_demand),resource_layout=session.requested_layout)
        if config['model']=='medsam2' and len(set(allocation['shares']))!=1:
            raise ValueError('MedSAM2官方训练要求各进程均分全局batch')
        allocation['spec'] = spec
        common['resource_session'] = session
        yield common,allocation


def all_states(value):
    values = [None]*dist.get_world_size()
    dist.all_gather_object(values,value)
    return values


def model_memory(common):
    from ..acceleration.legacy import SessionGuard
    guard = SessionGuard(common['resource_handle'])
    guard.config = common
    return guard


def pressure(common,device,ready,rank,phase,memory,release=None):
    r = common['resources']
    with memory.admit(0, label=phase, release=release) as ticket:
        low = ticket is None
    if device.type=='cuda':
        free,total = torch.cuda.mem_get_info(device)
        low |= free<max(r['vram_reserve_bytes'],total*r['vram_reserve_fraction'])
    ready.wait(rank, 'resource-ready:'+phase, device)
    flag = torch.tensor(int(low),device=device)
    dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())


def synchronize_gradients(model):
    for parameter in model.parameters():
        present = torch.tensor(int(parameter.grad is not None),device=parameter.device)
        dist.all_reduce(present,op=dist.ReduceOp.SUM)
        if present.item():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)


def verify_device(device,allocation,rank):
    if device.type=='cuda':
        torch.cuda.set_device(device)
        observed = str(torch.cuda.get_device_properties(device).uuid)
        observed = observed if observed.startswith('GPU-') else 'GPU-'+observed
        if observed.lower()!=allocation['gpus'][rank]['uuid'].lower():
            raise ValueError('实际CUDA设备与获准UUID不同')
        torch.cuda.reset_peak_memory_stats(device)


def rank_measurement(device,allocation,rank,common=None):
    memory = psutil.Process().memory_info()
    peak,allocated,reserved,external = None,None,None,dict(bytes=None,reason='cpu_execution')
    if device.type=='cuda':
        torch.cuda.synchronize(device)
        from ..acceleration.measurements import process_vram_observation
        allocated,reserved = torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)
        common = common or {}
        external = process_vram_observation(allocation['gpus'][rank]['uuid'],os.getpid(),
            resources=common.get('resources'),cancel=common.get('memory_cancel'),log_dir=common.get('run_dir'))
        if external['bytes'] is not None:
            peak = reserved+max(0,external['bytes']-torch.cuda.memory_reserved(device))
    return dict(rank=rank,rss_bytes=memory.rss,peak_rss_bytes=memory.peak_wset if sys.platform=='win32' else None,
                peak_vram_bytes=peak,torch_peak_allocated_bytes=allocated,torch_peak_reserved_bytes=reserved,
                external_process_vram=external,vram_complete=peak is not None)


def record_model_peaks(common,spec,gpus,shares,measurements):
    if not gpus:
        return
    if not all(row['vram_complete'] and row['peak_vram_bytes'] is not None for row in measurements):
        event(common['run_dir'],'gpu_peak_record_skipped',reason='external_process_memory_unobservable',
            ranks=measurements,capacity_evidence_written=False)
        return
    record_peaks(common,spec,gpus,shares,[row['peak_vram_bytes'] for row in measurements])


def rank_entry(rank,payload,common,allocation,gate,port):
    payload = deepcopy(payload)
    execution = {key:value for key,value in common['resources'].items() if key != 'technical_cpu'}
    payload['config']['execution'] = dict(execution,workers=sum(allocation['supply']['workers']))
    payload['config']['execution']['cache_bytes'] = allocation['supply']['cache_bytes'][rank]
    root = Path(payload['config']['run_dir'])/payload['stage']
    with (root/f'rank{rank}.log').open('a',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log), \
            Stage(root,'model_worker',heartbeat_seconds=common['logging']['heartbeat_seconds'],
                  model=payload['config']['model'],rank=rank,operation=payload.get('operation','train')), \
            model_memory(common) as memory:
        register_member(common,allocation['owner'])
        os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),LOCAL_RANK=str(rank),
                          WORLD_SIZE=str(allocation['world']))
        torch.set_num_threads(1)
        event(root,'model_runtime_applied',model=payload['config']['model'],rank=rank,
            local_batch=allocation['shares'][rank],workers=allocation['supply']['workers'][rank],
            reserve_ram_bytes=common['resources']['reserve_ram_bytes'],
            cache_bytes=allocation['supply']['cache_bytes'][rank],
            reader_slots=allocation['supply']['reader_slots'],prefetch=allocation['supply']['prefetch'],threads=1,
            cpu_preparation=payload['config'].get('cpu_preparation',{}))
        with memory.admit(0, label='model_initialization') as ticket:
            if ticket is None:
                raise RuntimeError('模型启动在取消或明确停止后结束，保留已有材料')
        if payload.get('operation')=='predict':
            device = torch.device('cuda',rank) if allocation['gpus'] else torch.device('cpu')
            verify_device(device,allocation,rank)
            from .prediction import _predict
            applied = dict(payload['config'],device=str(device),threads=1)
            applied['prediction_case_estimates'] = {row['case_id']:row['workspace_bytes']
                for row in allocation['supply']['case_estimates']}
            summary = _predict(applied,payload['prepared_path'],payload['checkpoint'],payload['output'],
                payload['case_ids'],payload['prompts'],memory=memory,**payload.get('prediction_options',{}))
            write_json(root/'summary.json',summary)
            measurement = rank_measurement(device,allocation,rank,common)
            record_model_peaks(common,allocation['spec'],allocation['gpus'],allocation['shares'],[measurement])
        elif payload['config']['model']=='medsam2':
            from .medsam2.worker import main
            main(payload,rank,common,allocation,gate,memory)
        else:
            device = torch.device('cuda',rank) if allocation['gpus'] else torch.device('cpu')
            verify_device(device,allocation,rank)
            gate.ready.wait(rank,'q-distributed-entry',device)
            dist.init_process_group('nccl' if allocation['gpus'] else 'gloo',rank=rank,
                world_size=allocation['world'],timeout=timedelta(seconds=common['resources']['collective_timeout_seconds']),
                **({'device_id':device} if allocation['gpus'] else {}))
            from .workflow import _train_q_rank
            _train_q_rank(payload,rank,common,allocation,gate,device,memory)
            dist.destroy_process_group()


def run_workers(payload,common,allocation):
    if payload.get('operation')=='predict':
        return _run_workers(payload,common,allocation)
    from .startup import initialization_attempt
    root = Path(payload['config']['run_dir'])/payload['stage']
    with initialization_attempt(root,payload['config'],payload['prepared']['identity'],payload['stage'],payload['resume']):
        return _run_workers(payload,common,allocation)


def _run_workers(payload,common,allocation):
    common = dict(common)
    session = common.pop('resource_session')
    root = Path(payload['config']['run_dir'])/payload['stage']
    root.mkdir(parents=True,exist_ok=True)
    if payload['config']['model']=='medsam2' and payload.get('operation')!='predict':
        write_json(root/'worker-input.json',payload)
        write_json(root/'identity.json',dict(identity=payload['identity']))
        write_json(root/'official-config.json',payload['recipe'])
    topology = dict(world=allocation['world'],shares=allocation['shares'],
        workers=allocation['supply']['workers'] if payload['config']['model']=='medsam2' else [])
    if payload.get('resume'):
        if read_json(root/'execution-topology.json')!=topology:
            raise ValueError('恢复需要相同的进程、批量份额及数据worker数量')
    else:
        write_json(root/'execution-topology.json',topology)
    write_json(root/'resources.json',dict(requested=payload['config']['execution'],resolved=common['resources'],
        applied=allocation,threads_per_rank=1))
    runtime = root/'runtime'
    runtime.mkdir(exist_ok=True)
    with socket.socket() as rendezvous:
        rendezvous.bind(('127.0.0.1',0))
        port = rendezvous.getsockname()[1]
    if allocation['gpus'] and torch.cuda.is_initialized():
        raise RuntimeError('协调进程已初始化CUDA，需要从新命令启动')
    project = Path(__file__).resolve().parents[3]
    keep = {'PATH','SYSTEMROOT','WINDIR','COMSPEC','PATHEXT','USERPROFILE','APPDATA','LOCALAPPDATA',
        'PROGRAMDATA','PROGRAMFILES','PROGRAMFILES(X86)','SYSTEMDRIVE','NUMBER_OF_PROCESSORS','USERNAME','USER','LOGNAME'}
    environment = {key:value for key,value in os.environ.items() if key.upper() in keep}
    if payload['config'].get('technical') is True:
        for key in ('CHD_MINI_TOOL_ROOT','CHD_MINI_ATTEMPT_LEDGER','CHD_MINI_STAGE','CHD_MINI_ROLES'):
            if key in os.environ:
                environment[key] = os.environ[key]
    paths = [str(project/'src')]
    if payload['config']['model']=='medsam2':
        paths += [payload['config']['dependency_root'],payload['config']['official_root']]
    environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(paths),USE_LIBUV='0',
        PYTHONUTF8='1',TMPDIR=str(runtime),MPLCONFIGDIR=str(runtime/'matplotlib'),
        TORCH_HOME=str(runtime/'torch-home'),HF_HOME=str(runtime/'huggingface'),XDG_CACHE_HOME=str(runtime/'cache'),
        TEMP=str(runtime),TMP=str(runtime),TORCHINDUCTOR_CACHE_DIR=str(runtime/'torch'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in allocation['gpus']))
    context = torch.multiprocessing.get_context('spawn')
    from .supply import CpuReady,OrderedSupply
    cancel = common['resource_handle'].cancel
    common = dict(common, run_dir=str(root), memory_owner=allocation['owner'], memory_cancel=cancel)
    ready = CpuReady(context,allocation['world'],root,owner=allocation['owner'],handle=common['resource_handle'])
    gate = OrderedSupply(context,allocation['shares'],allocation['supply']['reader_slots'],ready,root,
        capacity_bytes=allocation['supply']['breakdown']['case_workspaces_bytes'],
        estimates=allocation['supply']['case_estimates'],
        unit='sequence_batch' if payload['config']['model']=='medsam2' else 'case_entry',handle=common['resource_handle'])
    previous = dict(os.environ)
    with ExitStack() as cleanup:
        def restore_environment():
            os.environ.clear()
            os.environ.update(previous)
        cleanup.callback(restore_environment)
        os.environ.clear()
        os.environ.update(environment)
        memory = cleanup.enter_context(model_memory(common))
        reserve = allocation['supply']['model_memory']['checkpoint_reserve_bytes']
        ticket = cleanup.enter_context(memory.admit(reserve, label='checkpoint_reserve', reserve=True))
        if ticket is None:
            raise RuntimeError('模型准备已取消，完整检查点保持原位置')
        succeeded, last_console, last_done, last_validation = False, 0., None, 0.
        path = root/('progress-predict.json' if payload.get('operation') == 'predict' else 'progress-train.json')
        try:
            processes = torch.multiprocessing.spawn(rank_entry,args=(payload,common,allocation,gate,port),
                                                    nprocs=allocation['world'],join=False)
            from time import perf_counter
            while not processes.join(timeout=1):
                session.service_replans()
                memory.boundary()
                if path.is_file():
                    progress = read_json(path)
                    if progress['done'] != last_done or perf_counter()-last_console >= 15:
                        console_progress(progress)
                        last_console, last_done = perf_counter(), progress['done']
                if payload['config']['model'] == 'q_route1':
                    from .q_route1.validation_runtime import display_progress
                    last_validation = display_progress(root, allocation['world'], last_validation)
            succeeded = True
            if path.is_file():
                console_progress(read_json(path))
        finally:
            if not succeeded:
                cancel.set()
    return read_json(root/'summary.json')
