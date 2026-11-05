"""CPU-only owner of CUDA ranks, V2 admission, reconfiguration and recovery.

No shell, global process-name killing or SHM cleanup is used. Children inherit
the authorized CUDA visibility. Each launch imports the frozen source snapshot.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from uuid import uuid4
import psutil
import torch
from ..config import ROOT
from ..acceleration.preprocess_cpu import snapshot,interpreter_command
from ..observability.locking import run_lock
from ..observability.runlog import write_json
from .distributed_engine import prepare_contract
from .checkpoints import read_checkpoint
from .code_snapshot import freeze_code
from .engine import validate_state,repair_best_pointer,summary_from_state
from .feedback import TrainingLog
from .parallel_policy import layout,supply_plan,expansion_baseline,expansion_regressed
from .fleet import query_visible,query_live,eligible
from .data import PatchSampler


def max_ranks(value,batch):
    if value=='auto':return batch
    try:count=int(value)
    except (ValueError,TypeError):raise ValueError('--gpus must be auto or a positive integer')
    if count<1:raise ValueError('--gpus must be auto or a positive integer')
    return min(count,batch)


def remember_children(processes,owned):
    for process in processes:
        try:
            parent=psutil.Process(process.pid)
            # Register the actual Popen child before learning its descendants.
            if process.poll() is not None:continue
            for item in [parent]+parent.children(recursive=True):owned[item.pid]=item.create_time()
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue


def close_owned(processes,owned):
    remember_children(processes,owned)
    handles=[]
    for pid,born in owned.items():
        try:
            item=psutil.Process(pid)
            if item.create_time()!=born:continue
            handles.append(item);item.terminate()
        except psutil.NoSuchProcess:continue
    _,alive=psutil.wait_procs(handles,timeout=5)
    for item in alive:
        try:
            if item.create_time()==owned[item.pid]:item.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=5)
    for process in processes:
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:raise RuntimeError('An owned training process did not exit')


def learn_workers(directory,nonce,parents,owned):
    """Late initializer handshakes are scoped to this launch and parent ranks."""
    for path in directory.glob('worker-*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('nonce')!=nonce or row.get('parent_pid') not in parents:continue
        try:
            process=psutil.Process(row['pid'])
            if process.create_time()==row['born']:owned[row['pid']]=row['born']
        except psutil.NoSuchProcess:continue


def _performance(path,generation):
    if not path.exists():return []
    # Only completed newline-terminated records are visible to the controller.
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
        if not line.endswith('\n'):continue
        row=json.loads(line)
        if row['generation']==generation and row['complete_from_boundary']:rows.append(row)
    return rows


def _remove_control(output,*names):
    for name in names:(output/'control'/name).unlink(missing_ok=True)


def restore_full_epoch(folder,signature):
    payload=read_checkpoint(folder,signature,pointer_name='epoch.json')
    state=payload['state']
    if state['phase']!='train' or state['step']!=0:raise ValueError('Epoch recovery pointer is not a full boundary')
    pointer=json.loads((folder/'epoch.json').read_text(encoding='utf-8'))
    write_json(folder/'latest.json',pointer)
    return state


def launch(prepared,output,*,gpus='auto',elastic=True,wait_seconds=300,workers=None,resume=False,
           check_only=False,**options):
    prepared=Path(prepared).resolve(strict=True);output=Path(output).resolve()
    if output==prepared or output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError('Training output and prepared input must be separate')
    options.setdefault('fold',0)
    if options.get('weights') is not None:options['weights']=str(Path(options['weights']).resolve())
    print('阶段：检查输入和多卡科学合同；全局batch保持不变',flush=True)
    settings,dataset,signature=prepare_contract(prepared,options)
    maximum=max_ranks(gpus,settings['batch_size'])
    if check_only:
        return dict(status='checked',recipe=settings,maximum_ranks=maximum,
            layouts=[layout(settings['batch_size'],w) for w in range(1,maximum+1)],
            train_ids=dataset['train_ids'],val_ids=dataset['val_ids'],
            weights_file_checked=False,gpu_capacity_checked=False)
    previous=None
    if output.exists() and any(output.iterdir()):
        if not resume:raise FileExistsError('Output is not empty; use a new run or --resume')
        if not (output/'signature.json').is_file() or json.loads((output/'signature.json').read_text(encoding='utf-8'))!=signature:
            raise ValueError('Resume contract differs; keep the old run and use its frozen code')
        freeze_code(output,signature,create=False)
        if (output/'checkpoints/latest.json').is_file():
            previous=read_checkpoint(output/'checkpoints',signature);validate_state(previous['state'],settings)
    if previous is None and not options.get('weights'):
        raise ValueError('Author weights required for a new run; no random fallback')
    processes=[];owned={};streams=[];handlers={};stop=[False];stopping_at=None
    def request_stop(signum,frame):
        stop[0]=True
    with run_lock(output/'.training.lock'):
        if (output/'signature.json').is_file():
            if not resume:raise FileExistsError('Another invocation created this run; no training started')
            if json.loads((output/'signature.json').read_text(encoding='utf-8'))!=signature:
                raise ValueError('Resume contract changed while acquiring the run lock')
            freeze_code(output,signature,create=False)
            if (output/'checkpoints/latest.json').is_file():
                previous=read_checkpoint(output/'checkpoints',signature);validate_state(previous['state'],settings)
        if previous is not None and previous['state']['epoch']>=settings['epochs']:
            pointer=json.loads((output/'checkpoints/latest.json').read_text(encoding='utf-8'))
            repair_best_pointer(output,previous['state']);write_json(output/'checkpoints/final.json',pointer)
            result=summary_from_state(previous['state'],settings,dataset,pointer)
            write_json(output/'summary.json',result)
            TrainingLog(output).status('completed',stage='done',epoch=result['epoch'],
                step=result['step'],attempts=result['attempts'],updates=result['updates'],summary=result)
            return result
        pool_loaders={phase:PatchSampler(
            dataset['rows'],dataset[key],settings,settings['architecture'],phase=='train')
            for phase,key in (('train','train_ids'),('validation','val_ids'))}
        if not (output/'signature.json').exists():
            freeze_code(output,signature,create=True);write_json(output/'signature.json',signature)
        log=TrainingLog(output/'control');source=output/'code/src';entry=output/'code/run.py'
        TrainingLog(output).status(stage='gpu_admission',requested_gpus=gpus,global_batch=settings['batch_size'])
        controller_path=output/'control/controller.json'
        controller=json.loads(controller_path.read_text(encoding='utf-8')) if controller_path.exists() else dict(
            pressure_recoveries=0,worker_cap=workers,deferred=[],baseline=None)
        # A reduced supply cap survives restarts and process-group membership.
        if workers is not None:controller['worker_cap']=min(workers,controller['worker_cap']) if controller['worker_cap'] is not None else workers
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.getsignal(sig);signal.signal(sig,request_stop)
        write_json(output/'effective_config.json',dict(recipe=settings,prepared=str(prepared),options=options,
            requested_gpus=gpus,elastic=elastic,requested_workers=workers,wait_seconds=wait_seconds))
        write_json(output/'input_adapters.json',dataset['identity']['adapters'])
        members=None;deadline=time.monotonic()+wait_seconds;last_scan=0;candidate_seen={}
        try:
            while True:
                if stop[0]:return dict(status='stopped',reason='stopped before rank launch')
                rows=query_visible(source);available=eligible(rows);by_uuid={r['uuid']:r for r in available}
                checkpoint=read_checkpoint(output/'checkpoints',signature) if (output/'checkpoints/latest.json').exists() else None
                partial=checkpoint is not None and (checkpoint['state']['phase']!='train' or checkpoint['state']['step']!=0)
                if members is None:
                    if partial:
                        members=checkpoint['distributed']['members']
                        if len(members)>maximum:raise ValueError('Partial-epoch resume requires its original rank count')
                    else:members=[r['uuid'] for r in available if r['uuid'] not in controller['deferred']][:maximum]
                if not members or any(m not in by_uuid for m in members):
                    if time.monotonic()>=deadline:raise RuntimeError('No admitted visible GPU group within the bounded wait; batch/patch unchanged')
                    log.event('parallel.wait','等待获准空闲GPU；不改batch或patch',required=members,inventory=rows)
                    if not partial:members=None
                    time.sleep(min(10,max(0,deadline-time.monotonic())));continue
                if len(members)>1 and (sys.platform!='linux' or not torch.distributed.is_nccl_available()):
                    raise RuntimeError('Selected multi-GPU group requires Linux and a CUDA/NCCL PyTorch build')
                generation=uuid4().hex
                supply=supply_plan(snapshot(),settings,pool_loaders,len(members),controller['worker_cap'],controller.get('reserve_at_entry'))
                controller.setdefault('reserve_at_entry',supply['stage']['reserve_bytes'])
                write_json(controller_path,controller)
                job=dict(output=str(output),prepared=str(prepared),options=options,members=members,
                    ordinals=[by_uuid[m]['ordinal'] for m in members],supply=supply,generation=generation,
                    pressure_recoveries=controller['pressure_recoveries'],
                    rendezvous=(output/'control'/('rendezvous-'+generation)).as_uri())
                # file:// rendezvous belongs exclusively to this launch.
                jobfile=output/'control'/('job-'+generation+'.json');write_json(jobfile,job)
                write_json(output/'control/training_read_gate.json',dict(slots=max(1,min(2,sum(supply['workers']) or len(members)))))
                _remove_control(output,'stop.json','result.json','reconfigure.json','capacity-passed.json')
                log.event('parallel.launch','启动GPU训练组',requested_gpus=gpus,applied_world_size=len(members),
                    members=members,layout=supply['layout'],supply=supply,generation=generation)
                processes=[];owned={};streams=[]
                env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',
                    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                ledger=output/'control'/('owners-'+generation);ledger.mkdir()
                env.update(CHD_TRAIN_OWNER_DIR=str(ledger),CHD_TRAIN_NONCE=generation)
                for rank in range(len(members)):
                    folder=output/'ranks'/f'rank_{rank}';folder.mkdir(parents=True,exist_ok=True)
                    # Main rank streams to the user's terminal as well as its
                    # ordinary RunLog. Other rank stdout is separately retained.
                    stream=None if rank==0 else (folder/f'console-{generation}.log').open('a',encoding='utf-8')
                    if stream:streams.append(stream)
                    process=subprocess.Popen([interpreter_command(env),'-u',str(entry),'_train-rank','--job',str(jobfile),'--rank',str(rank)],
                        env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=sys.platform=='linux')
                    processes.append(process);remember_children(processes,owned)
                mapping=rows
                last_scan=time.monotonic();last_heartbeat=0;pending=None
                while any(p.poll() is None for p in processes):
                    remember_children(processes,owned)
                    learn_workers(ledger,generation,{p.pid for p in processes},owned)
                    failed=[p.returncode for p in processes if p.poll() not in (None,0)]
                    if failed or list((output/'control').glob(f'fatal-{generation}-*.json')):break
                    if stop[0]:
                        if stopping_at is None:stopping_at=time.monotonic();write_json(output/'control/stop.json',dict(requested=True))
                        if time.monotonic()-stopping_at>300:raise TimeoutError('Safe stop exceeded 300 seconds; retaining last committed checkpoint')
                    if time.monotonic()-last_heartbeat>=10:
                        log.event('parallel.heartbeat','GPU组运行中；详细步骤见training.log和各rank日志',
                                  members=members,layout=supply['layout'],pids=[p.pid for p in processes],generation=generation)
                        last_heartbeat=time.monotonic()
                    performance=_performance(output/'epoch-performance.jsonl',generation)
                    if elastic and not stop[0] and pending is None and len(members)>1 and expansion_regressed(controller.get('baseline'),performance,settings['batch_size']):
                        previous_members=controller['baseline']['previous_members']
                        controller['deferred']=list(set(controller['deferred'])|set(members)-set(previous_members))
                        pending=dict(generation=generation,reason='three post-warmup epochs slower by >25%',members=previous_members)
                        write_json(output/'control/reconfigure.json',pending)
                    if time.monotonic()-last_scan>=20:
                        last_scan=time.monotonic();live=query_live(mapping)
                        own_pids={p.pid for p in processes}
                        intruders={r['uuid']:list(set(r['external_pids'])-own_pids) for r in live
                                   if r['uuid'] in members and set(r['external_pids'])-own_pids}
                        if intruders:
                            log.event('parallel.ownership','所选GPU出现外部计算进程，请求全组安全暂停',external=intruders)
                            stop[0]=True
                        fresh=eligible(live)
                        candidates=[]
                        for row in fresh:
                            uuid=row['uuid']
                            if uuid in members or uuid in controller['deferred']:continue
                            if uuid in candidate_seen and time.monotonic()-candidate_seen[uuid]>=1:candidates.append(uuid)
                            candidate_seen.setdefault(uuid,time.monotonic())
                        fresh_ids={r['uuid'] for r in fresh}
                        candidate_seen={k:v for k,v in candidate_seen.items() if k in fresh_ids}
                        if elastic and not stop[0] and pending is None and len(members)<maximum and candidates:
                            expanded=members+candidates[:maximum-len(members)]
                            baseline=expansion_baseline(performance,settings['batch_size'],members)
                            pending=dict(generation=generation,reason='stable idle GPU available; join at full epoch only',members=expanded,baseline=baseline)
                            write_json(output/'control/reconfigure.json',pending)
                    time.sleep(.2)
                codes=[p.poll() for p in processes]
                learn_workers(ledger,generation,{p.pid for p in processes},owned)
                write_json(output/'control'/('exit-'+generation+'.json'),dict(codes=codes,pids=[p.pid for p in processes],generation=generation))
                close_owned(processes,owned);processes=[]
                for stream in streams:stream.close()
                streams=[]
                result_path=output/'control/result.json'
                result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
                if any(code!=0 for code in codes) or not result or result['generation']!=generation:
                    raise RuntimeError('Training rank group failed (exit codes '+str(codes)+'); last complete checkpoint retained; inspect rank console/training logs')
                if result['action'] in ('completed','stopped'):
                    return json.loads((output/'summary.json').read_text(encoding='utf-8'))
                if result['action']=='pressure':
                    if controller['pressure_recoveries']>=3:raise MemoryError('Three V2 group memory recoveries exhausted')
                    controller['pressure_recoveries']+=1
                    controller['worker_cap']=supply['applied_total_workers']//2
                    restored=restore_full_epoch(output/'checkpoints',signature)
                    log.event('parallel.pressure','主机压力：全组退出、缩减供数、重放未完成轮次',controller=controller,
                              restored_epoch=restored['epoch'],restored_attempts=restored['attempts'])
                elif result['action']=='reconfigure':
                    members=result['members'];controller['baseline']=result.get('baseline')
                    log.event('parallel.reconfigure','完整epoch检查点处重新准入GPU成员',decision=result)
                else:raise ValueError('Unknown group result action')
                write_json(controller_path,controller);deadline=time.monotonic()+wait_seconds
        except BaseException as error:
            if processes and 'ledger' in locals():
                learn_workers(ledger,generation,{p.pid for p in processes},owned)
            log.event('parallel.failed',str(error),traceback=traceback.format_exc());log.status('failed',error=str(error))
            TrainingLog(output).status('failed',stage='supervisor',error=str(error))
            raise
        finally:
            close_owned(processes,owned)
            for stream in streams:stream.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)
