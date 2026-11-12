"""Single-device STU training with explicit phase/cursor and full-state recovery.

An attempt is consumed after backward/scaler handling, even on AMP overflow.
Speculative CPU batches are never checkpointed as consumed. Unsafe exceptions
retain the previous committed checkpoint; SIGINT asks for a safe-point pause.
"""
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from ..observability.locking import run_lock
from ..preprocessing.shared_cache import protect_training
from ..observability.runlog import write_json
from ..acceleration.training_cpu import BatchPool,BatchQueue
from ..acceleration.preprocess_cpu import snapshot
from .contracts import recipe,make_signature
from .data import load_prepared,PatchSampler
from .model import topology,build_model
from .weights import load_pretrained,WEIGHT_SHA256
from .losses import DeepSupervisionLoss
from .resources import admit_device,transfer,capacity_probe
from .feedback import TrainingLog,MemoryMonitor
from .checkpoints import save_checkpoint,read_checkpoint,restore_checkpoint
from .code_snapshot import freeze_code


def initial_state():
    return dict(epoch=0,phase='train',step=0,attempts=0,updates=0,overflows=0,
        train_loss_sum=0.,train_batches=0,val_loss_sum=0.,val_batches=0,
        tp=[0]*7,fp=[0]*7,fn=[0]*7,best_val_loss=None,history=[],
        updates_by_cohort=dict(main=0,auxiliary=0))


def validate_state(state,settings):
    for key in ('epoch','step','attempts','updates','overflows'):
        if type(state[key]) is not int or state[key]<0:raise ValueError('Invalid checkpoint cursor: '+key)
    if state['phase'] not in ('train','validation') or state['epoch']>settings['epochs']:
        raise ValueError('Invalid checkpoint phase/epoch')
    steps=settings['train_steps_per_epoch'] if state['phase']=='train' else settings['validation_steps_per_epoch']
    if state['step']>=steps:raise ValueError('Checkpoint step outside its phase')
    expected=state['epoch']*settings['train_steps_per_epoch']
    expected+=state['step'] if state['phase']=='train' else settings['train_steps_per_epoch']
    if state['attempts']!=expected or state['updates']+state['overflows']!=state['attempts']:
        raise ValueError('Checkpoint attempts/updates/overflow cursor mismatch')
    if len(state['history'])!=state['epoch']:
        raise ValueError('Checkpoint completed-epoch history differs')
    if sum(state['updates_by_cohort'].values())!=state['updates']*settings['batch_size']:
        raise ValueError('Checkpoint successful-update coverage differs')


def summary_from_state(state,settings,dataset,pointer,recoveries=0,peak_rss=0):
    return dict(status='completed' if state['epoch']>=settings['epochs'] else 'stopped',scope=settings['scope'],
        epoch=state['epoch'],phase=state['phase'],step=state['step'],attempts=state['attempts'],
        updates=state['updates'],overflows=state['overflows'],updates_by_cohort=state['updates_by_cohort'],
        train_ids=dataset['train_ids'],val_ids=dataset['val_ids'],memory_pressure_recoveries=recoveries,
        peak_process_group_rss_bytes=peak_rss,checkpoint=pointer,
        metrics_scope='validation patches, not whole-volume test performance')


def repair_best_pointer(output,state):
    # A crash after committing the epoch but before writing best.json is
    # recoverable from the just-committed epoch state, before any next update.
    if state.get('best_epoch')==state['epoch']-1 and state['phase']=='train' and state['step']==0:
        pointer=json.loads((output/'checkpoints/latest.json').read_text(encoding='utf-8'))
        write_json(output/'checkpoints/best.json',pointer)


def optimizer_for(model,settings):
    return torch.optim.SGD(model.parameters(),lr=settings['initial_lr'],momentum=settings['momentum'],
        nesterov=settings['nesterov'],weight_decay=settings['weight_decay'])


def train_attempt(model,optimizer,scaler,criterion,batch,settings,device):
    data,targets,available=transfer(batch,device)
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type,enabled=settings['amp'] and device.type=='cuda'):
        outputs=model(data);loss=criterion(outputs,targets,available)
    if not bool(torch.isfinite(loss)):raise ValueError('Non-finite training loss; inspect data/precision')
    scale_before=scaler.get_scale()
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'])
    if not scaler.is_enabled() and not bool(torch.isfinite(norm)):
        raise ValueError('Non-finite FP32 gradient; optimizer was not updated')
    scaler.step(optimizer);scaler.update()
    overflow=scaler.get_scale()<scale_before
    # Logging floats is bounded; no full-logit histogram or arbitrary summed
    # half projection is introduced merely for status reporting.
    return dict(loss=float(loss.detach()),update_applied=not overflow,overflow=overflow,
        scaler_scale=scaler.get_scale(),gradient_norm=float(norm) if bool(torch.isfinite(norm)) else None)


@torch.no_grad()
def validate_attempt(model,criterion,batch,settings,device):
    data,targets,available=transfer(batch,device);model.eval()
    with torch.autocast(device.type,enabled=settings['amp'] and device.type=='cuda'):
        outputs=model(data);loss=criterion(outputs,targets,available)
    if not bool(torch.isfinite(loss)):raise ValueError('Non-finite validation loss')
    prediction=outputs[0].argmax(1);truth=targets[0][:,0]
    valid=(truth>=0)&(truth<8)
    tp=[];fp=[];fn=[]
    for label in range(1,8):
        predicted=(prediction==label)&valid;actual=(truth==label)&valid
        tp.append(int((predicted&actual).sum()));fp.append(int((predicted&~actual).sum()))
        fn.append(int((~predicted&actual).sum()))
    return dict(loss=float(loss),tp=tp,fp=fp,fn=fn)


def prune_checkpoints(folder):
    """Keep two recent generations plus best/final pointers, after commit only."""
    generations=sorted(folder.glob('state-*.pt'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    keep={p.name for p in generations[:2]}
    for name in ('latest.json','epoch.json','best.json','final.json'):
        if (folder/name).is_file():keep.add(json.loads((folder/name).read_text(encoding='utf-8'))['file'])
    for path in generations:
        if path.name not in keep:path.unlink()


@protect_training
def run_train(prepared,output,*,fold=0,weights=None,device='cuda:0',smoke=False,
              train_cases=(),val_cases=(),overrides=None,workers=None,resume=False,
              stop_after_steps=None,check_only=False):
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('This training release supports one CPU/CUDA process; multi-rank launch is not supported')
    settings=recipe(smoke,overrides)
    output=Path(output).resolve();prepared=Path(prepared).resolve(strict=True)
    if output==prepared or output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError('Training output and preprocessed input must be separate directories')
    settings['effective_precision']='fp16' if settings['amp'] and torch.device(device).type=='cuda' else 'fp32'
    if smoke and not train_cases and not val_cases:
        from .data import cached_smoke_ids
        train_cases,val_cases=cached_smoke_ids(prepared,fold,settings['include_auxiliary'])
    print('阶段：检查预处理来源、角色、完成标记、文件SHA和取块坐标',flush=True)
    dataset=load_prepared(prepared,fold,settings,train_cases=train_cases,val_cases=val_cases)
    architecture=topology(dataset['plan']['configurations']['3d_fullres']['spacing'],settings['patch_size'])
    settings['architecture']=architecture
    signature=make_signature(settings,dataset['identity'],WEIGHT_SHA256)
    # Refuse an incompatible resume BEFORE status, logs or checkpoints change.
    payload=None
    if output.exists() and any(output.iterdir()):
        if not resume:raise FileExistsError('Training output is not empty; use a new directory or --resume')
        if not (output/'signature.json').is_file() or json.loads((output/'signature.json').read_text(encoding='utf-8'))!=signature:
            raise ValueError('Training resume contract differs; old output was not changed')
        payload=read_checkpoint(output/'checkpoints',signature)
        validate_state(payload['state'],settings)
        freeze_code(output,signature,create=False)
    elif resume:raise ValueError('Resume requires an existing signed training run')
    if check_only:
        return dict(status='checked',train_ids=dataset['train_ids'],val_ids=dataset['val_ids'],
            auxiliary=sum(dataset['rows'][c]['cohort_group']=='auxiliary' for c in dataset['train_ids']),
            coordinate_adapters=dataset['identity']['adapters'],recipe=settings,signature_identity=signature['identity'],
            weights_file_checked=False,gpu_capacity_checked=False)
    if payload is not None and payload['state']['epoch']>=settings['epochs']:
        # Completion metadata is repairable even when the process died after
        # the final epoch commit, and does not require GPU admission or weights.
        with run_lock(output/'.training.lock'):
            payload=read_checkpoint(output/'checkpoints',signature)
            state=payload['state'];validate_state(state,settings)
            pointer=json.loads((output/'checkpoints/latest.json').read_text(encoding='utf-8'))
            repair_best_pointer(output,state)
            write_json(output/'checkpoints/final.json',pointer)
            summary=summary_from_state(state,settings,dataset,pointer,state.get('supply_recoveries',0))
            write_json(output/'history.json',state['history']);write_json(output/'summary.json',summary)
            TrainingLog(output).status('completed',stage='done',summary=summary)
            return summary
    if payload is None and not weights:raise ValueError('Author weights required; random initialization is not a silent fallback')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    random.seed(settings['seed']);np.random.seed(settings['seed']);torch.manual_seed(settings['seed'])
    device,admission=admit_device(device)
    pool=monitor=None;handlers={};stop_requested=False
    def stop_at_safe_point(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    with run_lock(output/'.training.lock'):
        # Another invocation may have committed a newer generation after the
        # initial read. The lock protects a second validation before any writes.
        if (output/'signature.json').is_file():
            if not resume:raise FileExistsError('Another run created this output; no files changed')
            if json.loads((output/'signature.json').read_text(encoding='utf-8'))!=signature:
                raise ValueError('Resume contract changed while acquiring the run lock')
            payload=read_checkpoint(output/'checkpoints',signature)
            validate_state(payload['state'],settings)
        output.mkdir(parents=True,exist_ok=True);log=TrainingLog(output)
        try:
            bootstrap=freeze_code(output,signature,create=payload is None)
            write_json(output/'signature.json',signature)
            write_json(output/'effective_config.json',dict(recipe=settings,prepared=str(prepared),fold=fold,
                output=str(output),train_ids=dataset['train_ids'],val_ids=dataset['val_ids'],requested_workers=workers,
                stop_after_steps=stop_after_steps,world_size=1,admission=admission))
            write_json(output/'input_adapters.json',dataset['identity']['adapters'])
            log.event('training.start',f"STU-Net-S {settings['scope']}；train {len(dataset['train_ids'])}例、val {len(dataset['val_ids'])}例；"
                f"B{settings['batch_size']}、patch {settings['patch_size']}、{settings['epochs']}轮",recipe=settings,admission=admission)
            log.status(stage='model_initialization')
            model=build_model(architecture).to(device)
            if payload is None:
                provenance=load_pretrained(model,weights)
                write_json(output/'pretrained_provenance.json',provenance)
                log.event('weights.loaded','作者骨干逐tensor校验加载，8类输出头重新初始化',**provenance)
            criterion=DeepSupervisionLoss(settings['batch_dice'],settings['dice_smooth'])
            optimizer=optimizer_for(model,settings)
            scaler=torch.amp.GradScaler('cuda',enabled=settings['amp'] and device.type=='cuda')
            state=initial_state() if payload is None else restore_checkpoint(payload,model,optimizer,scaler,device)
            if payload is not None:repair_best_pointer(output,state)
            # Startup objects and unconsumed loaders are outside the saved RNG
            # trajectory; restore_checkpoint is after model/optimizer creation.
            loaders=dict(train=PatchSampler(dataset['rows'],dataset['train_ids'],settings,architecture,True),
                         validation=PatchSampler(dataset['rows'],dataset['val_ids'],settings,architecture,False))
            applied_workers=workers
            if state.get('supply_recoveries',0):
                applied_workers=min(workers,state['supply_worker_cap']) if workers is not None else state['supply_worker_cap']
            pool=BatchPool(loaders,settings,output,log,applied_workers,bootstrap,signature['versions'])
            pool.recoveries=state.get('supply_recoveries',0)
            reserve=state.setdefault('host_reserve_bytes',pool.stage['reserve_bytes'])
            monitor=MemoryMonitor(output,log,reserve,device);monitor.start()
            for sig in (signal.SIGINT,signal.SIGTERM):
                handlers[sig]=signal.getsignal(sig);signal.signal(sig,stop_at_safe_point)
            if state['epoch']>=settings['epochs']:
                log.status('completed',stage='done',attempts=state['attempts'],updates=state['updates'])
                return dict(status='completed',**state)
            phase=state['phase'];total=settings['train_steps_per_epoch'] if phase=='train' else settings['validation_steps_per_epoch']
            queue=BatchQueue(pool,phase,state['epoch'],total,state['step'])
            log.status(stage='capacity_probe',epoch=state['epoch'],step=state['step'])
            probe=capacity_probe(model,criterion,queue.peek(),settings,device,admission,optimizer)
            write_json(output/'capacity_probe.json',probe);log.event('capacity.passed','真实所选尺寸前向/反向及动量内存准入通过；未更新或消费批次',**probe)
            if payload is None:save_checkpoint(output/'checkpoints',model,optimizer,scaler,state,signature,device)
            while state['epoch']<settings['epochs']:
                if monitor.error:raise monitor.error
                if stop_requested or (stop_after_steps is not None and state['attempts']>=stop_after_steps):break
                if monitor.pressure.is_set():
                    # CPU prefetch is unconsumed and replayable. Release it and
                    # the owned worker resident sets BEFORE cloning model/SGD
                    # state onto host RAM for the safe checkpoint.
                    queue=None;pool.close(force=True)
                    save_checkpoint(output/'checkpoints',model,optimizer,scaler,state,signature,device)
                    pool.shrink()
                    state.update(supply_recoveries=pool.recoveries,supply_worker_cap=pool.workers)
                    save_checkpoint(output/'checkpoints',model,optimizer,scaler,state,signature,device)
                    if snapshot()['available_memory_bytes']<monitor.reserve:
                        raise MemoryError('Host memory remains below entry reserve after bounded supply shrink')
                    monitor.samples=0;monitor.pressure.clear()
                    phase=state['phase'];total=settings['train_steps_per_epoch'] if phase=='train' else settings['validation_steps_per_epoch']
                    queue=BatchQueue(pool,phase,state['epoch'],total,state['step'])
                phase=state['phase']
                log.status(stage=phase,epoch=state['epoch'],step=state['step'],attempts=state['attempts'],updates=state['updates'])
                began=time.monotonic();batch=queue.peek();wait=time.monotonic()-began
                began=time.monotonic()
                if phase=='train':
                    lr=settings['initial_lr']*(1-state['epoch']/settings['epochs'])**settings['poly_power']
                    for group in optimizer.param_groups:group['lr']=lr
                    result=train_attempt(model,optimizer,scaler,criterion,batch,settings,device)
                    state['attempts']+=1;state['updates']+=int(result['update_applied'])
                    state['overflows']+=int(result['overflow']);state['train_loss_sum']+=result['loss'];state['train_batches']+=1
                    if result['update_applied']:
                        for case in batch['cases']:state['updates_by_cohort'][case['cohort_group']]+=1
                else:
                    result=validate_attempt(model,criterion,batch,settings,device)
                    state['val_loss_sum']+=result['loss'];state['val_batches']+=1
                    for key in ('tp','fp','fn'):state[key]=[a+b for a,b in zip(state[key],result[key])]
                state['step']+=1;queue.commit()
                log.event('training.step',f"epoch {state['epoch']+1}/{settings['epochs']} {phase} {state['step']}/{queue.total}；"
                    f"loss {result['loss']:.5f}"+(f"；实际更新 {state['updates']}，溢出 {state['overflows']}" if phase=='train' else ''),
                    epoch=state['epoch'],phase=phase,step=state['step'],attempts=state['attempts'],
                    cases=batch['cases'],supply=batch['supply'],wait_seconds=wait,compute_seconds=time.monotonic()-began,
                    learning_rate=optimizer.param_groups[0]['lr'],**result)
                del batch
                if state['step']==queue.total:
                    if phase=='train':
                        state.update(phase='validation',step=0)
                        queue=BatchQueue(pool,'validation',state['epoch'],settings['validation_steps_per_epoch'])
                    else:
                        dice=[2*t/(2*t+f+n) if 2*t+f+n else None for t,f,n in zip(state['tp'],state['fp'],state['fn'])]
                        summary=dict(epoch=state['epoch'],train_loss=state['train_loss_sum']/state['train_batches'],
                            val_loss=state['val_loss_sum']/state['val_batches'],patch_dice=dice,
                            mean_patch_dice=float(np.mean([x for x in dice if x is not None])) if any(x is not None for x in dice) else None,
                            attempts=state['attempts'],updates=state['updates'],overflows=state['overflows'])
                        improved=state['best_val_loss'] is None or summary['val_loss']<state['best_val_loss']
                        if improved:state.update(best_val_loss=summary['val_loss'],best_epoch=state['epoch'])
                        state['history'].append(summary)
                        state.update(epoch=state['epoch']+1,phase='train',step=0,train_loss_sum=0.,train_batches=0,
                                     val_loss_sum=0.,val_batches=0,tp=[0]*7,fp=[0]*7,fn=[0]*7)
                        pointer=save_checkpoint(output/'checkpoints',model,optimizer,scaler,state,signature,device)
                        if improved:write_json(output/'checkpoints/best.json',pointer)
                        write_json(output/'history.json',state['history']);prune_checkpoints(output/'checkpoints')
                        log.event('training.epoch','本轮训练和patch验证完成，完整状态已原子保存',**summary)
                        if state['epoch']<settings['epochs']:queue=BatchQueue(pool,'train',state['epoch'],settings['train_steps_per_epoch'])
            pointer=save_checkpoint(output/'checkpoints',model,optimizer,scaler,state,signature,device)
            completed=state['epoch']>=settings['epochs']
            if completed:write_json(output/'checkpoints/final.json',pointer)
            prune_checkpoints(output/'checkpoints')
            summary=summary_from_state(state,settings,dataset,pointer,pool.recoveries,monitor.peak_group_rss)
            write_json(output/'summary.json',summary);log.status(summary['status'],stage='done',summary=summary)
            return summary
        except BaseException as error:
            log.event('training.failed',str(error),traceback=traceback.format_exc())
            log.status('failed',stage='error',error=str(error))
            raise
        finally:
            if monitor:monitor.close()
            if pool:pool.close(force=True)
            for sig,handler in handlers.items():signal.signal(sig,handler)
