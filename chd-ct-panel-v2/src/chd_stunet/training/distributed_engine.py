"""One immutable-code rank: fixed local share, collective steps, epoch commits.

The supervisor owns the run lock and process lifecycle. All ranks execute the
same collective order; a rank exception fails the group rather than leaving
healthy ranks updating alone. Admission probes consume no sample or update.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..acceleration.training_cpu import BatchPool,BatchQueue
from ..observability.runlog import write_json
from .contracts import recipe,make_signature
from .data import PatchSampler,load_prepared,cached_smoke_ids
from .model import topology,build_model
from .weights import WEIGHT_SHA256,load_pretrained
from .engine import initial_state,validate_state,optimizer_for,prune_checkpoints,repair_best_pointer,summary_from_state
from .losses import DeepSupervisionLoss
from .resources import admit_device,capacity_probe
from .checkpoints import read_checkpoint,restore_checkpoint
from .code_snapshot import freeze_code
from .feedback import TrainingLog,MemoryMonitor
from .parallel_policy import slots_for
from .collectives import any_rank,all_objects
from .rank_runtime import restore_rank_rng,commit,train_attempt,validate_attempt


def prepare_contract(prepared,options):
    settings=recipe(options.get('smoke',False),options.get('overrides'))
    settings.update(execution_protocol='global-slot-rng-v2',supported_world_size=settings['batch_size'],
                    effective_precision='fp16' if settings['amp'] else 'fp32')
    train,val=options.get('train_cases',()),options.get('val_cases',())
    if settings['scope']=='smoke' and not train and not val:
        train,val=cached_smoke_ids(prepared,options['fold'],settings['include_auxiliary'])
    dataset=load_prepared(prepared,options['fold'],settings,train_cases=train,val_cases=val)
    settings['architecture']=topology(dataset['plan']['configurations']['3d_fullres']['spacing'],settings['patch_size'])
    return settings,dataset,make_signature(settings,dataset['identity'],WEIGHT_SHA256)


def run_rank(job_path,rank):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    job=json.loads(Path(job_path).read_text(encoding='utf-8'))
    output=Path(job['output']);members=job['members'];world=len(members)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rankdir=output/'ranks'/f'rank_{rank}';rankdir.mkdir(parents=True,exist_ok=True)
    (rankdir/'logs').mkdir(exist_ok=True)
    log=TrainingLog(output if rank==0 else rankdir);pool=monitor=None
    device=torch.device('cuda',job['ordinals'][rank])
    try:
        log.status(stage='gpu_admission',rank=rank,world_size=world,device=str(device))
        log.event('parallel.admission',f'rank {rank} 开始GPU准入：{device}',rank=rank,device=str(device))
        device,admission=admit_device(str(device))
        if admission['uuid'].lower()!=members[rank].lower():raise ValueError('Visible ordinal/physical UUID changed before rank admission')
        log.status(stage='process_group')
        log.event('parallel.device',f'rank {rank} 通信组绑定 {device}；等待同组初始化',
                  rank=rank,device=str(device),uuid=admission['uuid'],world_size=world,
                  torch_version=str(torch.__version__),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        # rank is a group position, not a CUDA ordinal: resource selection can
        # assign rank 1 to visible cuda:5. PyTorch 2.15 NCCL otherwise binds to
        # LOCAL_RANK/rank and ignores set_device(). Pass the admitted device,
        # using the SAME visible namespace as model.to() and DDP.device_ids.
        # One-rank Gloo remains unbound (its collectives use CPU tensors).
        binding={'device_id':device} if world>1 else {}
        dist.init_process_group('nccl' if world>1 else 'gloo',init_method=job['rendezvous'],
                                rank=rank,world_size=world,timeout=timedelta(seconds=300),**binding)
        log.event('parallel.initialized',f'rank {rank} 通信组初始化完成：{device}',rank=rank,device=str(device))
        log.status(stage='input_contract')
        settings,dataset,signature=prepare_contract(job['prepared'],job['options'])
        expected=json.loads((output/'signature.json').read_text(encoding='utf-8'))
        if signature!=expected:raise ValueError('Frozen rank data/code/environment contract differs')
        bootstrap=freeze_code(output,signature,create=False)
        payload=read_checkpoint(output/'checkpoints',signature) if (output/'checkpoints/latest.json').is_file() else None
        random.seed(settings['seed']);np.random.seed(settings['seed']);torch.manual_seed(settings['seed'])
        model=build_model(settings['architecture']).to(device)
        optimizer=optimizer_for(model,settings)
        scaler=torch.amp.GradScaler('cuda',enabled=settings['amp'])
        state=initial_state()
        if payload is None:
            provenance=load_pretrained(model,job['options']['weights'])
            if rank==0:write_json(output/'pretrained_provenance.json',provenance)
        else:
            state=restore_checkpoint(payload,model,optimizer,scaler,device)
            validate_state(state,settings)
            restore_rank_rng(payload,members,rank,device,settings['seed'])
        if rank==0 and payload is not None:repair_best_pointer(output,state)
        # Last zero-weight deep supervision head has intentionally no gradient.
        # find_unused_parameters retains the author's output/DS structure.
        if world>1:
            log.status(stage='ddp_initialization')
            log.event('parallel.ddp',f'rank {rank} 检查DDP模型与通信设备：{device}',rank=rank,device=str(device))
            model=DDP(model,device_ids=[device.index],output_device=device.index,
                      broadcast_buffers=False,find_unused_parameters=True)
        criterion=DeepSupervisionLoss(settings['batch_dice'],settings['dice_smooth'])
        slots=slots_for(settings['batch_size'],world,rank)
        loaders={phase:PatchSampler(dataset['rows'],dataset[key],settings,settings['architecture'],phase=='train',slots)
                 for phase,key in (('train','train_ids'),('validation','val_ids'))}
        local_settings=dict(settings,batch_size=len(slots))
        supply=job['supply'];stage=supply['ranks'][rank]
        pool=BatchPool(loaders,local_settings,rankdir,log,supply['workers'][rank],bootstrap,signature['versions'],stage,
                       read_gate=output/'control/training_read_gate.json')
        monitor=MemoryMonitor(rankdir,log,supply['stage']['reserve_bytes'],device);monitor.start()
        phase=state['phase'];total=settings['train_steps_per_epoch'] if phase=='train' else settings['validation_steps_per_epoch']
        queue=BatchQueue(pool,phase,state['epoch'],total,state['step'])
        log.event('parallel.rank','多卡分担固定全局batch，先执行真实尺寸准入',rank=rank,members=members,
                  global_batch=settings['batch_size'],local_batch=len(slots),global_slots=slots,admission=admission)
        probe=capacity_probe(model,criterion,queue.peek(),settings,device,admission,optimizer)
        log.event('capacity.passed','本rank真实尺寸容量准入通过，未更新或消费样本',rank=rank,**probe)
        write_json(rankdir/'capacity_probe.json',probe)
        dist.barrier()
        if rank==0:write_json(output/'control/capacity-passed.json',dict(generation=job['generation'],members=members))
        if payload is None:commit(output/'checkpoints',model,optimizer,scaler,state,signature,device,members)
        epoch_began=time.monotonic();full_epoch=state['phase']=='train' and state['step']==0
        while state['epoch']<settings['epochs']:
            stop=(output/'control/stop.json').exists()
            stop_steps=job['options'].get('stop_after_steps')
            stop=stop or (stop_steps is not None and state['attempts']>=stop_steps)
            if any_rank(stop,device):break
            if any_rank(monitor.error is not None,device):raise RuntimeError('Rank memory monitor failed; evidence in rank logs')
            if any_rank(monitor.pressure.is_set(),device):
                # Do not save the incomplete epoch. Parent closes this group,
                # shrinks its TOTAL budget and reloads the last full commit.
                queue=None;pool.close(force=True);dist.barrier()
                if rank==0:write_json(output/'control/result.json',dict(action='pressure',generation=job['generation']))
                return
            phase=state['phase'];log.status(stage=phase,rank=rank,world_size=world,
                epoch=state['epoch'],step=state['step'],attempts=state['attempts'],updates=state['updates'])
            began=time.monotonic();batch=queue.peek();wait=time.monotonic()-began;began=time.monotonic()
            if phase=='train':
                lr=settings['initial_lr']*(1-state['epoch']/settings['epochs'])**settings['poly_power']
                for group in optimizer.param_groups:group['lr']=lr
                result=train_attempt(model,optimizer,scaler,criterion,batch,settings,device)
                state['attempts']+=1;state['updates']+=int(result['update_applied']);state['overflows']+=int(result['overflow'])
                state['train_loss_sum']+=result['loss'];state['train_batches']+=1
                if result['update_applied']:
                    groups=all_objects([c['cohort_group'] for c in batch['cases']])
                    for group in groups:
                        for cohort in group:state['updates_by_cohort'][cohort]+=1
            else:
                result=validate_attempt(model,criterion,batch,settings,device)
                state['val_loss_sum']+=result['loss'];state['val_batches']+=1
                for key in ('tp','fp','fn'):state[key]=[a+b for a,b in zip(state[key],result[key])]
            state['step']+=1;queue.commit()
            log.event('training.step',f"rank{rank} epoch{state['epoch']+1}/{settings['epochs']} {phase} {state['step']}/{total} loss {result['loss']:.5f}",
                rank=rank,epoch=state['epoch'],phase=phase,step=state['step'],attempts=state['attempts'],updates=state['updates'],
                global_batch=settings['batch_size'],local_batch=len(slots),cases=batch['cases'],supply=batch['supply'],
                wait_seconds=wait,compute_seconds=time.monotonic()-began,learning_rate=optimizer.param_groups[0]['lr'],**result)
            del batch
            if state['step']!=total:continue
            if phase=='train':
                state.update(phase='validation',step=0);total=settings['validation_steps_per_epoch']
                queue=BatchQueue(pool,'validation',state['epoch'],total);continue
            dice=[2*t/(2*t+f+n) if 2*t+f+n else None for t,f,n in zip(state['tp'],state['fp'],state['fn'])]
            row=dict(epoch=state['epoch'],train_loss=state['train_loss_sum']/state['train_batches'],
                     val_loss=state['val_loss_sum']/state['val_batches'],patch_dice=dice,
                     attempts=state['attempts'],updates=state['updates'],overflows=state['overflows'])
            improved=state['best_val_loss'] is None or row['val_loss']<state['best_val_loss']
            if improved:state.update(best_val_loss=row['val_loss'],best_epoch=state['epoch'])
            state['history'].append(row)
            state.update(epoch=state['epoch']+1,phase='train',step=0,train_loss_sum=0.,train_batches=0,
                         val_loss_sum=0.,val_batches=0,tp=[0]*7,fp=[0]*7,fn=[0]*7)
            pointer=commit(output/'checkpoints',model,optimizer,scaler,state,signature,device,members)
            if rank==0:
                if improved:write_json(output/'checkpoints/best.json',pointer)
                write_json(output/'history.json',state['history']);prune_checkpoints(output/'checkpoints')
            dist.barrier()
            # Include validation, checkpoint, feedback and synchronization in
            # wall time. A resumed partial epoch is never a comparable sample.
            seconds=max(all_objects(time.monotonic()-epoch_began))
            performance=dict(epoch=state['epoch']-1,seconds=seconds,global_batch=settings['batch_size'],
                             members=members,generation=job['generation'],complete_from_boundary=full_epoch)
            if rank==0:
                with (output/'epoch-performance.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(performance)+'\n')
                log.event('training.epoch','全组本轮完成，模型/优化器/scaler/逐rank RNG已原子保存',**row,performance=performance)
            dist.barrier()
            request=output/'control/reconfigure.json'
            change=json.loads(request.read_text(encoding='utf-8')) if request.exists() else None
            changes=all_objects(change)
            change=changes[0]
            if change and change['generation']==job['generation'] and state['epoch']<settings['epochs']:
                if rank==0:write_json(output/'control/result.json',dict(change,action='reconfigure'))
                return
            epoch_began=time.monotonic();full_epoch=True;total=settings['train_steps_per_epoch']
            if state['epoch']<settings['epochs']:queue=BatchQueue(pool,'train',state['epoch'],total)
        pointer=commit(output/'checkpoints',model,optimizer,scaler,state,signature,device,members)
        rank_peaks=all_objects(monitor.peak_group_rss)
        if rank==0:
            completed=state['epoch']>=settings['epochs']
            if completed:write_json(output/'checkpoints/final.json',pointer)
            prune_checkpoints(output/'checkpoints')
            summary=summary_from_state(state,settings,dataset,pointer)
            summary.update(world_size=world,members=members,layout=job['supply']['layout'])
            summary.update(memory_pressure_recoveries=job['pressure_recoveries'],
                           peak_process_group_rss_bytes=None,peak_rank_tree_rss_bytes=rank_peaks)
            write_json(output/'summary.json',summary)
            log.status(summary['status'],stage='done',epoch=state['epoch'],phase=state['phase'],step=state['step'],
                       attempts=state['attempts'],updates=state['updates'],summary=summary)
            write_json(output/'control/result.json',dict(action=summary['status'],generation=job['generation']))
    except BaseException as error:
        write_json(output/'control'/f'fatal-{job["generation"]}-{rank}.json',
                   dict(rank=rank,stage=log.context.get('stage'),device=str(device),
                        error=str(error),traceback=traceback.format_exc()))
        log.event('training.failed',str(error),rank=rank,traceback=traceback.format_exc())
        log.status('failed',stage='error',error=str(error))
        raise
    finally:
        if monitor:monitor.close()
        if pool:pool.close(force=True)
        if dist.is_initialized():dist.destroy_process_group()
