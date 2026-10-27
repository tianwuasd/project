"""Bounded resource measurements and full-epoch membership changes, no parent CUDA."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

from server_data import read_json,write_json,digest


def release_devices(uuids,*,wait=False,final=False):
    uuids=set(uuids)
    if uuids or final:
        path=Path(os.environ['VISTA_GPU_LEASES'])
        lease=read_json(path) if path.exists() else {}
        generation=lease.get('generation')
        if final and not generation:raise ValueError('GPU发现生命周期缺少本次运行身份，未开始CPU收尾')
        request_path=path.with_name('gpu-release-request.json')
        if not final and request_path.exists():
            previous=read_json(request_path)
            if (previous.get('generation')==generation and previous.get('id')!=lease.get('release_ack')
                    and not previous.get('final')):
                uuids.update(previous.get('gpu_uuids',[]))
        pending=[]
        if wait:
            from gpu_fleet import verified_keepers
            pending=[r for uuid,r in verified_keepers(os.environ['VISTA_GPU_LEASES']).items() if uuid in uuids]
        request_id=str(time.time_ns())
        write_json(request_path,
            dict(id=request_id,generation=generation,gpu_uuids=list(uuids),final=final,
                reason='GPU phase complete; discovery suspended for this invocation' if final else
                    'actual pressure, fault or incompatible capacity; retry after10minutes'))
        if wait:
            from task_runtime import alive
            deadline=time.monotonic()+30
            while True:
                exited=not any(alive(record) for record in pending)
                current=read_json(path) if final and path.exists() else {}
                acknowledged=(not final or (current.get('generation')==generation and
                    current.get('release_ack')==request_id and current.get('discovery_enabled') is False))
                if exited and acknowledged:break
                if time.monotonic()>=deadline:raise TimeoutError('GPU保留进程尚未确认退出；未开始CPU收尾')
                time.sleep(.2)


def devices(wait=True):
    from gpu_fleet import verified_keepers
    path=os.environ.get('VISTA_GPU_LEASES')
    if not path:raise ValueError('自动协同运行缺少受管GPU保留组')
    while True:
        record=read_json(path) if Path(path).exists() else {}
        if record.get('error'):raise RuntimeError('GPU自动探测失败：'+record['error'])
        result=list(verified_keepers(path))
        if result or not wait:return result
        print('等待实际空闲GPU；每1秒重新核对获准范围。',flush=True)
        time.sleep(1)


def identity(work,fold,method):
    work=Path(work)
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    prep=work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}"
    from evaluation_gpu import query
    hardware=sorted({tuple(row[1:]) for row in query('uuid,name,memory.total,driver_version')
                     if row[0] in devices()})
    return dict(fold=fold,method=method,plan=digest(prep/'WholeheartBaselinePlans.json'),
        splits=digest(work/'04_data/manifests/splits_v1.json'),
        source_labels=digest(work/f'04_data/manifests/{fold}_source_labels.json'),
        hardware=[list(row) for row in hardware],
        environment={p:digest(work/'06_configs'/p) for p in ['pip_freeze_training.txt','conda_explicit_training.txt']},
        code={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def topology(gpus):
    """Equivalent replacements are compatible only with measured type counts."""
    from evaluation_gpu import query
    profiles={row[0]:row[1:] for row in query('uuid,name,memory.total,driver_version')}
    if any(gpu not in profiles for gpu in gpus):raise ValueError('测量拓扑的硬件身份不可核验')
    return sorted(profiles[gpu] for gpu in gpus)


def coverage_matches(coverage,gpus):
    if not isinstance(coverage,list) or not coverage:return False
    current=topology(gpus)
    return any(isinstance(row,dict) and row.get('status')=='passed' and row.get('hardware')==current
        and row.get('world_size')==len(gpus) and isinstance(row.get('effective_cpu'),dict)
        and isinstance(row.get('local_batch_size'),int) and row['local_batch_size']>0 for row in coverage)


def equivalent_devices(available,hardware):
    from evaluation_gpu import query
    profiles={row[0]:row[1:] for row in query('uuid,name,memory.total,driver_version')}
    result=[]
    for profile in hardware:
        matching=next((gpu for gpu in available if gpu not in result and profiles.get(gpu)==profile),None)
        if matching is None:return None
        result.append(matching)
    return result


def trial(work,fold,method,gpus,local_batch,resource,run_kind,*,output=None,checkpoint=None,execution=None):
    from ddp_runtime import launch_ranks
    from runtime_observations import event,failure
    work=Path(work)
    ready,_,excluded=admit(gpus)
    if ready!=list(gpus):raise ResourceAdmissionError(excluded)
    output=Path(output) if output else work/f'07_experiments/tuning/{fold}_{method}_{time.time_ns()}'
    output.mkdir(parents=True,exist_ok=False)
    args=dict(workspace=str(work),fold=fold,method=method,gpu_uuids=list(gpus),
        global_batch_size=local_batch*len(gpus),resource=resource,run_kind=run_kind,
        operation='trial',output=str(output),checkpoint=checkpoint,execution=execution or dict(mode='performance',compile=True))
    event(work,'tuning.started',configuration=args,output=str(output))
    try:
        records=launch_ranks('cooperative_training:run_rank',args,gpus,output/'ranks',timeout_seconds=600,
            init_timeout_seconds=120)
        value=records[0]['result']
        value.update(total_cpu_slots=resource['total_cpu_slots'],effective_cpu=resource,
            gpu_uuids=list(gpus),output=str(output),rank_logs=[r['log_dir'] for r in records])
        write_json(output/'trial.json',value)
        event(work,'tuning.passed',trial=str(output/'trial.json'),seconds=value['seconds'])
        return value
    except BaseException as error:
        failure(output,'capacity_or_tuning',error,configuration=args)
        event(work,'tuning.failed',output=str(output),error=str(error))
        raise


class ResourceAdmissionError(RuntimeError):
    def __init__(self, exclusions):
        super().__init__('GPU admission changed: '+str(exclusions))
        self.exclusions=exclusions


def admit(gpus):
    """Fresh external PID/visibility/capacity check, only after old ranks exit."""
    from evaluation_gpu import snapshot,allowed_ids,idle_reason
    rows=snapshot()
    settings=dict(allowed_gpus=None,min_free_mib=6000,max_utilization_percent=10,
                  planner_gpu_memory_gib=float(os.environ.get('WHOLE_HEART_GPU_MEMORY','20')))
    allowed=allowed_ids(rows,settings);by_uuid={r['uuid']:r for r in rows}
    ready=[];usable={};excluded=[]
    for uuid in gpus:
        row=by_uuid.get(uuid)
        reason=('outside inherited visibility' if uuid not in allowed else
                'GPU no longer visible' if row is None else idle_reason(row,settings))
        if reason:excluded.append(dict(gpu_uuid=uuid,reason=reason));continue
        ready.append(uuid)
        usable[uuid]=max(0,min(int(settings['planner_gpu_memory_gib']*2**30),int(row['free_mib']*2**20-max(2*2**30,.1*row['total_mib']*2**20))))
    return ready,usable,excluded


class StageCapacityError(ValueError):
    def __init__(self,policy):
        super().__init__('; '.join(policy['reasons']));self.policy=policy


def stage_config(snapshot,patch,local_batch,world,observed=None):
    import math
    from live_resources import training_policy
    from batch_pool import shared_free_bytes
    # Unknown source/augmentation uses a labelled conservative prior. A natural
    # capacity run replaces payload bytes with actual data + all DS targets.
    payload=(observed or {}).get('payload_bytes_per_sample')
    batch=math.ceil(payload*local_batch) if payload else math.prod(patch)*local_batch*64
    private=max(2*math.prod(patch)*local_batch*64,math.ceil((observed or {}).get('private_bytes_per_sample',0)*local_batch))
    snapshot=dict(snapshot,worker_private_bytes=private)
    result=training_policy(snapshot,world,batch,history=dict(batch_wait_seconds=5.04,compute_seconds=.77))
    if result['status']=='blocked':raise StageCapacityError(result)
    result.update(batch_bytes=batch,world_size=world,local_batch_size=local_batch,
                  global_batch_size=local_batch*world,allocation_snapshot=snapshot,
                  original_cpu_affinity=snapshot.get('cpu_ids',snapshot.get('affinity_cpus',[])),
                  memory_components=dict(payload_bytes_per_batch=batch,
                      private_augmentation_bytes_per_worker=private,
                      mapped_source_bytes=snapshot.get('source_shared_bytes',0),
                      main_and_pinned_bytes_per_rank=3*batch,
                      estimate_scope='observed payload; private=max(coarse patch prior, observed private resident scaled by batch)' if payload else
                          'unknown coarse patch voxels x64; not a measured RAM fact',
                      worker_private_observation_bytes_per_sample=(observed or {}).get('private_bytes_per_sample')),
                  demand_evidence='historical A=5.04 D=0.77 estimate, not current measurement')
    free=shared_free_bytes();remaining=free
    for row in result['rank_settings']:
        row['batch_bytes']=batch
        for queue,key in [('prefetch','max_shared_bytes'),('val_prefetch','val_max_shared_bytes')]:
            request=row[queue]*batch
            # Windows has no /dev/shm gate; Linux unknown fails safely inside
            # allocator. Dividing here bounds the sum of both resident pools.
            cap=request if remaining is None else min(request,max(0,remaining))
            row[key]=cap
            if remaining is not None:remaining-=cap
    result['shared_transport_budget_bytes']=sum(r['max_shared_bytes']+r['val_max_shared_bytes'] for r in result['rank_settings'])
    result['shared_memory_available_bytes']=free
    result['transport']='shared_memory_with_bounded_byte_pipe_fallback'
    result['estimated_shm_bytes']=result['shared_transport_budget_bytes']
    return result


def observed_payload(result):
    sizes=[];private=[]
    for row in result.get('rank_results',[]):
        for obs in row.get('supply_transfer',{}).values():
            size=obs.get('source',{}).get('peak_payload_bytes',0)
            local=row.get('local_batch_size',result['local_batch_size'])
            if size:sizes.append(size/local)
            for worker in obs.get('source',{}).get('workers',[]):
                if worker.get('worker_private_bytes') is not None:private.append(worker['worker_private_bytes']/local)
    return dict(payload_bytes_per_sample=max(sizes),private_bytes_per_sample=max(private,default=0)) if sizes else None


def infer_batch(result,usable):
    from training_execution import capacity_batch
    proven=result['local_batch_size'];ranks=[]
    for row in result.get('rank_results',[]):
        facts=row.get('memory_facts',{});uuid=row.get('gpu_uuid')
        if uuid not in usable:continue
        ranks.append(dict(gpu_uuid=uuid,**capacity_batch(proven_batch=proven,usable_bytes=usable[uuid],
            fixed_bytes=facts.get('fixed_bytes'),peak_allocated_bytes=facts.get('peak_allocated_bytes'),
            peak_reserved_bytes=facts.get('peak_reserved_bytes'),optimizer_initialized=facts.get('optimizer_initialized',False))))
    complete=len(ranks)==len(usable) and bool(ranks)
    return dict(local_batch_size=min(r['candidate_batch'] for r in ranks) if complete else proven,
                ranks=ranks,status='inferred_requires_validation' if complete else 'unknown_keeps_proven',
                usable_scope='min(frozen planner budget, fresh free minus one reservation max(2GiB,10% physical VRAM))')


def tune(work,fold,method,run_kind,gpus=None,*,max_local_batch=None,execution=None,reduction_budget=3):
    """Compatibility entry: one direct policy, capacity observation and confirmation."""
    from task_resources import probe
    from runtime_observations import event
    work=Path(work)
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    prep=work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}"
    config=read_json(prep/'WholeheartBaselinePlans.json')['configurations']['3d_fullres']
    reference=config['batch_size'];patch=config['patch_size']
    group=list(gpus if gpus is not None else devices())
    group,usable,excluded=admit(group)
    if excluded:release_devices([r['gpu_uuid'] for r in excluded],wait=True)
    if not group:raise ResourceAdmissionError(excluded)
    snapshot=probe()
    # Mapped files counted once globally. Missing/compressed source is explicit.
    mapped=sum(p.stat().st_size for p in prep.rglob('*.npy') if p.is_file())
    snapshot=dict(snapshot,source_shared_bytes=mapped)
    local=max_local_batch if max_local_batch is not None else reference
    execution=dict(execution or dict(mode='performance',compile=True))
    reductions=0;eager_retried=False;observed=None;inference=None;phase='observe'
    attempts=[]
    while True:
        while True:
            try:resource=stage_config(snapshot,patch,local,len(group),observed);break
            except StageCapacityError as error:
                if error.policy['cpu_budget']>=len(group) and local>1:
                    if reductions>=reduction_budget:raise
                    reductions+=1;local=max(1,local//2);phase='confirm'
                    attempts.append(dict(status='capacity_reduced',classification='predicted_host_ram',local_batch_size=local))
                    continue
                if len(group)<=1:raise
                removed=group.pop()
                excluded.append(dict(gpu_uuid=removed,reason='CPU/RAM group capacity: '+str(error)))
                release_devices([removed],wait=True)
                usable.pop(removed,None)

        event(work,'stage.decision',decision_schema_version=2,stage=run_kind,phase=phase,gpu_uuids=group,
              exclusions=excluded,resource=resource,local_batch_size=local,receipt_reused=False)
        try:
            result=trial(work,fold,method,group,local,resource,run_kind,execution=execution)
        except Exception as error:
            category=getattr(error,'classification','noncapacity')
            attempts.append(dict(phase=phase,local_batch_size=local,status='failed',classification=category,error=str(error)))
            event(work,'capacity.failure',decision_schema_version=2,attempt=attempts[-1])
            if category=='compile_retry' and not eager_retried:
                eager_retried=True;execution['compile']=False
            elif category=='gpu_oom' and local>1 and reductions<reduction_budget:
                reductions+=1;local=max(1,local//2);phase='confirm'
            else:raise
            # Whole old group has exited. Recheck other users and own-free CPU.
            ready,usable,new_excluded=admit(group)
            if new_excluded:release_devices([r['gpu_uuid'] for r in new_excluded],wait=True)
            excluded+=new_excluded;group=ready
            if not group:raise ResourceAdmissionError(excluded) from error
            snapshot=dict(probe(),source_shared_bytes=mapped)
            continue
        attempts.append(dict(phase=phase,local_batch_size=local,status='passed',output=result.get('output')))
        if phase=='observe' and max_local_batch is None:
            ready,usable,new_excluded=admit(group)
            if ready!=group:raise ResourceAdmissionError(new_excluded)
            inference=infer_batch(result,usable);candidate=inference['local_batch_size']
            observed=observed_payload(result)
            if candidate!=local or observed is not None:
                local=candidate;phase='confirm'
                # Prior trial process/CPU pools exited before this new snapshot.
                snapshot=dict(probe(),source_shared_bytes=mapped)
                continue
        break
    chosen=dict(result,schema_version=2,decision_kind='fresh_stage_capacity',gpu_uuids=group,
        world_size=len(group),global_batch_size=local*len(group),local_batch_size=local,
        effective_cpu=resource,execution_request=execution,resource_receipt_reusable=False,
        resource_snapshot=snapshot,capacity_inference=inference,capacity_attempts=attempts,
        capacity_reductions=reductions,eager_retried=eager_retried,
        excluded_gpus=excluded,reference_global_batch_size=reference,
        sampling_ratio=local*len(group)/reference,train_patches_per_epoch=local*len(group)*250,
        reference_train_patches_per_epoch=reference*250,
        formal_budget=dict(epochs=250,train_iterations=250,validation_iterations=50),
        function_status='passed',effect_status='insufficient_long_window_evidence',
        performance_sufficiency='short 12 train/2 validation capacity and recovery proof; not sustained throughput evidence')
    write_json(work/f'00_admin/stage-decisions/{run_kind}_{fold}_{method}_{time.time_ns()}.json',chosen)
    event(work,'stage.capacity_passed',decision_schema_version=2,stage=run_kind,configuration=chosen)
    return chosen


def train(work,fold,method,resume=False):
    from ddp_runtime import launch_ranks
    from runtime_observations import event
    from task_resources import probe
    from server_train import select_checkpoint,archive_uncheckpointed
    work=Path(work);out=work/f'07_experiments/formal/{fold}_{method}_seed0'
    record=out/'run.json'
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    prep=work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}"
    expected=dict(fold=fold,method=method,seed=0,plan_sha256=digest(prep/'WholeheartBaselinePlans.json'),
        splits_sha256=digest(work/'04_data/manifests/splits_v1.json'),
        environment_sha256=digest(work/'06_configs/pip_freeze_training.txt'),
        conda_environment_sha256=digest(work/'06_configs/conda_explicit_training.txt'),
        code_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    previous=read_json(record) if record.exists() else None
    if previous and previous['signature']!=expected:raise ValueError('训练身份改变，不能混用检查点')
    if previous and previous['status']=='completed':
        if not (out/'fold_0/checkpoint_final.pth').is_file():raise ValueError('最终检查点缺失')
        return
    if previous and not resume:raise ValueError('实验已存在；请选择续跑或独立新任务')
    checkpoint=None;restarted_from=None
    if previous:
        try:checkpoint=str(select_checkpoint(out/'fold_0'))
        except FileNotFoundError:
            # Explicit restart, same frozen signature, no completed epoch yet.
            # Keep the failure scene and restart only this model from seed0.
            restarted_from=archive_uncheckpointed(out)
            previous=None
    chosen=tune(work,fold,method,'formal')
    report=previous or dict(signature=expected,history=[],formal_training=True,split=split)
    if restarted_from:report['restarted_from_archive']=restarted_from
    report.update(status='running',planned_epochs=250,effective_cpu=chosen['effective_cpu'],
        budget=dict(epochs=250,train_iterations=250,validation_iterations=50))
    out.mkdir(parents=True,exist_ok=True)
    request=out/'regroup-request.json';request.unlink(missing_ok=True)
    oom_reductions=chosen.get('capacity_reductions',0);eager_retried=chosen.get('eager_retried',False)
    try:
        while True:
            group=chosen['gpu_uuids'];stop=threading.Event()
            def monitor():
                while not stop.wait(1):
                    try:
                        current=devices(False)
                        additions=[g for g in current if g not in group]
                        if additions:
                            write_json(request,dict(gpu_uuids=group+additions,requested=time.time(),
                                reason='new idle GPU held; join only after complete-epoch checkpoint'))
                            return
                    except Exception as error:
                        event(work,'gpu.monitor_error',error=str(error));return
            watch=threading.Thread(target=monitor,daemon=True);watch.start()
            args=dict(workspace=str(work),fold=fold,method=method,gpu_uuids=group,
                global_batch_size=chosen['global_batch_size'],resource=chosen['effective_cpu'],
                run_kind='formal',operation='train',output=str(out),checkpoint=checkpoint,
                regroup_request=str(request),execution=chosen.get('execution_request',dict(mode='performance',compile=True)))
            report['history'].append(dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),configuration=chosen,checkpoint=checkpoint))
            write_json(record,report)
            print(f"协同训练：{len(group)}卡，全局batch={chosen['global_batch_size']}；每轮{chosen['train_patches_per_epoch']} patches（原{chosen['reference_train_patches_per_epoch']}）；250轮×250训练/50验证。",flush=True)
            try:
                ready,_,exclusions=admit(group)
                if ready!=group:raise ResourceAdmissionError(exclusions)
                values=launch_ranks('cooperative_training:run_rank',args,group,out/'ranks',
                    timeout_seconds=30*24*3600,init_timeout_seconds=120)
            except Exception as error:
                category=getattr(error,'classification','noncapacity')
                # The supervisor reaps every old rank before returning an error.
                allowed=(category=='gpu_oom' and chosen['local_batch_size']>1 and oom_reductions<3
                         or category=='compile_retry' and not eager_retried
                         or isinstance(error,ResourceAdmissionError))
                if not allowed:raise
                if category=='gpu_oom':oom_reductions+=1
                if category=='compile_retry':eager_retried=True
                try:checkpoint=str(select_checkpoint(out/'fold_0'))
                except FileNotFoundError:checkpoint=None
                event(work,'group.recovery',decision_schema_version=2,classification=category,error=str(error),
                      checkpoint=checkpoint,discarded_uncommitted_updates='all since last complete-epoch checkpoint',
                      restart='checkpoint' if checkpoint else 'fresh seed0',rank_failures=getattr(error,'failures',[]))
                stop.set();watch.join(timeout=3)
                if isinstance(error,ResourceAdmissionError):
                    release_devices([r['gpu_uuid'] for r in error.exclusions],wait=True)
                execution=dict(args['execution'])
                if eager_retried:execution['compile']=False
                local=max(1,chosen['local_batch_size']//2) if category=='gpu_oom' else chosen['local_batch_size']
                chosen=tune(work,fold,method,'formal',max_local_batch=local,execution=execution,reduction_budget=3-oom_reductions)
                oom_reductions+=chosen.get('capacity_reductions',0)
                continue
            finally:stop.set();watch.join(timeout=3)
            result=values[0]['result'];checkpoint=result['checkpoint']
            report['initial_weight_sha256']=result['initial_weight_sha256']
            report['history'][-1].update(ended=time.strftime('%Y-%m-%d %H:%M:%S'),result=result)
            if result['status']=='completed':
                report.update(status='completed',completed_epochs=result['next_epoch']);break
            request.unlink(missing_ok=True)
            # Every ready compatible card joins; no speed contest or smaller winner.
            chosen=tune(work,fold,method,'formal',devices(),execution=chosen.get('execution_request'))
            oom_reductions=chosen.get('capacity_reductions',0)
            eager_retried=chosen.get('eager_retried',False)
            if result.get('pending_cpu_resource') and chosen['gpu_uuids']==group:
                from live_resources import feedback_policy
                adjustment=feedback_policy(chosen['effective_cpu'],result.get('feedback_windows',[]),chosen['effective_cpu'])
                chosen['effective_cpu']=adjustment['settings']
                event(work,'cpu.full_epoch_reconfiguration',decision_schema_version=2,decision=adjustment,checkpoint=checkpoint)
            event(work,'regroup.committed',decision_schema_version=2,next_configuration=chosen,checkpoint=checkpoint,
                rng_scope='existing UUID RNG retained; new UUID deterministic stream at next complete epoch; sampling may change')
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:write_json(record,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('fold');parser.add_argument('method')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    train(os.environ['WHOLE_HEART_WORKSPACE'],args.fold,args.method,args.resume)
