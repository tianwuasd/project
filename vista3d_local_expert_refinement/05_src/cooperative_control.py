"""Bounded resource measurements and full-epoch membership changes, no parent CUDA."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

from server_data import read_json,write_json,digest


def release_devices(uuids,*,wait=False):
    uuids=set(uuids)
    if uuids:
        pending=[]
        if wait:
            from gpu_fleet import verified_keepers
            pending=[r for uuid,r in verified_keepers(os.environ['VISTA_GPU_LEASES']).items() if uuid in uuids]
        write_json(Path(os.environ['VISTA_GPU_LEASES']).with_name('gpu-release-request.json'),
            dict(id=str(time.time_ns()),gpu_uuids=list(uuids),
                reason='measured configuration did not benefit or exceeds capacity; retry after10minutes'))
        if wait:
            from task_runtime import alive
            deadline=time.monotonic()+30
            while any(alive(record) for record in pending):
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


def trial(work,fold,method,gpus,local_batch,resource,run_kind,*,output=None,checkpoint=None):
    from ddp_runtime import launch_ranks
    from runtime_observations import event,failure
    work=Path(work)
    from evaluation_gpu import snapshot,idle_reason
    current={row['uuid']:row for row in snapshot()}
    settings=dict(min_free_mib=6000,max_utilization_percent=10,
                  planner_gpu_memory_gib=float(os.environ.get('WHOLE_HEART_GPU_MEMORY','20')))
    for uuid in gpus:
        if uuid not in current:raise ValueError('实际GPU已不可见：'+uuid)
        reason=idle_reason(current[uuid],settings)
        if reason:raise ValueError('启动前GPU复核失败 '+uuid+'：'+reason)
    output=Path(output) if output else work/f'07_experiments/tuning/{fold}_{method}_{time.time_ns()}'
    output.mkdir(parents=True,exist_ok=False)
    args=dict(workspace=str(work),fold=fold,method=method,gpu_uuids=list(gpus),
        global_batch_size=local_batch*len(gpus),resource=resource,run_kind=run_kind,
        operation='trial',output=str(output),checkpoint=checkpoint)
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


def tune(work,fold,method,run_kind,gpus=None):
    from task_resources import probe
    from training_policy import cpu_candidates,select_trial
    work=Path(work);gpus=list(gpus or devices())
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    plan=read_json(work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}/WholeheartBaselinePlans.json")
    config=plan['configurations']['3d_fullres'];reference=config['batch_size'];patch=config['patch_size']
    snapshot=probe();excess=gpus[max(1,snapshot['cpu_slots']):]
    release_devices(excess)
    gpus=gpus[:max(1,snapshot['cpu_slots'])]
    baseline=cpu_candidates(snapshot,patch,reference,1)[0]
    print(f'测量1/最多2：1卡，每卡batch={reference}，完整尺寸2步预热+10步测量、验证和恢复。',flush=True)
    first=trial(work,fold,method,gpus[:1],reference,baseline,run_kind)
    trials=[first];failures=[]
    # Keep time bounded: one baseline and one evidence-guided joint candidate.
    # The first real peak bounds batch growth; free VRAM is not all activations.
    used=max(first.get('peak_reserved_gib',20),first.get('peak_allocated_gib',20))
    budget=float(os.environ.get('WHOLE_HEART_GPU_MEMORY','20'))
    local=max(1,min(reference*2,int(reference*budget/max(used,1)*.8)))
    world=len(gpus)
    candidates=cpu_candidates(snapshot,patch,local*world,world)
    resource=candidates[-1] if candidates else None
    if resource and (world>1 or local!=reference or resource!=baseline):
        print(f'测量2/最多2：{world}卡协同，每卡batch={local}；主线程/增强worker按总CPU、RAM和共享内存限额。',flush=True)
        try:trials.append(trial(work,fold,method,gpus,local,resource,run_kind,checkpoint=first.get('checkpoint')))
        except Exception as error:failures.append(dict(error=str(error),gpus=gpus,local_batch=local))
    chosen=select_trial(trials,global_batch=None)
    chosen=dict(chosen,reference_global_batch_size=reference,
        sampling_ratio=chosen['global_batch_size']/reference,
        train_patches_per_epoch=chosen['global_batch_size']*250,
        reference_train_patches_per_epoch=reference*250,
        formal_budget=dict(epochs=250,train_iterations=250,validation_iterations=50))
    chosen['tested_gpu_uuids']=gpus
    write_json(work/f'00_admin/optimization/{fold}_{method}.json',dict(status='passed',chosen=chosen,
        candidates=trials,failed_candidates=failures,resource_snapshot=snapshot,
        scope='bounded joint search, not a global optimum; no convergence/quality claim'))
    return chosen


def train(work,fold,method,resume=False):
    from ddp_runtime import launch_ranks
    from runtime_observations import event
    from task_resources import probe
    from training_policy import receipt_compatibility
    from server_train import select_checkpoint,archive_uncheckpointed
    work=Path(work);out=work/f'07_experiments/formal/{fold}_{method}_seed0'
    record=out/'run.json'
    optimization_identity=identity(work,fold,method)
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
    recommendation=Path(os.environ['VISTA_RUNTIME'])/f'optimization/{fold}_{method}.json'
    chosen=None
    if recommendation.exists():
        receipt=read_json(recommendation)
        reasons=receipt_compatibility(receipt,optimization_identity,probe())
        try:
            if digest(receipt['assessment_report'])!=receipt['assessment_sha256']:
                reasons.append('原测评报告摘要不一致')
        except (OSError,KeyError):reasons.append('原测评报告不可核验')
        available=devices()
        if not reasons and receipt['chosen']['world_size']<=len(available):
            chosen=dict(receipt['chosen'],gpu_uuids=available[:receipt['chosen']['world_size']])
            event(work,'recommendation.reused',source=str(recommendation),configuration=chosen)
        else:event(work,'recommendation.rejected',reasons=reasons or ['实际可用卡数不足，需按当前组重测'])
    if chosen is None:chosen=tune(work,fold,method,'formal')
    release_devices(set(chosen.get('tested_gpu_uuids',[]))-set(chosen['gpu_uuids']))
    report=previous or dict(signature=expected,history=[],formal_training=True,split=split)
    if restarted_from:report['restarted_from_archive']=restarted_from
    report.update(status='running',planned_epochs=250,effective_cpu=chosen['effective_cpu'],
        budget=dict(epochs=250,train_iterations=250,validation_iterations=50))
    out.mkdir(parents=True,exist_ok=True)
    request=out/'regroup-request.json';request.unlink(missing_ok=True)
    refused={}
    try:
        while True:
            group=chosen['gpu_uuids'];stop=threading.Event()
            def monitor():
                while not stop.wait(1):
                    try:
                        current=devices(False)
                        additions=[g for g in current if g not in group and time.monotonic()>refused.get(g,0)]
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
                regroup_request=str(request))
            report['history'].append(dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),configuration=chosen,checkpoint=checkpoint))
            write_json(record,report)
            print(f"协同训练：{len(group)}卡，全局batch={chosen['global_batch_size']}；每轮{chosen['train_patches_per_epoch']} patches（原{chosen['reference_train_patches_per_epoch']}）；250轮×250训练/50验证。",flush=True)
            try:
                values=launch_ranks('cooperative_training:run_rank',args,group,out/'ranks',
                    timeout_seconds=30*24*3600,init_timeout_seconds=120)
            finally:stop.set();watch.join(timeout=3)
            result=values[0]['result'];checkpoint=result['checkpoint']
            report['initial_weight_sha256']=result['initial_weight_sha256']
            report['history'][-1].update(ended=time.strftime('%Y-%m-%d %H:%M:%S'),result=result)
            if result['status']=='completed':
                report.update(status='completed',completed_epochs=result['next_epoch']);break
            requested=read_json(request)['gpu_uuids'];request.unlink(missing_ok=True)
            try:
                candidate=tune(work,fold,method,'formal',requested)
                # A new group must improve the measured fixed-update epoch cost
                # (within5% prefer more samples/s) before disturbing the old group.
                from training_policy import select_trial
                winner=select_trial([chosen,candidate],global_batch=None)
                if winner is candidate:chosen=candidate
                else:
                    for uuid in set(requested)-set(group):refused[uuid]=time.monotonic()+600
                    release_devices(set(requested)-set(group))
            except Exception as error:
                event(work,'regroup.rejected',error=str(error),retained_checkpoint=checkpoint)
                for uuid in set(requested)-set(group):refused[uuid]=time.monotonic()+600
                release_devices(set(requested)-set(group))
            event(work,'regroup.committed',next_configuration=chosen,checkpoint=checkpoint,
                rng_scope='existing UUID RNG retained; new UUID deterministic stream at next complete epoch; batch sampling may change')
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:write_json(record,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('fold');parser.add_argument('method')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    train(os.environ['WHOLE_HEART_WORKSPACE'],args.fold,args.method,args.resume)
