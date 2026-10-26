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
                    'measured configuration did not benefit or exceeds capacity; retry after10minutes'))
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
    work=Path(work);requested=gpus is not None;gpus=list(gpus if requested else devices())
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
    # Keep the explicit regroup bounded/frozen. At initial startup additional
    # keepers may finish initialization during the baseline's twelve steps.
    if not requested:
        gpus=list(dict.fromkeys(gpus+devices(False)))
        snapshot=probe()
        release_devices(gpus[max(1,snapshot['cpu_slots']):])
        gpus=gpus[:max(1,snapshot['cpu_slots'])]
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
    chosen['selected_hardware']=topology(chosen['gpu_uuids'])
    chosen['measurement_coverage']=[dict(status='passed',world_size=t['world_size'],
        hardware=topology(t['gpu_uuids']),gpu_uuids=t['gpu_uuids'],
        local_batch_size=t['local_batch_size'],global_batch_size=t['global_batch_size'],
        effective_cpu=t['effective_cpu']) for t in trials]
    write_json(work/f'00_admin/optimization/{fold}_{method}.json',dict(status='passed',chosen=chosen,
        candidates=trials,failed_candidates=failures,resource_snapshot=snapshot,
        measurement_coverage=chosen['measurement_coverage'],
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
        covered=coverage_matches(receipt.get('measurement_coverage',[]),available) if not reasons else False
        if not reasons and not covered:reasons.append('当前可用硬件拓扑没有成功测量记录，首次正式轮前有界重测')
        selected=receipt['chosen'].get('selected_hardware',[])
        replacement=equivalent_devices(available,selected) if not reasons and len(selected)==receipt['chosen']['world_size'] else None
        if not reasons and not replacement:reasons.append('选优配置缺少匹配的实测硬件，首次正式轮前有界重测')
        if not reasons and receipt['chosen']['world_size']<=len(available):
            chosen=dict(receipt['chosen'],gpu_uuids=replacement,tested_gpu_uuids=available)
            event(work,'recommendation.reused',source=str(recommendation),configuration=chosen)
        else:event(work,'recommendation.rejected',reasons=reasons or ['实际可用卡数不足，需按当前组重测'])
    if chosen is None:chosen=tune(work,fold,method,'formal')
    initial_unused=set(chosen.get('tested_gpu_uuids',[]))-set(chosen['gpu_uuids'])
    release_devices(initial_unused)
    report=previous or dict(signature=expected,history=[],formal_training=True,split=split)
    if restarted_from:report['restarted_from_archive']=restarted_from
    report.update(status='running',planned_epochs=250,effective_cpu=chosen['effective_cpu'],
        budget=dict(epochs=250,train_iterations=250,validation_iterations=50))
    out.mkdir(parents=True,exist_ok=True)
    request=out/'regroup-request.json';request.unlink(missing_ok=True)
    refused={uuid:time.monotonic()+600 for uuid in initial_unused}
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
            except Exception as error:
                event(work,'regroup.rejected',error=str(error),retained_checkpoint=checkpoint)
            # Reconcile every outcome, including a faster candidate with fewer
            # ranks. Old members discarded by that winner are unused as well.
            unused=(set(requested)|set(devices(False))|set(chosen.get('tested_gpu_uuids',[])))-set(chosen['gpu_uuids'])
            for uuid in unused:refused[uuid]=time.monotonic()+600
            release_devices(unused)
            event(work,'regroup.committed',next_configuration=chosen,checkpoint=checkpoint,
                rng_scope='existing UUID RNG retained; new UUID deterministic stream at next complete epoch; batch sampling may change')
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:write_json(record,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('fold');parser.add_argument('method')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    train(os.environ['WHOLE_HEART_WORKSPACE'],args.fold,args.method,args.resume)
