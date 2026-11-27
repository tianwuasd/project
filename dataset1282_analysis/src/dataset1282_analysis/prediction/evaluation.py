from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import json
import math
import time

import nibabel as nib
import numpy as np

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock
from ..xu_evaluation import class_metrics, summarize
from ..resources import budget, snapshot
from .workflow import case_record,verify_job
from .evaluation_log import EvaluationLog


def fingerprints(out,name,row,record):
    paths=[Path(row['label'])]
    if record['state']!='roi_rejected':
        paths.append(out/'cases'/(name+'.nii.gz'))
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}


def restore_metrics(row):
    if 'metrics' in row:
        row=dict(row,metrics={int(k):v for k,v in row['metrics'].items()})
    return row


def load_checkpoint(out,job,cases):
    path=out/'evaluation-checkpoint.json'
    state=read_json(path) if path.exists() else dict(job=digest(job),metric='native-voxel-surface-v1',cases=legacy_results(out,job,cases))
    if state['job']!=digest(job) or state['metric']!='native-voxel-surface-v1':
        raise ValueError('评价检查点对应的预测任务或指标定义不同')
    results={}
    for name,saved in state['cases'].items():
        if name in cases and saved['fingerprints']==fingerprints(out,name,job['cases'][name],cases[name]):
            results[name]=restore_metrics(saved['result'])
    return state,results


def legacy_results(out,job,cases):
    path=out/'evaluation.log'
    found={}
    partial={}
    if not path.exists():
        return found
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.startswith('{') or not line.endswith('\n'):
                continue
            item=json.loads(line)
            name=item.get('case')
            if name not in cases:
                continue
            if item.get('event')=='metric_label_completed':
                partial.setdefault(name,{})[int(item['label_id'])]=item['metrics']
            if item.get('event')=='case_completed' and item.get('outcome')=='evaluated':
                metrics=partial.get(name,{})
                stamps=fingerprints(out,name,job['cases'][name],cases[name])
                if set(metrics)==set(range(1,11)) and all(v[1]<=int(item['updated_at']*1e9) for v in stamps.values()):
                    row=job['cases'][name]
                    found[name]={'fingerprints':stamps,'result':dict(role=row['role'],device=row['device'],state='evaluated',
                        complete_annotation=all(v['status'] in ('scored','empty_prediction') for v in metrics.values()),metrics=metrics)}
    return found


def evaluate_case(out,name,row,record,log,index):
    started=time.monotonic()
    log.emit('case_started',stage='read_reference',case=name,case_index=index,label_id=None,metrics=None)
    if record['state']=='roi_rejected':
        return dict(role=row['role'],device=row['device'],state='roi_rejected')
    # 原网格尺寸与毫米间距已冻结在预测任务中，评价不再打开预处理CT数组。
    geo=SimpleNamespace(native_shape_xyz=tuple(row['shape']),spacing_xyz=tuple(row['spacing']))
    volume=nib.load(row['label'])
    shift=np.eye(4)
    shift[:3,3]=1
    if volume.shape!=geo.native_shape_xyz or not any(np.allclose(volume.affine,a,rtol=0,atol=1e-5) for a in (np.eye(4),shift)):
        raise ValueError('Held-out reference grid differs from the reviewed array convention')
    reference=np.asanyarray(volume.dataobj)
    if not np.isfinite(reference).all() or reference.min()<0 or reference.max()>10 or not np.equal(reference,np.floor(reference)).all():
        raise ValueError('Invalid held-out reference labels')
    reference=np.ascontiguousarray(reference,dtype=np.uint8)
    present=sorted(set(map(int,np.unique(reference)))-{0})
    absent=row['known_absent']
    if not present or set(absent)&set(present) or set(absent)-set(range(1,11)):
        raise ValueError('Invalid held-out supervision metadata')
    info=dict(present_classes=present,known_absent_classes=absent,unknown_classes=sorted(set(range(1,11))-set(present)-set(absent)))
    log.emit('reference_completed',stage='read_prediction',case=name,reference_seconds=time.monotonic()-started)
    before=time.monotonic()
    image=nib.load(out/'cases'/(name+'.nii.gz'))
    if image.shape!=geo.native_shape_xyz or not np.allclose(image.affine,np.diag([*geo.spacing_xyz,1.])):
        raise ValueError('Prediction export geometry changed')
    prediction=np.ascontiguousarray(np.asanyarray(image.dataobj))
    log.emit('prediction_read_completed',stage='metrics',case=name,prediction_seconds=time.monotonic()-before)
    def progress(event,**fields):
        log.emit(event,case=name,case_index=index,**fields)
    metrics=class_metrics(reference,prediction,geo.spacing_xyz,info,progress=progress)
    return dict(role=row['role'],device=row['device'],state='evaluated',complete_annotation=len(present)==10,metrics=metrics)


def evaluate(out,workers=0):
    if workers<0:
        raise ValueError('评价并发数必须为非负整数')
    with operation_lock(out,'xu-prediction-evaluate',out), ExitStack() as contexts:
        job = verify_job(out)
        complete = read_json(out/'complete.json')
        cases = {name:case_record(out,name,job) for name in job['cases']}
        if complete['job'] != digest(job) or complete['cases'] != cases or any(r is None for r in cases.values()):
            raise ValueError('Evaluation requires the complete unchanged prediction set')
        ledger=out/'evaluation-checkpoint.json'
        # 先恢复旧日志的完整行，再追加新日志，避免进程终止留下的末尾半行参与恢复。
        state,results=load_checkpoint(out,job,cases)
        atomic_json(ledger,state)
        log=contexts.enter_context(EvaluationLog(out))
        log.emit('verification_completed',stage='verified',total=len(cases))
        reused=len(results)
        log.emit('resume_completed',stage='admission',completed=reused,reused_cases=reused)
        todo=[(i,name) for i,name in enumerate(cases,1) if name not in results]
        ceiling=workers or 4
        active={}
        with ThreadPoolExecutor(max_workers=ceiling,thread_name_prefix='evaluation-case') as pool:
            while todo or active:
                fresh=budget(snapshot(),dict(cpu_fraction=.8,ram_fraction=.8,reserve_gib=1,max_workers=ceiling,read_concurrency=ceiling))
                while todo and len(active)<ceiling:
                    index,name=todo[0]
                    demand=math.prod(job['cases'][name]['shape'])*128+256*1024**2
                    if (fresh['cpu_slots'] is None or fresh['memory_bytes'] is None or len(active)+1>fresh['cpu_slots']
                            or demand+sum(x[2] for x in active.values())>fresh['memory_bytes']):
                        break
                    todo.pop(0)
                    stamps=fingerprints(out,name,job['cases'][name],cases[name])
                    future=pool.submit(evaluate_case,out,name,job['cases'][name],cases[name],log,index)
                    active[future]=(name,stamps,demand)
                if not active:
                    log.emit('resource_wait',stage='waiting_resources',completed=len(results),budget=fresh)
                    time.sleep(5)
                    continue
                done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    name,stamps,_=active.pop(future)
                    result=future.result()
                    if stamps!=fingerprints(out,name,job['cases'][name],cases[name]):
                        raise ValueError('评价期间标签或分割文件发生变化：'+name)
                    results[name]=result
                    state['cases'][name]={'fingerprints':stamps,'result':result}
                    atomic_json(ledger,state)
                    log.emit('case_completed',stage='case_completed',case=name,completed=len(results),
                             outcome=result['state'],active_cases=[x[0] for x in active.values()])
        log.emit('summary_started',stage='summarize',case=None,label_id=None,metrics=None)
        groups = defaultdict(list)
        devices = defaultdict(list)
        for row in results.values():
            groups[row['role']].append(row)
            devices[row['role']+'/'+row['device']].append(row)
        def summary(rows):
            good = [r for r in rows if r['state']=='evaluated']
            return dict(selected=len(rows),evaluated=len(good),roi_rejected=len(rows)-len(good),
                        observed_structure=summarize([r['metrics'] for r in good]),
                        full_annotation=summarize([r['metrics'] for r in good if r['complete_annotation']]),
                        metric_population='successfully_predicted_cases; rejected cases reported separately')
        report = dict(job=digest(job),cases=results,groups={k:summary(v) for k,v in groups.items()},
                      devices={k:summary(v) for k,v in devices.items()},
                      coordinate_frame='local_array_excel_mm',models=digest(read_json(out/'source/manifest.json')))
        log.emit('report_write_started',stage='write_report')
        atomic_json(out/'evaluation.json',report)
        log.emit('evaluation_completed',stage='complete',state='complete',completed=len(cases),
                 result=str(out/'evaluation.json'))
        print('Evaluation:',out/'evaluation.json',flush=True)
        return report
