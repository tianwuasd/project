"""V2-style stage ceilings, live headroom and process-owned recovery."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import psutil

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..engine.runner import sample_owned, stop_owned
from ..observability.logging import event
from ..observability.status import register_workers, write_status
from ..resources import admit_case, budget, snapshot, thread_environment
from ..vendor.vista_read_admission import create_gate, update_gate


def committed(output, job, contract):
    marker = output/'commits'/(job['id']+'.json')
    if not marker.exists():
        return False
    data = read_json(marker)
    if data['contract'] != contract or data['job'] != job:
        raise ValueError('Job completion contract changed: '+job['id'])
    if not data.get('artifacts'):
        raise ValueError('Empty completion marker')
    for relative, expected in data['artifacts'].items():
        path = output/relative
        if output.resolve() not in path.resolve().parents or not path.is_file() or sha256_file(path) != expected:
            raise ValueError('Committed artifact is missing or changed: '+relative)
    return True


def run_jobs(output, jobs, python, policy, home, owner):
    contract = digest(read_json(output/'request.json'))
    pending, active = [], []
    reused = sum(committed(output, j, contract) for j in jobs)
    for job in jobs:
        if not (output/'commits'/(job['id']+'.json')).exists():
            pending.append(dict(job=job,reserved=job['memory'],threads=1,private_resident=0))
    start = time.monotonic()
    facts = snapshot()
    ceiling = budget(facts, policy)
    ceiling['workers'] = min(policy['max_workers'],policy.get('_effective_workers',policy['max_workers']))
    if ceiling['memory_bytes'] is not None:
        ceiling['memory_bytes'] = int(ceiling['memory_bytes']*policy.get('_memory_factor',1))
    gate = create_gate(min(policy['read_concurrency'],ceiling['workers']))
    event(output,'stage_budget',stage=jobs[0]['kind'],facts=facts,applied=ceiling,
          read_concurrency=policy['read_concurrency'],threads_per_process=1,reused=reused)
    env = dict(os.environ, **thread_environment(),PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=str(output/'code_snapshot/src'),VISTA_READ_GATE=str(gate))
    for key, sub in [('nnUNet_raw','raw'),('nnUNet_preprocessed','preprocessed'),('nnUNet_results','results')]:
        env[key] = str(output/sub)
    (output/'jobs').mkdir(exist_ok=True)
    (output/'worker_logs').mkdir(exist_ok=True)
    completed, peak, max_active = reused, 0, 0
    adjustments = 0
    last_progress, last_status = time.monotonic(),0
    stopped = False
    old = signal.getsignal(signal.SIGTERM)
    def stop_request(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop_request)
    try:
        while pending or active:
            if stopped:
                raise KeyboardInterrupt('Stop requested; committed jobs can resume')
            for item in list(active):
                sample_owned(item)
                peak = max(peak,item.get('peak_rss',0))
                code = item['process'].poll()
                if code is None:
                    continue
                item['stream'].close()
                active.remove(item)
                register_workers(home,owner,[a['process'].pid for a in active])
                if code or not committed(output,item['job'],contract):
                    raise RuntimeError('Numerical job failed: '+item['job']['id']+'; see worker_logs, exit='+str(code))
                completed += 1
                event(output,'job_done',job=item['job']['id'],seconds=time.monotonic()-item['started'],
                      peak_rss_bytes=item.get('peak_rss',0))
                last_progress=time.monotonic()
            fresh = budget(snapshot(),policy)
            reserve = max(v for v in (ceiling['reserve_bytes'],fresh['reserve_bytes'],policy['reserve_gib']*2**30) if v is not None)
            if active and (fresh['available_memory_bytes'] is None or fresh['available_memory_bytes']<reserve):
                adjustments += 1
                policy['_recoveries'] = policy.get('_recoveries',0)+1
                event(output,'memory_pressure',fresh=fresh,reserve_bytes=reserve,recovery=policy['_recoveries'])
                for item in active:
                    stop_owned(item)
                    item['stream'].close()
                    # A process may have committed just before termination.
                    if committed(output,item['job'],contract):
                        completed += 1
                    else:
                        pending.insert(0,dict(job=item['job'],reserved=item['reserved'],threads=1,private_resident=0))
                active=[]
                register_workers(home,owner,[])
                ceiling['workers']=max(1,ceiling['workers']//2)
                ceiling['memory_bytes']=int(ceiling['memory_bytes']*.75)
                policy['_effective_workers']=ceiling['workers']
                policy['_memory_factor']=policy.get('_memory_factor',1)*.75
                update_gate(gate,min(policy['read_concurrency'],ceiling['workers']))
                last_progress=time.monotonic()
                if policy['_recoveries']>3:
                    raise MemoryError('Three recovery attempts exhausted')
            if pending:
                # Smallest-first prevents one large case blocking smaller admissible work.
                pending.sort(key=lambda i:(i['reserved'],i['job']['id']))
                decision=admit_case(ceiling,fresh,active,pending[0]['reserved'])
                if decision['allowed']:
                    item=pending.pop(0)
                    path=output/'jobs'/(item['job']['id']+'.json')
                    atomic_json(path,dict(job=item['job'],output=str(output),contract=contract,
                                          home=str(home),owner=owner))
                    stream=(output/'worker_logs'/(item['job']['id']+'.log')).open('a',encoding='utf-8')
                    try:
                        proc=subprocess.Popen([python,'-B','-m','dataset1282_analysis.preprocessing.jobs',str(path)],
                            env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                            start_new_session=os.name!='nt')
                    except BaseException:
                        stream.close()
                        raise
                    item.update(process=proc,stream=stream,born=psutil.Process(proc.pid).create_time(),
                                started=time.monotonic(),peak_rss=0)
                    active.append(item)
                    max_active=max(max_active,len(active))
                    register_workers(home,owner,[a['process'].pid for a in active])
                    event(output,'job_admitted',job=item['job']['id'],pid=proc.pid,decision=decision,
                          fresh_budget=fresh,applied_stage_budget=ceiling)
                    last_progress=time.monotonic()
                elif not active and time.monotonic()-last_progress>policy['resource_wait_seconds']:
                    raise MemoryError('Bounded admission wait expired: '+decision['reason'])
            if time.monotonic()-last_status>=1:
                write_status(output,state='running',stage=jobs[0]['kind'],completed=completed,total=len(jobs),
                             active=[a['job']['id'] for a in active],pending=len(pending),
                             seconds=time.monotonic()-start,last_progress_age_seconds=time.monotonic()-last_progress,
                             fresh_budget=fresh,stage_budget=ceiling,peak_worker_rss_bytes=peak)
                last_status=time.monotonic()
            if pending or active:
                time.sleep(.35)
    finally:
        for item in active:
            stop_owned(item)
            item['stream'].close()
        register_workers(home,owner,[])
        signal.signal(signal.SIGTERM,old)
        shutil.rmtree(gate)
    result=dict(stage=jobs[0]['kind'],jobs=len(jobs),reused=reused,seconds=time.monotonic()-start,
                max_active=max_active,peak_worker_rss_bytes=peak,memory_recoveries=adjustments)
    event(output,'stage_complete',**result)
    return result
