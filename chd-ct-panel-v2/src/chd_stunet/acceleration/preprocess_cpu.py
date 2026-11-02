"""V2 stage policy adapted to independent, memory-weighted preprocessing jobs.

Linux facts and case_policy come from the installed skill's 0.7.2 source.
Windows has an explicit psutil adapter. Compute admission and actual file-read
admission are separate. No training pool, GPU allocation or external PID cleanup.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from .._vendor.vista_live_resources import snapshot as linux_snapshot, case_policy
from ..observability.runlog import write_json

GIB = 1024**3


def snapshot():
    if sys.platform == 'linux':
        return linux_snapshot()
    if sys.platform != 'win32':
        raise RuntimeError('Resource adapter supports Windows and Linux only')
    affinity = psutil.Process().cpu_affinity()
    usage = psutil.cpu_percent(interval=.15, percpu=True)
    available = sum(max(0., 1.-usage[c]/100.) for c in affinity)
    memory = int(psutil.virtual_memory().available)
    return dict(schema_version=2, collected_at=time.time(), cpu_permitted=len(affinity),
                cpu_available=available, cpu_budget=math.floor(.8*available),
                affinity_cpus=affinity, available_memory_bytes=memory,
                memory_budget_bytes=math.floor(.8*memory), io_pressure=None,
                cgroup_evidence=[], missing_evidence=['Windows: Linux cgroup/PSI/NUMA evidence not applicable'],
                cpu_availability_source='psutil_permitted_cpu_idle_sample')


def estimate_memory(shape, spacing, target=None, *, audit=False):
    """Conservative array/workspace prior includes raw AND resampled volumes.

    The project sampler avoids materializing/sorting all background coordinates;
    the 64-byte target-voxel prior is for this bounded adapter, not upstream's
    unrestricted class-location routine. Actual peaks are logged separately.
    """
    native = math.prod(shape)
    target_voxels = native if target is None else math.prod(
        max(1, math.ceil(n*s/t)) for n,s,t in zip(shape,spacing,target))
    return int(native*24 + 256*1024**2) if audit else int(max(native,target_voxels)*64 + GIB)


def policy(facts, jobs, maximum=None):
    count = len(jobs) if maximum is None else min(len(jobs), maximum)
    estimate = min((j['estimate'] for j in jobs), default=0)
    result = case_policy(facts, count, estimate, supported_threads=1)
    memory = facts.get('available_memory_bytes')
    if memory is not None:
        reserve = max(4*GIB, math.ceil(.2*memory))
        result['memory_budget_bytes'] = max(0, min(int(.8*memory), memory-reserve))
        result['workers'] = min(result['workers'], result['memory_budget_bytes']//max(1,estimate))
    else:
        reserve = None
    return dict(facts | result, reserve_bytes=reserve, threads_per_worker=1,
                requested_workers=maximum, source='server-engineering V2 0.7.2 CPU case policy + project weighted queue')


def admits(stage, fresh, active, estimate, threads=1):
    """Stage CPU is total; fresh CPU is spare. Never subtract activity twice.

    Stage RAM reserves every job in full. Fresh free RAM already excludes
    resident private bytes, so only unfulfilled reservations are subtracted
    again. Unavailable USS retains the full reservation. Shared RSS is never
    credited, as required by the V2 0.7.2 queue correction.
    """
    if len(active) >= stage['workers'] or fresh['workers'] < 1:
        return False
    if sum(a['threads'] for a in active)+threads > stage['cpu_budget'] or threads > fresh['cpu_budget']:
        return False
    ceiling = stage.get('memory_budget_bytes')
    if ceiling is not None and sum(a['estimate'] for a in active)+estimate > ceiling:
        return False
    spare = fresh.get('memory_budget_bytes')
    outstanding = sum(max(0,a['estimate']-a.get('private',0)) for a in active)
    return spare is None or outstanding+estimate <= spare


def private_bytes(item):
    """Only the directly spawned, still-live process with its recorded birth."""
    try:
        p = psutil.Process(item['process'].pid)
        if item['process'].poll() is not None or p.create_time() != item['born']:
            return 0
        value = int(p.memory_full_info().uss)
        return value if p.is_running() and p.create_time() == item['born'] else 0
    except (psutil.Error, OSError, AttributeError):
        return 0


def interpreter_command(env):
    """Bypass Windows venv's waiting redirector, preserving its Python paths.

    CPython's venv redirector uses __PYVENV_LAUNCHER__ when launching the base
    interpreter. Doing that directly makes Popen.pid the actual compute PID,
    so verified direct-child USS and owned cleanup observe the real process.
    No PYTHONHOME/site-packages injection or environment mixing is necessary.
    The worker's returned PID must match the spawned PID before acceptance.
    """
    base = getattr(sys,'_base_executable',sys.executable)
    if sys.platform=='win32' and Path(base).resolve()!=Path(sys.executable).resolve():
        env['__PYVENV_LAUNCHER__']=sys.executable
        return base
    return sys.executable


def stop_owned(item):
    """Popen ownership plus process birth must match before any signal."""
    p = item['process']
    if p.poll() is not None:
        return
    try:
        if psutil.Process(p.pid).create_time() != item['born']:
            raise RuntimeError('Owned worker birth identity changed; refusing signal')
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if psutil.Process(p.pid).create_time() == item['born']:
                p.kill()
                p.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass


def execute_jobs(jobs, output, log, *, stage_name, workers=None, readers=None, completed=None):
    """One supervised subprocess per case, bounded by fresh capacity.

    A one-second memory guard preserves the entry reserve. Sustained pressure
    stops only our workers, retains committed cases, halves the worker ceiling
    and replays incomplete jobs, at most three times. No batch/plan changes.
    """
    if not jobs:
        return dict(completed=0, reused_only=True)
    output = Path(output)
    pending = sorted(jobs, key=lambda j:j['estimate'])
    stage = policy(snapshot(), pending, workers)
    write_json(output/'resources'/f'{stage_name}.json', stage)
    log.event('resource.stage', f"{stage_name}：预算CPU {stage['cpu_budget']}槽，最多{stage['workers']}个病例进程；内层1线程；读入单独限流", policy=stage)
    if stage['workers'] < 1:
        raise MemoryError('Stage CPU/RAM cannot admit even the smallest case; see resources')
    active, done, recoveries = [], 0, 0
    peak_rss, peak_uss, peak_jobs = 0, 0, 0
    peaks_by_job = {}
    last_sample, last_console, pressure = 0., time.monotonic(), 0
    fresh = stage
    gate = output/'read_gate.json'
    try:
        while pending or active:
            now = time.monotonic()
            if now-last_sample >= 1:
                fresh = policy(snapshot(), pending or jobs, workers)
                used_rss, used_uss = 0, 0
                for a in active:
                    a['private'] = private_bytes(a)
                    used_uss += a['private']
                    try:
                        rss = psutil.Process(a['process'].pid).memory_info().rss
                    except psutil.Error:
                        rss = 0
                    used_rss += rss
                    peaks_by_job[a['name']] = max(peaks_by_job.get(a['name'],0),rss)
                peak_rss, peak_uss = max(peak_rss,used_rss), max(peak_uss,used_uss)
                peak_jobs = max(peak_jobs,len(active))
                free = fresh.get('available_memory_bytes')
                pressure = pressure+1 if free is not None and stage['reserve_bytes'] is not None and free < stage['reserve_bytes'] else 0
                slots = min(stage['workers'], readers or stage['workers'])
                if (fresh.get('io_pressure') or {}).get('avg10',0) >= 10:
                    slots = min(slots,2)
                write_json(gate, dict(slots=max(1,slots)))
                record = dict(time=time.time(), stage=stage_name, active=len(active), pending=len(pending),
                              completed=done, free_bytes=free, rss_bytes=used_rss, uss_bytes=used_uss,
                              fresh_cpu_budget=fresh['cpu_budget'], read_slots=max(1,slots),
                              pressure_samples=pressure, recoveries=recoveries)
                with (output/'logs/resources.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record)+'\n')
                log.status('running',stage=stage_name,**{k:v for k,v in record.items() if k!='stage'})
                last_sample = now
                if pressure >= 3 and active:
                    if recoveries >= 3:
                        raise MemoryError('Memory guard exhausted three owned-worker recoveries')
                    for a in active:
                        stop_owned(a)
                        a['stream'].close()
                        pending.append(a['job'])
                    active.clear()
                    stage['workers'] = max(1,stage['workers']//2)
                    recoveries += 1
                    pressure = 0
                    log.event('resource.recover', '持续内存压力：已停本次病例进程，保留完成项，缩减并发后重做未完成项', recoveries=recoveries,workers=stage['workers'])
            released = False
            for a in list(active):
                rc = a['process'].poll()
                if rc is None:
                    continue
                active.remove(a)
                released = True
                a['stream'].close()
                done += 1
                result_path = Path(a['job']['result'])
                result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else dict(success=False,error=f'Worker exit {rc} without result')
                if rc:
                    result['success'] = False
                if result.get('worker_pid') != a['process'].pid:
                    result.update(success=False,error='Worker PID differs from directly owned compute PID')
                result['observed_peak_rss_bytes'] = peaks_by_job.get(a['name'],0)
                if completed:
                    completed(a['name'],result)
                log.event('case.done',f"{stage_name} {done}/{len(jobs)}：{a['name']} {'完成' if result['success'] else '失败，见逐例日志'}", result=result)
            if released and pending:
                # The last periodic sample still includes the exiting child's
                # resident RAM. Once its reservation is removed, refresh host/
                # cgroup headroom too; otherwise an empty queue can falsely
                # reject the next full-size case using stale pre-exit memory.
                fresh = policy(snapshot(),pending,workers)
            for job in list(pending):
                if not admits(stage,fresh,active,job['estimate']):
                    continue
                name = job['name']
                stream = (output/'logs'/f'{name}.log').open('a',encoding='utf-8')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1',
                           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='1')
                try:
                    python = interpreter_command(env)
                    p = subprocess.Popen([python,'-u',str(job['entry']),str(job['task'])],
                        env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    born = psutil.Process(p.pid).create_time()
                except BaseException:
                    stream.close()
                    raise
                active.append(dict(name=name,process=p,born=born,stream=stream,job=job,
                                   estimate=job['estimate'],private=0,threads=1))
                pending.remove(job)
                log.event('case.start',f"启动 {name}，预留{job['estimate']/GIB:.2f}GiB，PID {p.pid}",pid=p.pid,estimate_bytes=job['estimate'])
                # The next admission must use fresh facts, not the same free RAM
                # counted as if the just-launched job had no reservation.
                fresh = policy(snapshot(),pending or jobs,workers)
            if pending and not active:
                raise MemoryError('Remaining full-size cases do not fit current CPU/RAM budget; committed smaller cases preserved')
            if now-last_console >= 20:
                log.event('stage.heartbeat',f"{stage_name}：已完成{done}/{len(jobs)}，运行{len(active)}，待处理{len(pending)}；可用内存{(fresh.get('available_memory_bytes') or 0)/GIB:.1f}GiB")
                last_console = now
            if pending or active:
                time.sleep(.15)
    finally:
        for a in active:
            stop_owned(a)
            a['stream'].close()
        write_json(output/'resources'/f'{stage_name}-observed.json',dict(completed=done,
            peak_simultaneous_jobs=peak_jobs, peak_worker_group_rss_bytes=peak_rss,
            peak_worker_group_uss_bytes=peak_uss, case_peak_rss_bytes=peaks_by_job,
            recoveries=recoveries, sample_interval_seconds=1,
            note='sampled memory, not a hard limit or a measured speedup'))
    return dict(completed=done,peak_jobs=peak_jobs,recoveries=recoveries)
