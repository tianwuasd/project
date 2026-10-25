"""Shared CPU/RAM budget from affinity and cgroup; never infer a whole-host quota."""
import math
import os
import time
import shutil
from pathlib import Path
from server_data import write_json


def probe():
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    quota=None
    try:
        limit,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if limit!='max':quota=float(limit)/float(period)
    except (OSError,ValueError):
        try:
            limit=float(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period=float(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if limit>0:quota=limit/period
        except (OSError,ValueError):pass
    if quota is not None:cpus=min(cpus,max(1,math.floor(quota)))
    available=None
    try:
        entries=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        available=int(entries['MemAvailable'].split()[0])*1024
        cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used=int(Path('/sys/fs/cgroup/memory.current').read_text())
        if cap!='max':available=min(available,max(0,int(cap)-used))
    except (OSError,ValueError,KeyError):pass
    # Two /proc/stat samples avoid treating other users' busy CPU as free quota.
    idle_fraction=None
    try:
        def sample():
            return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
        first=sample();time.sleep(.2);second=sample()
        delta=[b-a for a,b in zip(first,second)]
        idle_fraction=(delta[3]+delta[4])/max(1,sum(delta[:8]))
    except (OSError,ValueError,IndexError):pass
    if idle_fraction is not None:cpus=min(cpus,max(1,math.floor(cpus*idle_fraction)))
    shared=None;io_pressure=None
    try:shared=shutil.disk_usage('/dev/shm').free
    except OSError:pass
    try:
        first=Path('/proc/pressure/io').read_text().splitlines()[0].split()[1:]
        io_pressure={key:float(value) for key,value in (item.split('=') for item in first)}
    except (OSError,ValueError,IndexError):pass
    return dict(cpu_slots=max(1,cpus),cgroup_cpu_quota=quota,cpu_idle_fraction=idle_fraction,available_memory_bytes=available,
                shared_memory_available_bytes=shared,io_pressure=io_pressure,
                load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None)


def allocate(snapshot, parallel=1):
    parallel=max(1,int(parallel))
    slots=max(1,int(snapshot['cpu_slots']))
    # Independent nnU-Net jobs have two loading pools; retain its serial loader until
    # measured preflight selects otherwise. Thread totals remain within one budget.
    parallel=min(parallel,slots)
    memory=snapshot.get('available_memory_bytes')
    if memory is not None:parallel=min(parallel,max(1,int(memory)//(4*1024**3)))
    cpu_workers=min(slots,8)
    if (snapshot.get('io_pressure') or {}).get('avg10',0)>=10:cpu_workers=min(cpu_workers,2)
    if memory is not None:cpu_workers=min(cpu_workers,max(1,int(memory)//(4*1024**3)))
    return dict(snapshot,parallel_workers=parallel,threads_per_worker=max(1,min(4,slots//parallel)),
                cpu_tail_workers=max(1,cpu_workers),cpu_case_threads=1,nnunet_augmentation_workers=0,
                selection='affinity/cgroup/CPU-idle/RAM/I-O-pressure bounded initial budget; serial DA avoids SHM pools; not a measured throughput optimum')


def configure(job,plan,env):
    budget=allocate(probe(),plan.get('gpu_count') or 1)
    write_json(Path(job)/'resource-plan.json',budget)
    env['WHOLE_HEART_CPU_THREADS']=str(budget['threads_per_worker'])
    env['WHOLE_HEART_CPU_TAIL_WORKERS']=str(budget['cpu_tail_workers'])
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]=str(budget['threads_per_worker'])
    env['WHOLE_HEART_GPU_LIMIT']=str(min(plan.get('gpu_count') or 1,budget['parallel_workers']))
    return budget


def case_budget(work,paths,spacing=None):
    """Read headers only; bound raw/resampled logits and temporary arrays per case."""
    import nibabel as nib
    largest=0
    for path in paths:
        image=nib.load(str(path));voxels=math.prod(image.shape)
        if spacing:
            voxels=max(voxels,math.prod(max(1,math.ceil(n*s/t)) for n,s,t in zip(image.shape,image.header.get_zooms(),spacing)))
        largest=max(largest,voxels*128+1024**3)
    snapshot=probe();limit=max(1,min(8,snapshot['cpu_slots'],int(os.environ.get('WHOLE_HEART_CPU_TAIL_WORKERS','1'))))
    if snapshot.get('available_memory_bytes') is not None:
        limit=min(limit,max(1,int(snapshot['available_memory_bytes']*.75)//max(1,largest)))
    if (snapshot.get('io_pressure') or {}).get('avg10',0)>=10:limit=min(limit,2)
    report=dict(snapshot,workers=limit,threads_per_worker=1,estimated_bytes_per_case=largest,
                rule='header raw/resampled voxels * 128 + 1 GiB, <=75% available RAM; bounded estimate, not peak measurement')
    write_json(Path(work)/'00_admin/cpu-case-budget.json',report)
    return report


def apply_training_cpu(effective,torch):
    """Apply measured settings even when NumPy/OpenMP libraries are already loaded."""
    count=int(effective['threads'])
    os.environ['WHOLE_HEART_CPU_THREADS']=str(count)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=str(count)
    from threadpoolctl import threadpool_limits
    limiter=threadpool_limits(limits=count)
    torch.set_num_threads(count)
    import sys
    if 'numexpr' in sys.modules:sys.modules['numexpr'].set_num_threads(count)
    return limiter


def torch_threads(torch):
    count=max(1,int(os.environ.get('WHOLE_HEART_CPU_THREADS','1')))
    torch.set_num_threads(count)
    return count
