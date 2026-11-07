"""VISTA v2 stage ceilings + fresh headroom; each admitted process uses one CPU thread."""
import math
import os
import time
import psutil
from .vendor.vista_live_resources import snapshot as linux_snapshot

GIB=1024**3


def available_memory_bytes():
    # 每批供数只读取 RAM/cgroup 余量，不等待 CPU 利用率采样。
    if os.name == 'nt':
        return psutil.virtual_memory().available
    return linux_snapshot(sample_seconds=0)['available_memory_bytes']


def preprocessing_memory(input_voxels, output_voxels=0):
    """Sequential nnU-Net working-set reservation, including interpolation buffers."""
    return 1200 * 2**20 + 40 * max(input_voxels, output_voxels)


def snapshot():
    if os.name != 'nt':
        return linux_snapshot(sample_seconds=.1)
    missing=[]
    try: ids=psutil.Process().cpu_affinity()
    except (psutil.Error,AttributeError): ids=[];missing.append('CPU affinity unavailable')
    loads=psutil.cpu_percent(interval=.1,percpu=True)
    available=sum(max(0,1-loads[i]/100) for i in ids if i<len(loads)) if ids else None
    return {'collected_at':time.time(),'cpu_permitted':len(ids) if ids else None,
            'cpu_available':available,'cpu_ids':ids,'affinity_source':'psutil.cpu_affinity',
            'available_memory_bytes':psutil.virtual_memory().available,
            'cgroup_evidence':[],'missing_evidence':missing,'io_pressure':None,
            'cpu_availability_source':'permitted_cpu_idle_sample'}


def budget(facts,policy):
    available=facts.get('available_memory_bytes'); cpu=facts.get('cpu_available')
    reserve=max(policy.get('reserve_gib',4)*GIB, (1-policy.get('ram_fraction',.8))*available) if available is not None else None
    memory=max(0, min(int(policy.get('ram_fraction',.8)*available),int(available-reserve))) if available is not None else None
    # Linux missing finite cgroup headroom is propagated as unknown by the v2 sampler.
    return {'cpu_slots':math.floor(policy.get('cpu_fraction',.8)*cpu) if cpu is not None else None,
            'memory_bytes':memory,'reserve_bytes':reserve,'available_memory_bytes':available,
            'workers':min(policy.get('max_workers',2),policy.get('read_concurrency',2)),
            'missing_evidence':facts.get('missing_evidence',[])}


def admit_case(stage,fresh,active,estimate):
    if any(v is None for v in (stage['cpu_slots'],stage['memory_bytes'],fresh['cpu_slots'],fresh['memory_bytes'])):
        return {'allowed':False,'reason':'unknown_resource_capacity'}
    reserved=sum(a['reserved'] for a in active)
    outstanding=sum(max(0,a['reserved']-max(0,a.get('private_resident',0))) for a in active)
    if len(active)>=stage['workers']: reason='worker_or_io_limit'
    elif sum(a['threads'] for a in active)+1>stage['cpu_slots']: reason='stage_cpu_limit'
    elif fresh['cpu_slots']<1: reason='fresh_cpu_headroom'
    elif reserved+estimate>stage['memory_bytes']: reason='stage_ram_limit'
    elif outstanding+estimate>fresh['memory_bytes']: reason='fresh_ram_headroom'
    else: reason='admitted'
    return {'allowed':reason=='admitted','reason':reason,'reserved_bytes':reserved,
            'outstanding_bytes':outstanding,'estimate_bytes':estimate}


def estimate_case(image_path,label_path,policy,stage='full'):
    if stage!='full': return 384*1024**2
    import nibabel as nib
    try:
        image=nib.load(str(image_path));label=nib.load(str(label_path))
        n=math.prod(label.shape); dtype=label.get_data_dtype().itemsize
        if label.dataobj.slope!=1 or label.dataobj.inter!=0: dtype=8
        # Full label + masks + cropped int32 components + worst-case bincount temporaries.
        # Image slabs are released before the label stage; sum is deliberately conservative.
        # Data + boolean mask + int32 CCL + worst-case int64 sizes (at most N/2
        # isolated components) + one byte/N slack. CCL histogram is slab bounded.
        return int(n*(dtype+10)+math.prod(image.shape[:2])*policy['chunk_slices']*64+384*1024**2)
    except (OSError,ValueError): return 512*1024**2


def thread_environment():
    return {key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                               'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')}
