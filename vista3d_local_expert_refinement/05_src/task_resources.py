"""Shared CPU/RAM budget from affinity and cgroup; never infer a whole-host quota."""
import math
import os
from pathlib import Path
from server_data import write_json,read_json


def probe():
    """Compatibility wrapper; each invocation collects fresh stage facts."""
    from live_resources import snapshot
    return snapshot()


def allocate(snapshot, parallel=1):
    from live_resources import case_policy
    capacity=case_policy(snapshot,max(1,int(parallel)),4*1024**3,supported_threads=4)
    tail=case_policy(snapshot,max(0,int(snapshot.get('cpu_budget',snapshot.get('cpu_slots')) or 1)),
                     4*1024**3,supported_threads=1)
    if capacity['workers']<1:
        raise MemoryError('known insufficient CPU/RAM stage capacity: '+capacity['capacity_status'])
    return dict(snapshot,parallel_workers=capacity['workers'],threads_per_worker=capacity['threads_per_worker'],
                cpu_tail_workers=tail['workers'],cpu_case_threads=1,nnunet_augmentation_workers=0,
                read_concurrency=tail['read_concurrency'],capacity_status=capacity['capacity_status'],
                reasons=capacity['reasons']+tail['reasons'],
                selection='fresh direct CPU80%/RAM80% stage policy; I/O read admission separate from compute')


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
    paths=list(paths)
    largest=0
    for path in paths:
        image=nib.load(str(path));voxels=math.prod(image.shape)
        if spacing:
            voxels=max(voxels,math.prod(max(1,math.ceil(n*s/t)) for n,s,t in zip(image.shape,image.header.get_zooms(),spacing)))
        largest=max(largest,voxels*128+1024**3)
    from live_resources import case_policy
    snapshot=probe()
    policy=case_policy(snapshot,len(paths),largest,supported_threads=1)
    report=dict(snapshot | policy,
                rule='header raw/resampled voxels *128 +1 GiB; aggregate CPU80%/RAM80%; no fixed case cap')
    write_json(Path(work)/'00_admin/cpu-case-budget.json',report)
    return report


def prediction_budget(work,records,maximum):
    """Aggregate host gate for GPU infer, using raw and per-fold resampled headers."""
    import nibabel as nib
    work=Path(work)
    inventory={c['case_id']:c for c in read_json(work/'04_data/manifests/case_inventory.json')['cases']}
    splits=read_json(work/'04_data/manifests/splits_v1.json')['folds']
    headers=[];seen=set()
    for record in records:
        key=(record['fold'],record['case_id'])
        if key in seen:continue
        seen.add(key)
        image=nib.load(str(inventory[record['case_id']]['image']))
        raw=math.prod(image.shape)
        dataset=splits[record['fold']]['dataset_id']
        plan=read_json(work/f'04_data/derived/nnUNet_preprocessed/Dataset{dataset:03d}_{record["fold"]}/WholeheartBaselinePlans.json')
        config=plan['configurations']['3d_fullres']
        spacing=config.get('spacing')
        # NibabelIO presents z,y,x to nnU-Net; its planned spacing uses that order.
        shape=list(reversed(image.shape));zooms=list(reversed(image.header.get_zooms()[:3]))
        axes=plan.get('transpose_forward',[0,1,2])
        shape=[shape[i] for i in axes];zooms=[zooms[i] for i in axes]
        resampled=[max(1,math.ceil(n*s/t)) for n,s,t in zip(shape,zooms,spacing)] if spacing else shape
        estimated=raw*16+max(raw,math.prod(resampled))*128+1024**3
        headers.append(dict(fold=record['fold'],case_id=record['case_id'],raw_shape=list(image.shape),
            raw_spacing=[float(v) for v in image.header.get_zooms()[:3]],planned_spacing=spacing,
            transpose_forward=axes,resampled_shape_plans=resampled,estimated_bytes=estimated,
            spacing_evidence='planned 3d_fullres spacing' if spacing else 'missing; raw-grid estimate only'))
    from live_resources import case_policy
    snapshot=probe()
    largest=max((row['estimated_bytes'] for row in headers),default=0)
    policy=case_policy(snapshot,min(int(maximum),len(headers)),largest,
                       supported_threads=max(1,int(os.environ.get('WHOLE_HEART_CPU_THREADS','1'))))
    report=dict(snapshot | policy,cases=headers,
        rule='raw voxels *16 + max(raw,resampled) voxels *128 +1 GiB; aggregate CPU80%/RAM80%; separate I/O read admission')
    write_json(work/'00_admin/prediction-host-budget.json',report)
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
