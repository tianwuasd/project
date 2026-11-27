from contextlib import contextmanager
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from ...artifacts.cache import release_private_mapping
from ...acceleration.accounting import available
from ...observability.events import event
from .targets import build_heatmap_targets


def grid_plan(grid, halo, block_bytes, per_voxel=512):
    extent = list(grid)
    def amount(values):
        # 距离变换和监督池化的工作区包含halo及32倍数补齐，只核算一次。
        return math.prod(math.ceil((4*n+2*h)/32)*32 for n,h in zip(values,halo))*per_voxel
    minimum = amount((1,1,1))
    if minimum > block_bytes:
        raise MemoryError(f'定位监督最小halo工作区需要{minimum}字节，当前块预算为{block_bytes}字节')
    while amount(extent) > block_bytes:
        axis = max(range(3),key=lambda i:extent[i])
        extent[axis] = max(1,(extent[axis]+1)//2)
    for axis in sorted(range(3),key=lambda i:extent[i]):
        low,high = extent[axis],grid[axis]
        while low < high:
            middle = (low+high+1)//2
            trial = list(extent)
            trial[axis] = middle
            if amount(trial) <= block_bytes:
                low = middle
            else:
                high = middle-1
        extent[axis] = low
    return dict(extent=tuple(extent),minimum_workspace_bytes=minimum,workspace_bytes=amount(extent))


def input_plan(shape, spacing, block_bytes):
    if len(shape) != 3 or min(shape) < 1 or len(spacing) != 3 or min(spacing) <= 0 or block_bytes < 1:
        raise ValueError('定位器分块需要有效的三维尺寸、毫米间距和正数预算')
    pooled = tuple(math.ceil(int(n)/32)*8 for n in shape)
    halo = tuple((math.ceil(8/float(s)/4)+1)*4 for s in spacing)
    block = grid_plan(pooled,halo,block_bytes)
    count = math.prod(pooled)
    return dict(pooled_shape=pooled,halo=halo,**block,
                blocks=math.prod(math.ceil(n/b) for n,b in zip(pooled,block['extent'])),retained_bytes=count*23,
                backward_bytes=count*2048,device_workspace_bytes=count*512+128*2**20)


def training_plan(shape, spacing, settings):
    return input_plan(shape,spacing,settings.get('localizer_block_bytes',2*2**30))


def sections(shape, extent):
    for d in range(0,shape[0],extent[0]):
        for h in range(0,shape[1],extent[1]):
            for w in range(0,shape[2],extent[2]):
                yield tuple(slice(a,min(n,a+b)) for a,b,n in zip((d,h,w),extent,shape))


def planned_sections(grid, halo, maximum, guard=None, per_voxel=512):
    minimum = grid_plan((1,1,1),halo,maximum,per_voxel)['minimum_workspace_bytes']
    pending = [tuple(slice(0,n) for n in grid)]
    done = 0
    while pending:
        budget = maximum
        if guard is not None:
            handle = guard.client.handle
            with handle.condition:
                budget = min(budget,max(minimum,available(handle.state,protect_pool=True)))
        region = pending.pop()
        shape = tuple(s.stop-s.start for s in region)
        plan = grid_plan(shape,halo,budget,per_voxel)
        section = tuple(slice(s.start,s.start+n) for s,n in zip(region,plan['extent']))
        # 三个互不重叠的余区保留可扩大的形状，后续预算变化不会重复或遗漏体素。
        for axis in reversed(range(3)):
            rest = tuple(slice(s.start,section[i].stop) if i<axis else
                         slice(section[i].stop,s.stop) if i==axis else s for i,s in enumerate(region))
            if all(s.stop>s.start for s in rest):
                pending.append(rest)
        done += 1
        total = done+sum(math.prod(math.ceil((s.stop-s.start)/b) for s,b in zip(rest,plan['extent'])) for rest in pending)
        yield section,plan['workspace_bytes'],done,total


def label_region(label, axes):
    indices = np.ix_(*(a.clip(0,n-1) for a,n in zip(axes,label.shape)))
    raw = np.array(label[indices],copy=True)
    for axis,(positions,n) in enumerate(zip(axes,label.shape)):
        selection = [slice(None)]*3
        selection[axis] = (positions < 0) | (positions >= n)
        raw[tuple(selection)] = -1
    return raw


@contextmanager
def boundary(guard, amount=0):
    if guard is None:
        yield
    else:
        with guard.admit(amount,label='q_input_block') as ticket:
            if ticket is None:
                raise InterruptedError('定位器输入分块已取消')
            guard.check_active()
            yield


def target_block(entry, dataset, section, halo):
    from .adapter import allowed

    axes = [np.arange(t.start*4-h,t.stop*4+h) for t,h in zip(section,halo)]
    raw = label_region(entry['label'],axes)
    source_region = [(max(0,min(len(a),-int(a[0]))),max(0,min(len(a),n-int(a[0]))))
                     for a,n in zip(axes,entry['label'].shape)]
    heat,mask = build_heatmap_targets(allowed(raw,entry['unknown_classes'],dataset),raw>=0,
                                     entry['processed_spacing'],source_region=source_region)
    crop = tuple(slice(h//4,h//4+t.stop-t.start) for h,t in zip(halo,section))
    return heat[(0,slice(None),*crop)].clone(),mask[(0,slice(None),*crop)].clone()


def pooled_block(entry, section, use_label):
    shape = entry['image'].shape[1:]
    axes = [np.arange(t.start*4,t.stop*4) for t in section]
    indices = np.ix_(*(a.clip(0,n-1) for a,n in zip(axes,shape)))
    raw = np.array(entry['image'][(0, *indices)],dtype=np.float32,copy=True)
    valid = label_region(entry['label'],axes)>=0 if use_label else np.ones(raw.shape,dtype=bool)
    if not use_label:
        for axis,(positions,n) in enumerate(zip(axes,shape)):
            sizes = [1,1,1]
            sizes[axis] = len(positions)
            valid &= ((positions>=0)&(positions<n)).reshape(sizes)
    return (F.avg_pool3d(torch.from_numpy(raw)[None,None],4,4)[0],
            F.avg_pool3d(torch.from_numpy(valid)[None,None].float(),4,4)[0])


def pooled_image(entry, block_bytes, *, use_label=False, guard=None, parallel=None):
    grid = tuple(math.ceil(n/32)*8 for n in entry['image'].shape[1:])
    image,valid = torch.empty((1,*grid)),torch.empty((1,*grid))
    if parallel is not None:
        def consume(section,value):
            image[(slice(None),*section)],valid[(slice(None),*section)] = value
        parallel.sections(grid,(0,0,0),lambda section:pooled_block(entry,section,use_label),consume,per_voxel=16)
        return image[None],valid[None]
    for section,amount,_,_ in planned_sections(grid,(0,0,0),block_bytes,guard,per_voxel=16):
        with boundary(guard,amount):
            x,mask = pooled_block(entry,section,use_label)
            image[(slice(None),*section)],valid[(slice(None),*section)] = x,mask
        for value in (entry['image'],entry.get('label')):
            if isinstance(value,np.memmap):
                release_private_mapping(value)
    return image[None],valid[None]


def prepare_inputs(entry, dataset, block_bytes, *, guard=None, log_dir=None, heartbeat_seconds=15,parallel=None):
    shape = entry['label'].shape
    spacing = entry['processed_spacing']
    plan = input_plan(shape,spacing,block_bytes)
    grid = plan['pooled_shape']
    image,valid = np.empty((1,*grid),np.float32),np.empty((1,*grid),np.float32)
    heat,mask = np.empty((3,*grid),np.float32),np.empty((3,*grid),bool)
    if parallel is not None:
        def produce(section):
            return (*pooled_block(entry,section,True),*target_block(entry,dataset,section,plan['halo']))
        def consume(section,values):
            x,validity,block_heat,block_mask = values
            image[(slice(None),*section)],valid[(slice(None),*section)] = x.numpy(),validity.numpy()
            heat[(slice(None),*section)],mask[(slice(None),*section)] = block_heat.numpy(),block_mask.numpy()
        parallel.sections(grid,plan['halo'],produce,consume)
        return tuple(torch.from_numpy(value)[None] for value in (image,valid,heat,mask))
    total = plan['blocks']
    began,last_log = perf_counter(),-float('inf')
    if log_dir is not None:
        event(Path(log_dir),'q_localizer_input_plan',case_id=entry['case_id'],source_shape=list(shape),
              pooled_shape=list(grid),extent=list(plan['extent']),total=total,
              block_bytes=plan['workspace_bytes'],budget_bytes=block_bytes)
    for section,amount,index,total in planned_sections(grid,plan['halo'],block_bytes,guard):
        with boundary(guard,amount):
            # 半径外的距离被截断；原网格halo保留中心区域的距离、可靠掩膜及池化语义。
            x,validity = pooled_block(entry,section,True)
            block_heat,block_mask = target_block(entry,dataset,section,plan['halo'])
            image[(slice(None),*section)],valid[(slice(None),*section)] = x.numpy(),validity.numpy()
            heat[(slice(None),*section)],mask[(slice(None),*section)] = block_heat.numpy(),block_mask.numpy()
            del x,validity,block_heat,block_mask
        for value in (entry['image'],entry['label']):
            if isinstance(value,np.memmap):
                release_private_mapping(value)
        now = perf_counter()
        if log_dir is not None and (index == total or now-last_log >= heartbeat_seconds):
            event(Path(log_dir),'q_localizer_inputs',case_id=entry['case_id'],completed=index,total=total,
                  source_shape=list(shape),pooled_shape=list(grid),block_bytes=amount,
                  extent=[s.stop-s.start for s in section],total_estimated=True,
                  seconds=now-began,retained_bytes=plan['retained_bytes'])
            last_log = now
    return tuple(torch.from_numpy(value)[None] for value in (image,valid,heat,mask))
