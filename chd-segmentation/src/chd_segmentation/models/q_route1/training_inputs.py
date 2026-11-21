from contextlib import contextmanager
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from ...artifacts.cache import release_private_mapping
from ...observability.events import event
from ..stunet.streaming import tile_extent
from .targets import build_heatmap_targets


def input_plan(shape, spacing, block_bytes):
    pooled = tuple(math.ceil(int(n)/32)*8 for n in shape)
    halo = tuple((math.ceil(8/float(s)/4)+1)*4 for s in spacing)
    extent = list(tile_extent(pooled,max(1,int(block_bytes)//(64*512))))
    def amount():
        return math.prod(4*n+2*h for n,h in zip(extent,halo))*512
    while amount() > block_bytes and max(extent) > 1:
        axis = max(range(3),key=lambda i:extent[i])
        extent[axis] = max(1,extent[axis]//2)
    count = math.prod(pooled)
    return dict(pooled_shape=pooled,halo=halo,extent=tuple(extent),
                workspace_bytes=amount(),retained_bytes=count*23,
                backward_bytes=count*2048,device_workspace_bytes=count*512+128*2**20)


def sections(shape, extent):
    for d in range(0,shape[0],extent[0]):
        for h in range(0,shape[1],extent[1]):
            for w in range(0,shape[2],extent[2]):
                yield tuple(slice(a,min(n,a+b)) for a,b,n in zip((d,h,w),extent,shape))


def label_region(label, axes):
    indices = np.ix_(*(a.clip(0,n-1) for a,n in zip(axes,label.shape)))
    raw = np.array(label[indices],copy=True)
    for axis,(positions,n) in enumerate(zip(axes,label.shape)):
        selection = [slice(None)]*3
        selection[axis] = (positions < 0) | (positions >= n)
        raw[tuple(selection)] = -1
    return raw


@contextmanager
def boundary(guard):
    if guard is None:
        yield
    else:
        guard.check_active()
        yield


def prepare_inputs(entry, dataset, block_bytes, *, guard=None, log_dir=None, heartbeat_seconds=15):
    from .adapter import allowed

    shape = entry['label'].shape
    spacing = entry['processed_spacing']
    plan = input_plan(shape,spacing,block_bytes)
    grid = plan['pooled_shape']
    image,valid = np.empty((1,*grid),np.float32),np.empty((1,*grid),np.float32)
    heat,mask = np.empty((3,*grid),np.float32),np.empty((3,*grid),bool)
    total = math.prod(math.ceil(n/b) for n,b in zip(grid,plan['extent']))
    began,last_log = perf_counter(),-float('inf')
    for index,section in enumerate(sections(grid,plan['extent']),1):
        with boundary(guard):
            axes = [np.arange(t.start*4,t.stop*4) for t in section]
            indices = np.ix_(*(a.clip(0,n-1) for a,n in zip(axes,shape)))
            raw_image = np.array(entry['image'][0][indices],dtype=np.float32,copy=True)
            raw_label = label_region(entry['label'],axes)
            image[(slice(None),*section)] = F.avg_pool3d(torch.from_numpy(raw_image)[None,None],4,4)[0].numpy()
            valid[(slice(None),*section)] = F.avg_pool3d(torch.from_numpy(raw_label>=0)[None,None].float(),4,4)[0].numpy()
            del raw_image,raw_label
            axes = [np.arange(t.start*4-h,t.stop*4+h) for t,h in zip(section,plan['halo'])]
            raw = label_region(entry['label'],axes)
            source_region = [(max(0,min(len(a),-int(a[0]))),max(0,min(len(a),n-int(a[0]))))
                             for a,n in zip(axes,shape)]
            # 半径外的距离被截断；原网格halo保留中心区域的距离、可靠掩膜及池化语义。
            block_heat,block_mask = build_heatmap_targets(allowed(raw,entry['unknown_classes'],dataset),raw>=0,
                                                        spacing,source_region=source_region)
            crop = tuple(slice(h//4,h//4+t.stop-t.start) for h,t in zip(plan['halo'],section))
            heat[(slice(None),*section)] = block_heat[(0,slice(None),*crop)].numpy()
            mask[(slice(None),*section)] = block_mask[(0,slice(None),*crop)].numpy()
            del raw,block_heat,block_mask
        for value in (entry['image'],entry['label']):
            if isinstance(value,np.memmap):
                release_private_mapping(value)
        now = perf_counter()
        if log_dir is not None and (index == total or now-last_log >= heartbeat_seconds):
            event(Path(log_dir),'q_localizer_inputs',case_id=entry['case_id'],completed=index,total=total,
                  source_shape=list(shape),pooled_shape=list(grid),block_bytes=plan['workspace_bytes'],
                  seconds=now-began,retained_bytes=plan['retained_bytes'])
            last_log = now
    return tuple(torch.from_numpy(value)[None] for value in (image,valid,heat,mask))
