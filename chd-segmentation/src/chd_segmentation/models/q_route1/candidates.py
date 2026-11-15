from dataclasses import dataclass
from itertools import product
import math
import numpy as np


@dataclass(frozen=True)
class Candidate:
    start: tuple
    source: str
    task: int
    score: float

    @property
    def center(self):
        return tuple(s+63.5 for s in self.start)

    @property
    def core_start(self):
        return tuple(s+16 for s in self.start)


def _candidate(center, source, task=-1, score=0.):
    return Candidate(tuple(math.floor(c-63) for c in center),source,task,float(score))


def _iou(a,b,shape):
    aa = [(max(0,s),min(n,s+96)) for s,n in zip(a.core_start,shape)]
    bb = [(max(0,s),min(n,s+96)) for s,n in zip(b.core_start,shape)]
    va = math.prod(max(0,y-x) for x,y in aa)
    vb = math.prod(max(0,y-x) for x,y in bb)
    inter = math.prod(max(0,min(y,v)-max(x,u)) for (x,y),(u,v) in zip(aa,bb))
    return inter/(va+vb-inter) if va+vb-inter else 0.


def _grid_box(start, size, grid_shape):
    low = [max(0,min(n,math.ceil((s-1.5)/4))) for s,n in zip(start,grid_shape)]
    high = [max(0,min(n,math.ceil((s+size-1.5)/4))) for s,n in zip(start,grid_shape)]
    return tuple(low),tuple(high)


def select_candidates(heatmap, shape, quotas=(4,4,8), threshold=.2, dedup_iou=.8):
    heatmap = np.asarray(heatmap,dtype=np.float32)
    if heatmap.ndim != 4 or heatmap.shape[0] != 3 or not np.isfinite(heatmap).all():
        raise ValueError('定位图需要3张有限三维数组')
    if np.any(heatmap < 0) or np.any(heatmap > 1):
        raise ValueError('定位图范围必须在0到1之间')
    selected = []
    for fractions in product((.25,.75),repeat=3):
        candidate = _candidate(tuple(f*n for f,n in zip(fractions,shape)),'fixed')
        if all(_iou(candidate,c,shape) < dedup_iou for c in selected):
            selected.append(candidate)
    coords = np.stack(np.meshgrid(*(np.arange(n)*4+1.5 for n in heatmap.shape[1:]),indexing='ij'),-1)
    valid = (coords < np.asarray(shape)).all(-1)
    remaining = heatmap * (heatmap >= threshold) * valid[None]
    def clear(candidate):
        low,high = _grid_box(tuple(s+24 for s in candidate.start),80,heatmap.shape[1:])
        remaining[(slice(None),*(slice(a,b) for a,b in zip(low,high)))] = 0
    for c in selected:
        clear(c)
    centers = coords[valid]
    starts = np.floor(centers-63).astype(np.int64)
    lows = np.maximum(0,np.ceil((starts+16-1.5)/4).astype(np.int64))
    highs = np.minimum(np.array(heatmap.shape[1:]),np.ceil((starts+112-1.5)/4).astype(np.int64))
    lows = np.minimum(lows,np.array(heatmap.shape[1:]))
    highs = np.maximum(highs,0)
    for task,quota in enumerate(quotas):
        for _ in range(quota):
            integral = np.pad(remaining[task].astype(np.float64),((1,0),)*3).cumsum(0).cumsum(1).cumsum(2)
            scores = np.zeros(len(centers),dtype=np.float64)
            for bits in product((0,1),repeat=3):
                indices = np.where(np.array(bits)[None],highs,lows)
                scores += (-1)**(3-sum(bits))*integral[tuple(indices.T)]
            scores /= 24**3
            order = np.argsort(-scores,kind='stable')
            chosen = None
            for i in order:
                if scores[i] <= 0:
                    break
                candidate = _candidate(centers[i],'learned',task,scores[i])
                if all(_iou(candidate,c,shape) < dedup_iou for c in selected):
                    chosen = candidate
                    break
            if chosen is None:
                break
            selected.append(chosen)
            clear(chosen)
    if len(selected) > 24:
        raise ValueError('候选超过24个预算')
    return selected

