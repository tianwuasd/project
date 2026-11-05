"""Complete lossless read-back, aligned with actual compressed chunk geometry.

Visit every logical voxel exactly once, including partial edge chunks. A fixed
thin slab can revisit the same compressed chunk many times on large volumes.
This changes verification access, never compression, interpolation or QC scope.
"""
from itertools import product
import math
import numpy as np


def verify_full_array(stored,expected,progress=None):
    if stored.shape!=expected.shape or stored.dtype!=expected.dtype:
        raise ValueError('Stored shape/dtype differs from memory')
    chunks=tuple(int(x) for x in stored.chunks)
    if len(chunks)!=expected.ndim or any(x<1 for x in chunks):raise ValueError('Invalid stored chunk geometry')
    covered=0;tiles=0;total=math.prod(expected.shape)
    for origin in product(*(range(0,n,c) for n,c in zip(expected.shape,chunks))):
        slices=tuple(slice(s,min(n,s+c)) for s,n,c in zip(origin,expected.shape,chunks))
        if not np.array_equal(stored[slices],expected[slices]):
            raise ValueError('Stored array failed full read-back verification at '+str(origin))
        tiles+=1;covered+=math.prod(s.stop-s.start for s in slices)
        if progress:progress(covered,total)
    if covered!=total:raise RuntimeError('Incomplete read-back coverage')
    return dict(verified_voxels=covered,tiles=tiles,chunks=list(chunks),verification='all_values')
