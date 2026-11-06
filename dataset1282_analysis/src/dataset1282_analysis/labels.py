"""Numeric labels remain numeric; topology is descriptive, never an automatic repair."""
from collections import Counter
import numpy as np
from scipy import ndimage
from .voxels import NiftiStream


def analyze_label(path, policy):
    source=NiftiStream(path,policy.get('chunk_slices',8))
    dtype=source.proxy.dtype if source.proxy.slope==1 and source.proxy.inter==0 else np.dtype('float64')
    data=np.empty(source.shape,dtype=dtype,order='F')
    counts=Counter(); nonfinite=0; noninteger=0;foreground=0;over_limit=False
    maximum=int(policy.get('max_label_values',128))
    for z,block in source.blocks():
        data[:,:,z:z+block.shape[2]]=block
        values=block.ravel(order='F'); finite=np.isfinite(values)
        nonfinite+=int(values.size-finite.sum()); valid=values[finite]
        foreground+=int(np.count_nonzero(valid))
        if np.issubdtype(valid.dtype,np.floating): noninteger+=int(np.count_nonzero(valid!=np.floor(valid)))
        if over_limit: continue
        if len(valid) and np.issubdtype(valid.dtype,np.integer) and valid.min()>=0 and valid.max()<4096:
            histogram=np.bincount(valid.astype(np.int64),minlength=int(valid.max())+1)
            unique=np.flatnonzero(histogram);numbers=histogram[unique]
        else:
            unique,numbers=np.unique(valid,return_counts=True)
        if len(unique)>maximum or len(set(counts)|set(float(k) for k in unique))>maximum:
            over_limit=True;counts.clear()
        else:
            counts.update({float(k):int(v) for k,v in zip(unique,numbers)})
    structure=ndimage.generate_binary_structure(3,policy.get('connectivity',1))
    classes=[]
    for value,count in sorted(counts.items()):
        item={'value':value,'voxels':count,'fraction_all_voxels':count/data.size,
              'components':None,'largest_component_voxels':None,'bbox_voxel_inclusive':None,'touches_boundary':None}
        if value!=0:
            mask=data==value
            coords=[np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j!=i))) for i in range(3)]
            bounds=[(int(c[0]),int(c[-1])) for c in coords]
            crop=tuple(slice(a,b+1) for a,b in bounds)
            components,number=ndimage.label(mask[crop],structure=structure)
            # bincount of only foreground component values avoids allocating a full int64 volume.
            # Count per plane: avoid a full-volume int64 copy inside np.bincount.
            sizes=np.zeros(number+1,dtype=np.int64)
            for plane in np.moveaxis(components,2,0):
                if number<100000:
                    sizes+=np.bincount(plane.ravel(),minlength=number+1)
                else:
                    values,frequencies=np.unique(plane,return_counts=True)
                    sizes[values]+=frequencies
            item.update(components=int(number),largest_component_voxels=int(sizes[1:].max()) if number else 0,
                        bbox_voxel_inclusive=[list(b) for b in bounds],
                        touches_boundary=any(a==0 or b==data.shape[i]-1 for i,(a,b) in enumerate(bounds)))
            del mask,components,sizes
        classes.append(item)
    return {'voxel_count':int(data.size),'finite_count':int(data.size)-nonfinite,
            'nonfinite_count':nonfinite,'noninteger_count':noninteger,
            'foreground_voxels':foreground,
            'empty':foreground==0,'classes':classes,
            'per_class_statistics_status':'not_computed_cardinality_limit' if over_limit else 'complete',
            'max_label_values':maximum,'distinct_values_lower_bound':maximum+1 if over_limit else len(counts),
            'connectivity':policy.get('connectivity',1),'sha256':source.sha256,
            'complete_voxel_read':True,'label_semantics_status':'unverified_numeric_labels'}
