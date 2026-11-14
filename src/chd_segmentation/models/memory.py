import math

import nibabel as nib
import numpy as np

from ..datasets.manifest import LABELS


GIB=2**30


def case_estimates(config,prepared,stage):
    rows=prepared['manifest']['cases']
    if not rows:
        raise ValueError('模型内存预算缺少实际病例')
    roi=prepared['technical']['roi_xyz']
    records=[]
    classes=len(LABELS[prepared['dataset']])
    for row in rows:
        shape=np.asarray([end-begin for begin,end in roi] if roi else nib.load(row['image']).shape,dtype=np.int64)
        if shape.shape!=(3,) or (shape<=0).any():
            raise ValueError('模型内存预算需要有效三维网格')
        native=math.prod(map(int,shape))
        if config['model']=='q_route1':
            spacing=np.asarray(prepared['plan']['configurations']['3d_fullres']['spacing'])
            order=prepared['plan']['transpose_forward']
            target=np.ceil(shape[::-1][order]*np.asarray(row['spacing_xyz'])[::-1][order]/spacing).astype(np.int64)
            voxels=math.prod(map(int,target))
            working=voxels*64
            case=native*32+working
            records.append(dict(case_id=row['case_id'],native_shape_xyz=shape.tolist(),working_shape=target.tolist(),
                native_read_bytes=native*32,working_bytes=working,workspace_bytes=case,
                retained_case_bytes=voxels*6,rank_prediction_bytes=voxels*320))
        else:
            # 原生序列不进入nnU-Net重采样；视频帧按需转换，传播记忆按层保留。
            working=native*32
            history=int(shape[2])*max(1,classes-1)*2**20 if stage=='prediction' else 0
            records.append(dict(case_id=row['case_id'],native_shape_xyz=shape.tolist(),
                native_read_bytes=native*32,working_bytes=working,workspace_bytes=working,
                retained_case_bytes=native*10,rank_prediction_bytes=native*8*classes+history,
                propagation_history_bytes=history))
    return records


def memory_profile(config,prepared,stage,shares,records=None):
    if records is None:
        records=case_estimates(config,prepared,stage)
    case=max(row['workspace_bytes'] for row in records)
    retained=max(row['retained_case_bytes'] for row in records)
    prediction=max(row['rank_prediction_bytes'] for row in records)
    if config['model']=='medsam2':
        fixed=[(18*GIB*batch if config['resources']['device']=='cpu' and stage!='prediction' else 2*GIB)
               for batch in shares]
        queue_per_sample=8*512*512*(3*4+3*2)
        cache_bytes=0
        rank_case=[prediction if stage=='prediction' else 0 for _ in shares]
    else:
        patch=math.prod(config['training']['patch'])
        fixed=[GIB+(patch*512*batch if config['resources']['device']=='cpu' and stage!='prediction' else 0)
               for batch in shares]
        queue_per_sample=patch*16
        cache_bytes=config['resources']['cache_bytes']
        if stage in ('local','prediction'):
            rank_case=[prediction for _ in shares]
        elif stage=='base':
            rank_case=[prediction if rank==0 else 0 for rank in range(len(shares))]
        else:
            rank_case=[prediction//5 for _ in shares]
    return dict(protocol='chd-model-memory-v1',model=config['model'],stage=stage,
        coordinator_bytes=GIB,rank_fixed_bytes=fixed,worker_fixed_bytes=GIB,rank_case_bytes=rank_case,
        retained_case_bytes=retained,case_bytes=case,queue_per_sample_bytes=queue_per_sample,
        worker_layout=config.get('model_topology',{}).get('workers') if config['model']=='medsam2' else None,
        cache_limit_bytes=cache_bytes,records=records,
        interpretation='按实际读取路径估算的准入预算；正式规格容量仍需实测')
