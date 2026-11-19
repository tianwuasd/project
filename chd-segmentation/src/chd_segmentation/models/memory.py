import math
from pathlib import Path

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
    source = None
    if prepared.get('storage'):
        from ..preprocessing.inputs import resolve_preprocessed
        source = resolve_preprocessed(prepared['storage']['root'],dataset=prepared['dataset'],
            case_ids=[row['case_id'] for row in rows],storage=prepared['storage'])
    for row in rows:
        stored = prepared.get('storage',{}).get('cases',{}).get(row['case_id'])
        entry = source.open(row['case_id'],with_label=False) if stored is not None else None
        original_shape = np.asarray(entry['source_shape_xyz'] if entry is not None else nib.load(row['image']).shape,dtype=np.int64)
        original_voxels = math.prod(map(int, original_shape))
        actual_roi = entry['roi_xyz'] if entry is not None else roi
        input_shape = np.asarray([end-begin for begin,end in actual_roi] if actual_roi else original_shape,dtype=np.int64)
        shape=np.asarray(stored['shape_czyx'][1:][::-1] if stored is not None else
            input_shape,dtype=np.int64)
        if shape.shape!=(3,) or (shape<=0).any():
            raise ValueError('模型内存预算需要有效三维网格')
        native=math.prod(map(int,input_shape))
        if config['model']=='q_route1':
            spacing=np.asarray(prepared['plan']['configurations']['3d_fullres']['spacing'])
            order=prepared['plan']['transpose_forward']
            target=np.asarray(stored['shape_czyx'][1:]) if stored is not None else np.ceil(
                shape[::-1][order]*np.asarray(row['spacing_xyz'])[::-1][order]/spacing).astype(np.int64)
            voxels=math.prod(map(int,target))
            working=0 if stored is not None else voxels*64
            case=max(1,stored['locations_bytes']*2) if stored is not None else native*32+working
            block = config['resources'].get('validation_block_bytes',64*1024**2)
            bounded_validation = 8*block + math.prod(config['training']['patch'])*11*32 + 256*2**20
            validation_names = prepared.get('q_validation_names')
            is_validation = row['case_id'] in validation_names if validation_names is not None else 'validation' in row['roles']
            retained = max(voxels*6,stored['array_bytes']+stored['locations_bytes']) if stored is not None else voxels*6
            if stage=='base' and stored is not None:
                retained = stored['locations_bytes'] + math.prod(config['training']['patch'])*16
            records.append(dict(case_id=row['case_id'],native_shape_xyz=input_shape.tolist(),working_shape=target.tolist(),
                native_read_bytes=0 if stored is not None else native*32,working_bytes=working,workspace_bytes=case,
                retained_case_bytes=retained, validation_workspace_bytes=bounded_validation if is_validation else 0,
                is_validation=is_validation,
                mapped_file_bytes=stored['array_bytes'] if stored is not None else 0,
                rank_prediction_bytes=voxels*(11*32+288)+native*classes*8+original_voxels*2))
        else:
            from ..data.input_binding import intensity_policy
            mode = intensity_policy(entry,config.get('input_source',{}))['mode'] if entry is not None else 'source_window'
            processed=math.prod(map(int,shape))
            depth=int(stored['shape_czyx'][1]) if stored is not None else int(shape[2])
            # 转码、完整float32视频与标签按实际处理网格；原图与ROI仅用于空间恢复。
            conversion=processed*({'windowed_uint8':8,'source_window':24,'normalized_percentile':32}[mode])
            working=conversion+(stored['locations_bytes'] if stored is not None else native*32)
            history=depth*max(1,classes-1)*4*2**20 if stage=='prediction' else 0
            restore=native*(classes*8+10)+original_voxels*2
            records.append(dict(case_id=row['case_id'],native_shape_xyz=input_shape.tolist(),
                processed_shape_xyz=shape.tolist(),processed_voxels=processed,intensity_mode=mode,
                conversion_workspace_bytes=conversion,restore_workspace_bytes=restore,
                native_read_bytes=0 if stored is not None else native*32,working_bytes=working,workspace_bytes=working,
                mapped_file_bytes=stored['array_bytes'] if stored is not None else 0,
                retained_case_bytes=processed*10,rank_prediction_bytes=processed*8*classes+history+restore,
                propagation_history_bytes=history))
        if stage == 'preprocess':
            block = config['preprocessing']['block_bytes']
            output = voxels*5 if config['model'] == 'q_route1' else native*3
            records[-1].update(storage='bounded_disk_blocks', input_shape_xyz=shape.tolist(),
                raw_workspace_bytes=2*GIB, resample_workspace_bytes=4*block,
                workspace_bytes=2*GIB+4*block, output_array_bytes=output,
                legacy_whole_case_bytes=case if config['model'] == 'q_route1' else native*32,
                retained_case_bytes=0, rank_prediction_bytes=0)
    return records


def memory_profile(config,prepared,stage,shares,records=None):
    if records is None:
        records=case_estimates(config,prepared,stage)
    case=max(row['workspace_bytes'] for row in records)
    retained=max(row['retained_case_bytes'] for row in records)
    prediction=max(row['rank_prediction_bytes'] for row in records)
    if config['model']=='medsam2':
        # CPU保留骨干及官方最多3目标的多对象计算图；成功峰值仍需独立实测。
        sequence_work = math.prod(config['training']['patch'])*4*256*3*(1+3)
        fixed=[(GIB+sequence_work*batch if config['resources']['device']=='cpu' and stage!='prediction' else 2*GIB)
               for batch in shares]
        queue_per_sample=8*512*512*(3*4+3*2)
        cache_bytes=0
        rank_case=[prediction if stage=='prediction' else 0 for _ in shares]
    else:
        patch=math.prod(config['training']['patch'])
        # 无初始化权重的定位器和局部阶段按128MiB状态规划，包含其完整保存及历史余量。
        checkpoint_weight = (Path(config['pretrained']).stat().st_size if config['pretrained'] is not None else 128*2**20) if stage != 'prediction' else 0
        fixed=[GIB+4*checkpoint_weight+(patch*512*batch if config['resources']['device']=='cpu' and stage!='prediction' else 0)
               for batch in shares]
        queue_per_sample=patch*16
        cache_bytes=config['resources']['cache_bytes']
        if stage in ('local','prediction'):
            rank_case=[prediction for _ in shares]
        elif stage=='base':
            validation = max(row['validation_workspace_bytes'] for row in records)
            rank_case=[validation if rank==0 else 0 for rank in range(len(shares))]
        else:
            rank_case=[prediction//5 for _ in shares]
    checkpoint = 0 if stage == 'prediction' else (6*Path(config['pretrained']).stat().st_size if config['model'] == 'medsam2' else 4*checkpoint_weight)+16*2**20
    return dict(protocol='chd-model-memory-v2',model=config['model'],stage=stage,
        coordinator_bytes=GIB+checkpoint,checkpoint_reserve_bytes=checkpoint,
        rank_fixed_bytes=fixed,worker_fixed_bytes=GIB,rank_case_bytes=rank_case,
        retained_case_bytes=retained,case_bytes=case,queue_per_sample_bytes=queue_per_sample,
        worker_layout=config.get('model_topology',{}).get('workers') if config['model']=='medsam2' else None,
        cache_limit_bytes=cache_bytes,records=records,
        interpretation='按实际读取路径估算的准入预算；正式规格容量仍需实测')
