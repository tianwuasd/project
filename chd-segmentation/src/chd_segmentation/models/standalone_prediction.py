from dataclasses import replace
import os
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import psutil
import torch

from ..acceleration import ResourceSession, ResourceClient, RuntimeRequest, ResourceDemand, SafeBoundary
from ..artifacts.store import digest, read_json, write_json, require_output, operation_lock
from ..observability.events import event
from ..prediction.commit import CaseCommit
from ..prediction.metrics import case_data_identity
from ..preprocessing.prepare import load_prepared


def working_image(source, row):
    spacing = np.asarray(row['spacing_xyz'],dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('病例缺少真实毫米间距')
    affine = source.affine.copy()
    if row['geometry_policy']=='shared_index_excel':
        affine = np.diag([*spacing,1.])
    elif row['geometry_policy']=='matching_affine_author_spacing':
        norm = np.linalg.norm(affine[:3,:3],axis=0)
        if (norm<=0).any():
            raise ValueError('原始影像方向矩阵无效')
        affine[:3,:3] *= spacing/norm
    else:
        raise ValueError('未知病例几何策略：'+row['geometry_policy'])
    header = source.header.copy()
    header.set_xyzt_units('mm')
    return nib.Nifti1Image(source.dataobj,affine,header)


def case_workspace(image, row, model):
    if len(image.shape)!=3:
        raise ValueError('仅支持三维CT')
    voxels = int(np.prod(image.shape,dtype=np.int64))
    if model=='totalsegmentator':
        lowres = int(np.prod(np.maximum(1,np.rint(np.asarray(image.shape)*np.asarray(row['spacing_xyz'])/3))))
        return max(voxels,lowres)*96 + 1024**3
    return voxels*8 + max(image.shape)**2*64 + 2*1024**3


def native_uint8(binding, row, image, config):
    if binding is None or config['model']!='medsam3':
        return None
    recipe = binding.catalog['recipe']
    if recipe.get('window_conversion')!='fixed_linear_uint8' or recipe.get('gray_window')!=config['backend']['gray_window']:
        return None
    entry = binding.open(row['case_id'],with_label=False)
    full_roi = [[0,int(v)] for v in image.shape]
    if entry.get('roi_xyz')!=full_roi or entry.get('source_shape_xyz')!=list(image.shape) or not np.allclose(entry['source_affine'],image.affine,rtol=0,atol=1e-5) or entry.get('axis_order')!=[0,1,2] or not np.allclose(entry['spacing_xyz'],row['spacing_xyz']):
        return None
    values = entry['image']
    if values.dtype!=np.uint8 or values.shape!=(1,*image.shape[::-1]):
        raise ValueError('声明兼容的原生uint8病例实际dtype或shape无效')
    return values[0].transpose(2,1,0)


def input_reference(binding,row):
    if binding is None:
        return None
    pointer = binding.cases[row['case_id']]
    return dict(root=str(binding.root),case_key=pointer['key'],recipe_identity=binding.catalog['recipe_identity'],
        shape_czyx=pointer['shape_czyx'])


def predict(config, prepared_path, output, case_ids, test_set, test_group):
    from .prediction import prediction_rows
    prepared = load_prepared(prepared_path)
    rows = prediction_rows(config,prepared,case_ids,test_set,test_group)
    labels = prepared['dataset_json']['labels']
    from ..datasets.manifest import LABELS
    if prepared['dataset'] not in LABELS or set(labels) != set(LABELS[prepared['dataset']]) or sorted(labels.values())!=list(range(len(labels))) or labels['background']!=0:
        raise ValueError('数据集类别名称或编号无效')
    source_path = config['input_source'].get('path')
    input_binding = None
    if source_path:
        from ..preprocessing.inputs import resolve_preprocessed
        input_binding = resolve_preprocessed(source_path,dataset=prepared['dataset'],case_ids=[row['case_id'] for row in rows])
        if input_binding.catalog['recipe']['labels']!=prepared['dataset_json']:
            raise ValueError('显式预处理来源的类别语义不同')
    protected = [prepared['manifest']['data_root'],Path(__file__).parents[3],
        *[v for k,v in config['backend'].items() if k in ('official_root','base_weights','lora_weights','weights_dir','tokenizer','lora_config')]]
    if source_path:
        protected.append(source_path)
    run = require_output(config['run_dir'],config['work_root'],protected)
    root = Path(output or config['output']).resolve()
    root = require_output(root,root.parent,[*protected,run])
    allowed = prediction_rows(config,prepared,None,test_set,test_group)
    task = dict(model=config['model'],backend=config['backend'],dataset=prepared['dataset'],
        labels=labels,cases=[row['case_id'] for row in allowed],input_source=config['input_source'],test_set=test_set,test_group=test_group,
        split=[{key:row.get(key) for key in ('case_id','roles','patient','test_group','unknown_classes','ignore_label_ids',
            'image','spacing_xyz','geometry_policy')} for row in allowed],
        geometry='manifest working millimeters; export source shape and affine')
    with operation_lock(root), operation_lock(run):
        return run_prediction(config,prepared,rows,labels,root,task,input_binding)


def run_prediction(config,prepared,rows,labels,root,task,input_binding):
    began = perf_counter()
    task_path, progress = root/'prediction-task.json', root/'predictions.json'
    if task_path.exists() and read_json(task_path)!=task:
        raise ValueError('固定预测设置或病例名单变化，请使用独立结果目录')
    if not task_path.exists():
        write_json(task_path,task)
    completed = read_json(progress)['cases'] if progress.exists() else {}
    def save(records):
        write_json(progress,dict(cases=records))
    commits = CaseCommit(root,digest(task))
    commits.recover(completed,save)
    pending = []
    for row in rows:
        if row['case_id'] in completed:
            receipt = completed[row['case_id']]
            if receipt['data_identity']!=case_data_identity(prepared,row) or not commits.path(receipt['file']).is_file():
                raise ValueError('已完成病例的数据语义或输出文件缺失')
            if receipt['input_adaptation']['preprocessed_reference']!=input_reference(input_binding,row):
                raise ValueError('已完成病例的显式输入指针或配方变化')
        else:
            pending.append(row)
    if not pending:
        return dict(model=config['model'],cases=len(completed),reused=True,seconds=perf_counter()-began)
    precision = 'fp16' if config['model']=='totalsegmentator' and config['device']=='cuda' else 'fp32'
    resources = dict(config['execution'],precision=precision,gpu_spec=dict(resource_model=config['model'],stage='prediction'))
    if resources['wait_seconds'] is not None:
        resources['max_wait'] = resources['wait_seconds']
    demand = ResourceDemand(fixed_bytes=config['backend']['model_memory_bytes'],cpu_slots=config['threads'],
        gpu_count=1 if config['device']=='cuda' else 0,gpu_ids=resources['gpu_ids'],local_batches=[1])
    request = RuntimeRequest(config['model']+'-prediction','prediction',Path(config['work_root'])/'resources',
        config['run_dir'],resources,demand)
    with ResourceSession(request,layout=dict(workers=[0],prefetch=1)) as session:
        client = ResourceClient(session.handle)
        last_progress = -float('inf')
        def boundary(phase,**details):
            nonlocal last_progress
            mark = perf_counter()
            decision = client.boundary()
            if decision.action=='wait':
                event(config['run_dir'],'prediction_resource_wait',phase=phase,reason=decision.reason,**details)
            while decision.action=='wait':
                if perf_counter()-last_progress >= config['logging']['heartbeat_seconds']:
                    event(config['run_dir'],'prediction_resource_wait',phase=phase,reason=decision.reason,
                        waited_seconds=perf_counter()-mark,elapsed_seconds=perf_counter()-began,**details)
                    last_progress = perf_counter()
                with client.handle.condition:
                    client.handle.condition.wait(timeout=min(1.,resources['poll_seconds']))
                decision = client.boundary()
            if decision.action=='stop':
                raise InterruptedError('预测安全边界停止：'+decision.reason)
            console = perf_counter()-last_progress >= config['logging']['heartbeat_seconds']
            event(config['run_dir'],'prediction_boundary',console=console,phase=phase,
                waited_seconds=perf_counter()-mark,elapsed_seconds=perf_counter()-began,
                rss_bytes=psutil.Process().memory_info().rss,**details)
            if console:
                last_progress = perf_counter()
        device = 'cpu'
        if session.allocation.gpus:
            gpu = session.allocation.gpus[0]
            if torch.cuda.is_initialized():
                raise RuntimeError('CUDA已初始化，请用对应独立解释器启动新预测进程')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
            device = 'cuda:0'
            if not torch.cuda.is_available():
                raise RuntimeError('已分配CUDA设备不可用')
            from .execution import verify_device
            verify_device(torch.device(device),dict(gpus=[gpu]),0)
        torch.set_num_threads(session.allocation.cpu_threads)
        backend = None
        for row in pending:
            mark = perf_counter()
            name = row['case_id']
            if not name.replace('_','').replace('-','').isalnum():
                raise ValueError('病例编号包含不允许的路径字符')
            source = nib.load(row['image'])
            needed = case_workspace(source,row,config['model'])
            session.replan(replace(demand,workspace_bytes=needed,reader_slots=1),safe_boundary=SafeBoundary('before_case'))
            boundary('before_case',case_id=name)
            event(config['run_dir'],'prediction_case_started',case_id=name,completed=len(completed),total=len(rows),
                shape=list(source.shape),workspace_bytes=needed,device=device)
            with client.case(needed,purpose='prediction '+name):
                image = working_image(source,row)
                if backend is None:
                    boundary('model_initialization_started')
                    if config['model']=='totalsegmentator':
                        from .totalsegmentator.adapter import TotalSegmentatorAdapter
                        backend = TotalSegmentatorAdapter(config,device,boundary)
                    else:
                        from .medsam3.adapter import MedSAM3Adapter
                        backend = MedSAM3Adapter(config,device,boundary)
                    event(config['run_dir'],'prediction_model_ready',model=config['model'],device=device)
                native = native_uint8(input_binding,row,source,config)
                if config['model']=='medsam3':
                    native_image = nib.Nifti1Image(native,image.affine) if native is not None else None
                    result, metadata = backend.predict(image,labels,native_uint8=native_image)
                    del native_image, native
                else:
                    result, metadata = backend.predict(image,labels)
                if result.shape!=source.shape:
                    raise ValueError('模型预测未恢复原始空间尺寸')
                final, temporary = name+'.nii.gz',name+'.pending.nii.gz'
                receipt = dict(file=final,scope=prepared['scope'],roi_xyz=[[0,int(v)] for v in source.shape],
                    checkpoint=digest(task['backend']),data_identity=case_data_identity(prepared,row),
                    reference_sha256=row.get('sha256',{}).get('label'),model=config['model'],
                    geometry_policy=row['geometry_policy'],spacing_xyz=row['spacing_xyz'],
                    input_adaptation=dict(selected_preprocessed=str(input_binding.root) if input_binding else None,
                        preprocessed_reference=input_reference(input_binding,row),
                        consumed='native fixed-window uint8' if config['model']=='medsam3' and metadata['intensity']['source']=='native uint8' else 'manifest original HU',
                        working_affine=image.affine.tolist(),source_affine=source.affine.tolist(),
                        reason='reuse compatible native source' if config['model']=='medsam3' and metadata['intensity']['source']=='native uint8' else
                            'original HU preserves physical CT; native recipe/window/ROI/geometry not equivalent or not selected'),**metadata)
                commits.begin(name,receipt,final,temporary)
                header = source.header.copy()
                header.set_data_dtype(np.uint8)
                nib.save(nib.Nifti1Image(result.astype(np.uint8,copy=False),source.affine,header),root/temporary)
                del result, image
                receipt['seconds'],receipt['file_bytes'] = perf_counter()-mark,(root/temporary).stat().st_size
                commits.ready(receipt)
                commits.recover(completed,save)
            boundary('prediction_case_completed',case_id=name)
            event(config['run_dir'],'prediction_case_completed',case_id=name,completed=len(completed),total=len(rows),
                seconds=perf_counter()-mark,rss_bytes=psutil.Process().memory_info().rss,
                gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device!='cpu' else None)
    return dict(model=config['model'],cases=len(completed),reused=False,seconds=perf_counter()-began)
