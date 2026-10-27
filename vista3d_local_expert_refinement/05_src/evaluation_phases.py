"""Same nnU-Net 2.8.1 inference path, split before original-shape restoration.

predict_single_npy_array itself uses PreprocessAdapterFromNpy, predict_logits,
then convert_predicted_logits_to_segmentation_with_correct_shape. We persist
that exact boundary; CPU tail imports no network and never creates CUDA tensors.
"""
import os
import pickle
import time
from pathlib import Path
import numpy as np
import nibabel as nib
from server_data import read_json,write_json,digest
from evaluation_contract import folder,signature,cached,committed


def preprocess_case(work,model,case_id,threads=1):
    """CPU-only process entry; loads plans/reader, never a network or CUDA."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    import torch
    from task_resources import apply_training_cpu
    limiter=apply_training_cpu(dict(threads=threads),torch)
    try:
        split=read_json(Path(work)/'04_data/manifests/splits_v1.json')['folds'][model['fold']]
        prep=Path(work)/f'04_data/derived/nnUNet_preprocessed/Dataset{split["dataset_id"]:03d}_{model["fold"]}'
        manager=PlansManager(read_json(prep/'WholeheartBaselinePlans.json'))
        config=manager.get_configuration('3d_fullres');dataset=read_json(prep/'dataset.json')
        from read_admission import AdmittedReader
        reader=AdmittedReader(manager.image_reader_writer_class())
        case=next(c for c in read_json(Path(work)/'04_data/manifests/case_inventory.json')['cases'] if c['case_id']==case_id)
        started=time.perf_counter();images,properties=reader.read_images([case['image']])
        adapter=PreprocessAdapterFromNpy([images],[None],[properties],[None],manager,dataset,config,
            num_threads_in_multithreaded=1,verbose=False)
        data=next(adapter)
        # IPC transports disk paths, never an unbounded full-volume pickle/pipe.
        base=folder(work,model)/'staging';base.mkdir(parents=True,exist_ok=True)
        array=base/f'{case_id}.npy';metadata=base/f'{case_id}.pkl'
        np.save(array,data['data'].numpy(),allow_pickle=False)
        with metadata.open('wb') as stream:pickle.dump(data['data_properties'],stream,pickle.HIGHEST_PROTOCOL)
        from task_resources import process_observation
        return dict(array=str(array),properties=str(metadata),prepare_seconds=time.perf_counter()-started,
            cpu_observation=dict(process_observation(),threads=threads))
    finally:limiter.restore_original_limits()


def infer(work,model,case_id,sha,predictor=None,prepared=None):
    import torch
    from evaluation import make_predictor
    if cached(work,model,case_id,sha,'gpu'):
        print('已校验GPU病例结果，直接复用：'+case_id,flush=True);return
    expected=signature(work,model,case_id,sha)
    own=predictor is None
    if own:predictor,_=make_predictor(model)
    prepared=prepared or preprocess_case(work,model,case_id)
    started=time.perf_counter()
    try:
        array=np.load(prepared['array'],allow_pickle=False)
        logits=predictor.predict_logits_from_preprocessed_data(torch.from_numpy(array)).cpu().numpy()
        with Path(prepared['properties']).open('rb') as stream:properties=pickle.load(stream)
        base=folder(work,model)/'intermediate';base.mkdir(parents=True,exist_ok=True)
        outputs={}
        for name,ext,value in [('logits','npy',logits),('properties','pkl',properties)]:
            path=base/f'{case_id}.{ext}';temporary=path.with_suffix(path.suffix+'.tmp')
            with temporary.open('wb') as stream:
                if name=='logits':np.save(stream,value,allow_pickle=False)
                else:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary,path);outputs[name]=dict(path=str(path),sha256=digest(path))
        from task_resources import process_observation
        write_json(base/f'{case_id}.json',dict(status='completed',signature=expected,outputs=outputs,
            seconds=time.perf_counter()-started,prepare_seconds=prepared['prepare_seconds'],cpu_preparation=prepared.get('cpu_observation'),
            observations=process_observation()))
        print('GPU病例中间结果已提交：'+case_id,flush=True)
    finally:
        for key in ['array','properties']:Path(prepared[key]).unlink(missing_ok=True)
        if own:
            del predictor;torch.cuda.empty_cache()


def finish(work,model,case_id,sha):
    import torch
    from evaluation import save_prediction,largest_components,score_case
    from data_pipeline import map_labels,inverse_labels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    if torch.cuda.is_initialized():raise ValueError('CPU收尾进程意外建立CUDA上下文')
    if cached(work,model,case_id,sha,'cpu'):return
    expected=signature(work,model,case_id,sha);base=folder(work,model)
    path=base/'intermediate'/f'{case_id}.json'
    if not committed(path,expected,['logits','properties']):raise ValueError('GPU病例未完成：'+case_id)
    saved=read_json(path)
    logits=np.load(saved['outputs']['logits']['path'],allow_pickle=False,mmap_mode='r')
    with Path(saved['outputs']['properties']['path']).open('rb') as stream:properties=pickle.load(stream)
    split=read_json(Path(work)/'04_data/manifests/splits_v1.json')['folds'][model['fold']]
    prep=Path(work)/f'04_data/derived/nnUNet_preprocessed/Dataset{split["dataset_id"]:03d}_{model["fold"]}'
    manager=PlansManager(read_json(prep/'WholeheartBaselinePlans.json'));configuration=manager.get_configuration('3d_fullres')
    dataset=read_json(prep/'dataset.json');reader=manager.image_reader_writer_class()
    started=time.perf_counter()
    seg=convert_predicted_logits_to_segmentation_with_correct_shape(logits,manager,configuration,
        manager.get_label_manager(dataset),properties,return_probabilities=False,num_threads_torch=torch.get_num_threads())
    temporary=base/f'{case_id}.axes.tmp.nii.gz';reader.write_seg(seg,str(temporary),properties)
    case=next(c for c in read_json(Path(work)/'04_data/manifests/case_inventory.json')['cases'] if c['case_id']==case_id)
    original=nib.load(case['image']);restored=nib.load(temporary)
    if restored.shape!=original.shape or not np.allclose(restored.affine,original.affine,atol=1e-4,rtol=0):raise ValueError('推理未恢复原图空间')
    raw=np.asarray(restored.dataobj).copy();inverse_labels(raw);del restored;temporary.unlink()
    restore_seconds=time.perf_counter()-started;metric_seconds=write_seconds=0.
    reference_file=nib.load(case['label'])
    if reference_file.shape!=original.shape or not np.allclose(reference_file.affine,original.affine,atol=1e-4,rtol=0):raise ValueError('标注与影像不在相同网格')
    reference=map_labels(np.asanyarray(reference_file.dataobj));outputs={};results={}
    for variant,prediction in [('raw',raw),('lcc',largest_components(raw))]:
        destination=Path(work)/f'08_results/predictions/{model["fold"]}_{model["method"]}/{case_id}_{variant}.nii.gz'
        stamp=time.perf_counter();save_prediction(destination,prediction,original)
        outputs[variant]=dict(path=str(destination),sha256=digest(destination));write_seconds+=time.perf_counter()-stamp
        stamp=time.perf_counter();results[variant]=score_case(prediction,reference,original.affine,threads=torch.get_num_threads());metric_seconds+=time.perf_counter()-stamp
    from task_resources import process_observation
    write_json(base/f'{case_id}.json',dict(signature=expected,status='completed',fold=model['fold'],method=model['method'],
        modality=case['modality'],case_id=case_id,seconds=time.perf_counter()-started,outputs=outputs,results=results,
        phase_seconds=dict(restore=restore_seconds,write=write_seconds,metrics=metric_seconds),metric_threads=torch.get_num_threads(),observations=process_observation()))
    print(f'{model["fold"]}/{model["method"]}/{case_id}: raw Dice={results["raw"]["macro_dice"]}',flush=True)
