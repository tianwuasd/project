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


def infer(work,model,case_id,sha):
    import torch
    from evaluation import make_predictor
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    if cached(work,model,case_id,sha,'gpu'):
        print('已校验GPU病例结果，直接复用：'+case_id,flush=True);return
    expected=signature(work,model,case_id,sha)
    case=next(c for c in read_json(Path(work)/'04_data/manifests/case_inventory.json')['cases'] if c['case_id']==case_id)
    predictor,reader=make_predictor(model)
    images,properties=reader.read_images([case['image']])
    adapter=PreprocessAdapterFromNpy([images],[None],[properties],[None],predictor.plans_manager,
                                   predictor.dataset_json,predictor.configuration_manager,num_threads_in_multithreaded=1,verbose=False)
    data=next(adapter)
    started=time.perf_counter()
    logits=predictor.predict_logits_from_preprocessed_data(data['data']).cpu().numpy()
    base=folder(work,model)/'intermediate';base.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name,ext,value in [('logits','npy',logits),('properties','pkl',data['data_properties'])]:
        path=base/f'{case_id}.{ext}';temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:
            if name=='logits':np.save(stream,value,allow_pickle=False)
            else:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary,path);outputs[name]=dict(path=str(path),sha256=digest(path))
    write_json(base/f'{case_id}.json',dict(status='completed',signature=expected,outputs=outputs,seconds=time.perf_counter()-started))
    print('GPU病例中间结果已提交：'+case_id,flush=True)
    del predictor,reader,logits;torch.cuda.empty_cache()


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
    reference_file=nib.load(case['label'])
    if reference_file.shape!=original.shape or not np.allclose(reference_file.affine,original.affine,atol=1e-4,rtol=0):raise ValueError('标注与影像不在相同网格')
    reference=map_labels(np.asanyarray(reference_file.dataobj));outputs={};results={}
    for variant,prediction in [('raw',raw),('lcc',largest_components(raw))]:
        destination=Path(work)/f'08_results/predictions/{model["fold"]}_{model["method"]}/{case_id}_{variant}.nii.gz'
        save_prediction(destination,prediction,original)
        outputs[variant]=dict(path=str(destination),sha256=digest(destination));results[variant]=score_case(prediction,reference,original.affine)
    write_json(base/f'{case_id}.json',dict(signature=expected,status='completed',fold=model['fold'],method=model['method'],
        modality=case['modality'],case_id=case_id,seconds=time.perf_counter()-started,outputs=outputs,results=results))
    print(f'{model["fold"]}/{model["method"]}/{case_id}: raw Dice={results["raw"]["macro_dice"]}',flush=True)
