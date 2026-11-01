"""One calibration-case prediction/metrics from an isolated short-run checkpoint."""
import argparse
import os
from pathlib import Path
import time


def restore_reader_grid(reader,segmentation,properties,original,path,expected_affine=None):
    """Use the selected nnU-Net writer to undo its reader-specific axis order."""
    import nibabel as nib
    import numpy as np
    temporary=Path(path).with_name(Path(path).name+'.axes.tmp.nii.gz')
    temporary.parent.mkdir(parents=True,exist_ok=True)
    try:
        reader.write_seg(segmentation,str(temporary),properties)
        restored=nib.load(temporary)
        expected=original.affine if expected_affine is None else expected_affine
        if restored.shape!=original.shape or not np.allclose(restored.affine,expected,atol=1e-4,rtol=0):
            raise ValueError('测评推理未正确恢复原图空间')
        result=np.asarray(restored.dataobj).copy()
        del restored
        return result
    finally:
        temporary.unlink(missing_ok=True)


def predict(work,fold,method,training,case_id,phase='gpu'):
    os.environ['WHOLE_HEART_WORKSPACE']=str(work)
    from prepare_nnunet import configure_paths
    configure_paths()
    import pickle
    import nibabel as nib
    import numpy as np
    import torch
    from server_data import read_json,write_json,digest
    from evaluation import save_prediction,score_case,PROTOCOL
    from data_pipeline import map_labels
    from task_resources import apply_training_cpu
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    work=Path(work);training=Path(training)
    assessment=read_json(work/'00_admin/assessment.json')
    if assessment['job_kind']!='assessment' or case_id not in assessment['scope']['prediction_ids']:
        raise ValueError('测评只允许预先选定的来源cal病例')
    cases=read_json(work/'04_data/manifests/case_inventory.json')['cases']
    case=next(c for c in cases if c['case_id']==case_id)
    checkpoint=training/'fold_0/checkpoint_latest.pth';started=time.monotonic()
    count=max(1,int(os.environ.get('WHOLE_HEART_CPU_THREADS','1')))
    limiter=apply_training_cpu(dict(threads=count),torch)
    base=work/f'08_results/assessment/{fold}_{method}';base.mkdir(parents=True,exist_ok=True)
    array=base/f'{case_id}.logits.npy';props=base/f'{case_id}.properties.pkl';marker=base/f'{case_id}.gpu.json'
    try:
        if phase=='gpu':
            from model_catalog import trainer_class
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
            trainer=trainer_class(method)(read_json(training/'plans.json'),'3d_fullres',0,read_json(training/'dataset.json'))
            trainer._vista_inference_only=True
            trainer.load_checkpoint(str(checkpoint));trainer.network.eval();trainer.set_deep_supervision_enabled(False)
            predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=PROTOCOL['mirroring'],
                perform_everything_on_device=False,device=torch.device('cuda'),verbose=False,allow_tqdm=False)
            predictor.manual_initialization(trainer.network,trainer.plans_manager,trainer.configuration_manager,
                [trainer.network.state_dict()],trainer.dataset_json,trainer.__class__.__name__,trainer.inference_allowed_mirroring_axes)
            reader=trainer.plans_manager.image_reader_writer_class()
            from imagechd_geometry import derived_image
            images,properties=reader.read_images([str(derived_image(work,case))])
            adapter=PreprocessAdapterFromNpy([images],[None],[properties],[None],trainer.plans_manager,trainer.dataset_json,
                trainer.configuration_manager,num_threads_in_multithreaded=1,verbose=False)
            data=next(adapter);logits=predictor.predict_logits_from_preprocessed_data(data['data']).cpu().numpy()
            np.save(array,logits,allow_pickle=False)
            with props.open('wb') as stream:pickle.dump(data['data_properties'],stream,pickle.HIGHEST_PROTOCOL)
            write_json(marker,dict(case_id=case_id,checkpoint_sha256=digest(checkpoint),logits_sha256=digest(array),
                properties_sha256=digest(props),seconds=time.monotonic()-started))
            return
        if phase!='cpu':raise ValueError('unknown assessment prediction phase')
        if torch.cuda.is_initialized():raise ValueError('CPU收尾进程意外建立CUDA上下文')
        saved=read_json(marker)
        if saved['case_id']!=case_id or saved['checkpoint_sha256']!=digest(checkpoint) or saved['logits_sha256']!=digest(array) or saved['properties_sha256']!=digest(props):
            raise ValueError('测评GPU中间结果不完整或已变化')
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        manager=PlansManager(read_json(training/'plans.json'));config=manager.get_configuration('3d_fullres')
        dataset=read_json(training/'dataset.json');reader=manager.image_reader_writer_class()
        with props.open('rb') as stream:properties=pickle.load(stream)
        segmentation=convert_predicted_logits_to_segmentation_with_correct_shape(np.load(array,allow_pickle=False,mmap_mode='r'),
            manager,config,manager.get_label_manager(dataset),properties,return_probabilities=False,num_threads_torch=count)
        original=nib.load(case['image']);reference_file=nib.load(case['label'])
        if reference_file.shape!=original.shape or not np.allclose(reference_file.affine,original.affine,atol=1e-4,rtol=0):
            raise ValueError('cal标签与影像空间不一致')
        path=base/f'{case_id}.nii.gz'
        from imagechd_geometry import physical_affine
        physical=physical_affine(original,case['spacing_mm'])
        segmentation=restore_reader_grid(reader,segmentation,properties,original,path,expected_affine=physical)
        save_prediction(path,segmentation,original)
        result=dict(status='passed',case_id=case_id,prediction=str(path),prediction_sha256=digest(path),
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),seconds=time.monotonic()-started,
            gpu_seconds=saved['seconds'],original_grid_restored=True,protocol=PROTOCOL,cpu_threads=count,
            metrics=score_case(segmentation,map_labels(np.asanyarray(reference_file.dataobj)),physical,threads=count),
            formal_effectiveness_result=False,scope='calibration-case engineering check after bounded short trials; not trained-model efficacy')
        write_json(work/f'09_reports/assessment_prediction_{fold}_{method}.json',result)
        return result
    finally:limiter.restore_original_limits()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('fold');parser.add_argument('method')
    parser.add_argument('training');parser.add_argument('case');parser.add_argument('--phase',choices=['gpu','cpu'],required=True)
    args=parser.parse_args()
    predict(os.environ['WHOLE_HEART_WORKSPACE'],args.fold,args.method,args.training,args.case,args.phase)
