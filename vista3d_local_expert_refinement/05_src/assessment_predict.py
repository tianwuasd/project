"""One calibration-case prediction/metrics from an isolated short-run checkpoint."""
import argparse
import os
from pathlib import Path
import time


def predict(work,fold,method,training,case_id):
    os.environ['WHOLE_HEART_WORKSPACE']=str(work)
    from prepare_nnunet import configure_paths
    configure_paths()
    import nibabel as nib
    import numpy as np
    import torch
    from server_data import read_json,write_json,digest
    from model_catalog import trainer_class
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from evaluation import save_prediction,score_case,PROTOCOL
    from data_pipeline import map_labels
    from task_resources import torch_threads
    torch_threads(torch)
    work=Path(work);training=Path(training)
    assessment=read_json(work/'00_admin/assessment.json')
    if assessment['job_kind']!='assessment' or case_id not in assessment['scope']['prediction_ids']:
        raise ValueError('测评只允许预先选定的来源cal病例')
    cases=read_json(work/'04_data/manifests/case_inventory.json')['cases']
    case=next(c for c in cases if c['case_id']==case_id)
    checkpoint=training/'fold_0/checkpoint_latest.pth'
    trainer=trainer_class(method)(read_json(training/'plans.json'),'3d_fullres',0,read_json(training/'dataset.json'))
    trainer._vista_inference_only=True
    trainer.load_checkpoint(str(checkpoint));trainer.network.eval();trainer.set_deep_supervision_enabled(False)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=PROTOCOL['mirroring'],
        perform_everything_on_device=False,device=torch.device('cuda'),verbose=False,allow_tqdm=False)
    predictor.manual_initialization(trainer.network,trainer.plans_manager,trainer.configuration_manager,
        [trainer.network.state_dict()],trainer.dataset_json,trainer.__class__.__name__,trainer.inference_allowed_mirroring_axes)
    reader=trainer.plans_manager.image_reader_writer_class()
    started=time.monotonic()
    images,properties=reader.read_images([case['image']])
    segmentation=predictor.predict_single_npy_array(images,properties,None,None,False)
    original=nib.load(case['image']);reference_file=nib.load(case['label'])
    if reference_file.shape!=original.shape or not np.allclose(reference_file.affine,original.affine,atol=1e-4,rtol=0):
        raise ValueError('cal标签与影像空间不一致')
    path=work/f'08_results/assessment/{fold}_{method}/{case_id}.nii.gz'
    save_prediction(path,segmentation,original)
    result=dict(status='passed',case_id=case_id,prediction=str(path),prediction_sha256=digest(path),
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),seconds=time.monotonic()-started,
        original_grid_restored=True,protocol=PROTOCOL,
        metrics=score_case(segmentation,map_labels(np.asanyarray(reference_file.dataobj)),original.affine),
        formal_effectiveness_result=False,scope='calibration-case engineering check after bounded short trials; not trained-model efficacy')
    write_json(work/f'09_reports/assessment_prediction_{fold}_{method}.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('fold');parser.add_argument('method')
    parser.add_argument('training');parser.add_argument('case');args=parser.parse_args()
    predict(os.environ['WHOLE_HEART_WORKSPACE'],args.fold,args.method,args.training,args.case)
