"""Synthetic end-to-end preprocessing, real pretrained GPU step, resume and inference.

Explicit diagnostic batch=1. Never launches formal training or edits source data.
The output directory and synthetic intermediate files are retained as evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'05_src'))


def run(output,weights):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('诊断输出必须是新目录')
    output.mkdir(parents=True)
    work=output/'workspace';source=output/'synthetic_input';source.mkdir()
    os.environ['WHOLE_HEART_WORKSPACE']=str(work)
    os.environ['IMAGECHD_STUNET_WEIGHTS']=str(Path(weights).resolve(strict=True))
    os.environ['WHOLE_HEART_CPU_THREADS']='2'
    os.environ['nnUNet_n_proc_DA']='0'
    os.environ['nnUNet_compile']='false'
    from imagechd_cohort import PROJECT_ID,file_hash
    import numpy as np
    import nibabel as nib
    cases=[]
    rng=np.random.default_rng(20261031)
    for i in range(6):
        cid='synthetic_'+str(i);shape=(48,48,64)
        x,y,z=np.indices(shape)
        label=((x//6)%8).astype(np.uint8)
        label[:2,:2,:2]=12  # Annotation outside the seven target structures.
        image=(x*7+y*3+z+200*label+rng.normal(0,3,shape)+i).astype(np.int16)
        affine=np.diag([1.,1.,1.,1.]);affine[:3,3]=[10,20,30]
        a=source/(cid+'_image.nii.gz');b=source/(cid+'_label.nii.gz')
        nib.save(nib.Nifti1Image(image,affine),a);nib.save(nib.Nifti1Image(label,affine),b)
        roles={str(f):('test' if i//2==f else 'val' if i%2 else 'train') for f in range(3)}
        cases.append(dict(case_id=cid,group='ImageCHD',modality='CT',patient_id=None,
            image=str(a),label=str(b),image_sha256=file_hash(a),status='eligible',
            spacing_mm=[.8,1.,1.2],spacing_source='imagechd_author_excel',roles=roles))
    (source/'imagechd_input.json').write_text(json.dumps(dict(protocol=PROJECT_ID,cases=cases)),encoding='utf-8')
    from server_data import bind_workspace,read_json,write_json
    bind_workspace(CODE,work,source,4)
    from prepare_nnunet import configure_paths
    configure_paths()
    from server_prepare import prepare
    started=time.monotonic();prepare('imagechd_fold0',4)
    import torch
    from model_catalog import trainer_class
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    prep=work/'04_data/derived/nnUNet_preprocessed/Dataset921_imagechd_fold0'
    plan=read_json(prep/'WholeheartBaselinePlans.json')
    plan['configurations']['3d_fullres']['batch_size']=1
    dataset=read_json(prep/'dataset.json')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def construct(name):
        trainer=trainer_class('S1')(plan,'3d_fullres',0,dataset,device=device)
        trainer.output_folder=str(output/name);Path(trainer.output_folder).mkdir()
        trainer.output_folder_base=trainer.output_folder;trainer.log_file=str(output/(name+'.log'))
        trainer.logger=MetaLogger(trainer.output_folder,resume=False)
        trainer.initialize();return trainer
    trainer=construct('first')
    from stunet_recipe import require_trainer_budget,recipe
    require_trainer_budget(trainer)
    group=trainer.optimizer.param_groups[0]
    assert len(trainer.optimizer.param_groups)==1
    assert group['lr']==1e-3 and group['weight_decay']==3e-5
    assert group['momentum']==.99 and group['nesterov']
    assert trainer.lr_scheduler.__class__.__name__=='PolyLRScheduler'
    train,val=trainer.get_dataloaders()
    trainer.dataloader_train=train;trainer.dataloader_val=val
    batch=next(train);validation=next(val)
    before_step=trainer.network.conv_blocks_context[0][0].conv1.weight.detach().clone()
    attempts=0
    for attempts in range(1,9):
        result=trainer.train_step(batch)
        if trainer._vista_last_optimizer_step:break
    assert trainer._vista_last_optimizer_step,'诊断8次尝试仍未得到实际参数更新'
    assert not torch.equal(before_step,trainer.network.conv_blocks_context[0][0].conv1.weight)
    dev=trainer.validation_step(validation)
    checkpoint=output/'diagnostic_checkpoint.pth';trainer.save_checkpoint(checkpoint)
    expected={k:v.detach().cpu().clone() for k,v in trainer.network.state_dict().items()}
    momentum=[v['momentum_buffer'].detach().cpu().clone() for v in trainer.optimizer.state.values() if 'momentum_buffer' in v]
    assert momentum,'没有优化器动量，不能声称恢复成功'
    axes=trainer.inference_allowed_mirroring_axes
    resumed=construct('resumed');resumed.load_checkpoint(str(checkpoint))
    assert all(torch.equal(expected[k],v.cpu()) for k,v in resumed.network.state_dict().items())
    restored=[v['momentum_buffer'].detach().cpu() for v in resumed.optimizer.state.values() if 'momentum_buffer' in v]
    assert len(momentum)==len(restored) and all(torch.equal(a,b) for a,b in zip(momentum,restored))
    for resumed_attempts in range(1,9):
        resumed.train_step(batch)
        if resumed._vista_last_optimizer_step:break
    assert resumed._vista_last_optimizer_step,'恢复后没有真实参数更新'
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    resumed.network.eval();resumed.set_deep_supervision_enabled(False)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
        perform_everything_on_device=False,device=device,verbose=False,allow_tqdm=False)
    predictor.manual_initialization(resumed.network,resumed.plans_manager,resumed.configuration_manager,
        [resumed.network.state_dict()],dataset,resumed.__class__.__name__,axes)
    model=dict(fold='imagechd_fold0',method='S1',checkpoint=str(checkpoint))
    from evaluation_phases import preprocess_case,infer
    prepared=preprocess_case(work,model,'synthetic_0',threads=2)
    infer(work,model,'synthetic_0','synthetic-diagnostic-only',predictor=predictor,prepared=prepared)
    model_file=output/'diagnostic_model.json';write_json(model_file,model)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--finish-tail',str(work),str(model_file)],env=env,check=True)
    scored=read_json(work/'08_results/evaluation/imagechd_fold0_S1/synthetic_0.json')
    prediction=nib.load(scored['outputs']['raw']['path'])
    native=nib.load(cases[0]['image'])
    assert prediction.shape==native.shape and np.allclose(prediction.affine,native.affine)
    assert set(np.unique(np.asanyarray(prediction.dataobj)))<=set(range(8))
    evidence=dict(status='passed',synthetic_only=True,device=str(device),diagnostic_batch=1,
        training_recipe=recipe('S1'),formal_budget_verified_before_diagnostic=True,
        formal_global_batch_unchanged=4,preprocessing='official nnUNet train-only fingerprint/planning',
        fit_ids=read_json(prep/'fingerprint_provenance.json')['fit_ids'],
        native_grid_prediction=True,author_spacing_input=True,extra_annotation_ignored=True,
        loaded_tensors=trainer.pretrained_provenance['loaded_tensors'],optimizer_state_exact=True,
        checkpoint_weights_exact=True,resumed_training_step=True,initial_step_attempts=attempts,
        resumed_step_attempts=resumed_attempts,momentum_tensors=len(momentum),
        train_loss=float(result['loss']),validation_loss=float(dev['loss']),
        original_grid_shapes=list(prediction.shape),seconds=time.monotonic()-started,
        accuracy_or_formal_batch_capacity_claim=False)
    write_json(output/'integration_result.json',evidence)
    print('INTEGRATION_RESULT='+json.dumps(evidence))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--weights');parser.add_argument('--finish-tail',nargs=2)
    args=parser.parse_args()
    if args.finish_tail:
        os.environ['WHOLE_HEART_WORKSPACE']=args.finish_tail[0]
        from evaluation_phases import finish
        finish(args.finish_tail[0],json.loads(Path(args.finish_tail[1]).read_text(encoding='utf-8')),'synthetic_0','synthetic-diagnostic-only')
    else:run(args.output,args.weights)
