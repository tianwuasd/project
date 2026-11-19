import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def run(args):
    root, report = Path(args.work_root), Path(args.report)
    root.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    for name, value in dict(nnUNet_raw=root/'data/raw', nnUNet_preprocessed=root/'data/preprocessed',
                            nnUNet_results=root/'course/checkpoints', MPLCONFIGDIR=root/'plot-cache',
                            TEMP=root/'temp', TMP=root/'temp').items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(value)
    os.environ.update(nnUNet_compile='false', nnUNet_n_proc_DA='0', nnUNet_wandb_enabled='0')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project/'src'))
    import nibabel as nib
    import numpy as np
    import psutil
    import torch
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.datasets.manifest import LABELS
    from chd_segmentation.finetuning.total import DATASET, PLANS, dataset_definition, transferred_plans
    from chd_segmentation.finetuning.total_inputs import export_case
    from chd_segmentation.finetuning.total_trainer import ImageCHDTotalTrainer
    from chd_segmentation.preprocessing.validation import SourceReader
    from chd_segmentation.training.checkpoint import tensors_equal

    started = perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261120)
    original = read_json(args.manifest)
    rows = []
    for name, role in (('ct_1003', 'train'), ('ct_1012', 'validation')):
        row = deepcopy(next(row for row in original['cases'] if row['case_id']==name))
        row['roles'] = [role]
        row['source_stamp'] = {}
        for kind in ('image','label'):
            row[kind] = str(Path(args.data_root)/row[kind])
            stat = Path(row[kind]).stat()
            row['source_stamp'][kind] = [stat.st_size, stat.st_mtime_ns]
        rows.append(row)
    groups = dict(train=[rows[0]], validation=[rows[1]], test=[])
    settings = dict(epochs=2, learning_rate=.001)
    dataset = dataset_definition(settings, groups)
    prepared = dict(technical=dict(roi_xyz=None), dataset_json=dict(labels={name:i for i,name in enumerate(LABELS['imagechd'])}))
    raw = root/'data/raw'/DATASET
    exported = []
    stage_started = perf_counter()
    for row in rows:
        paths = export_case(SourceReader(prepared,512*1024**2,with_label=True),row,row['roles'][0],raw,root/'course')
        saved = nib.load(paths['image'])
        assert saved.shape==nib.load(row['image']).shape and saved.header.get_xyzt_units()[0]=='mm'
        assert np.allclose(np.linalg.norm(saved.affine[:3,:3],axis=0),row['spacing_xyz'])
        roi = tuple(slice(n//2-32,n//2+32) for n in saved.shape)
        assert np.array_equal(np.asarray(saved.dataobj[roi]),np.asarray(nib.load(row['image']).dataobj[roi],dtype=np.float32))
        stamps = {key: path.stat().st_mtime_ns for key,path in paths.items()}
        export_case(SourceReader(prepared,512*1024**2,with_label=True),row,row['roles'][0],raw,root/'course')
        assert stamps=={key:path.stat().st_mtime_ns for key,path in paths.items()}
        exported.append(dict(case_id=row['case_id'],shape=list(saved.shape),reused=True,spacing_xyz=row['spacing_xyz']))
        del saved
        gc.collect()
    export_seconds = perf_counter()-stage_started
    write_json(raw/'dataset.json',dataset)
    plan = deepcopy(read_json(args.plans))
    configuration = deepcopy(PlansManager(deepcopy(plan)).get_configuration('3d_fullres').configuration)
    configuration.update(batch_size=1,patch_size=[64]*3,data_identifier=PLANS+'_3d_fullres')
    plan.update(dataset_name=DATASET, plans_name=PLANS, configurations={'3d_fullres':configuration})
    processed = root/'data/preprocessed'/DATASET
    write_json(processed/'dataset.json',dataset)
    write_json(processed/(PLANS+'.json'),plan)
    write_json(processed/'splits_final.json',[dict(train=['ct_1003'],val=['ct_1012'])])
    write_json(processed/'unknown_classes.json',{r['case_id']:r['unknown_classes'] for r in rows})
    fingerprint = DatasetFingerprintExtractor(DATASET,num_processes=1)
    fingerprint.dataset = {'ct_1003':fingerprint.dataset['ct_1003']}
    fingerprint.run(overwrite_existing=True)
    manager = PlansManager(deepcopy(plan))
    spec = manager.get_configuration('3d_fullres')
    folder = processed/configuration['data_identifier']
    folder.mkdir(parents=True)
    stage_started = perf_counter()
    for row in rows:
        DefaultPreprocessor(verbose=False).run_case_save(str(folder/row['case_id']),
            [str(raw/'imagesTr'/(row['case_id']+'_0000.nii.gz'))],
            str(raw/'labelsTr'/(row['case_id']+'.nii.gz')), manager,spec,dataset)
    preprocess_seconds = perf_counter()-stage_started
    training_plan = dict(deepcopy(plan),continue_training=False)
    trainer = ImageCHDTotalTrainer(training_plan,'3d_fullres',0,dataset,torch.device('cpu'))
    trainer.num_iterations_per_epoch = trainer.num_val_iterations_per_epoch = 1
    trainer.on_train_start()

    def one_epoch(current):
        current.on_epoch_start()
        current.on_train_epoch_start()
        batch = next(current.dataloader_train)
        assert batch['keys']==['ct_1003']
        outputs = [current.train_step(batch)]
        current.on_train_epoch_end(outputs)
        with torch.no_grad():
            current.on_validation_epoch_start()
            validation = next(current.dataloader_val)
            assert validation['keys']==['ct_1012']
            values = [current.validation_step(validation)]
            current.on_validation_epoch_end(values)
        current.on_epoch_end()
        return dict(training_loss=float(outputs[0]['loss']),validation_loss=float(values[0]['loss']))

    stage_started = perf_counter()
    first = one_epoch(trainer)
    checkpoint = Path(trainer.output_folder)/'checkpoint_latest.pth'
    assert checkpoint.is_file()
    weights = torch.load(checkpoint,map_location='cpu',weights_only=False)
    target = nnUNetTrainer.build_network_architecture(manager,spec,1,8,True)
    heads = {key:value.clone() for key,value in target.state_dict().items() if '.seg_layers.' in key}
    load_pretrained_weights(target,str(checkpoint),verbose=False)
    assert all(torch.equal(value,target.state_dict()[key]) for key,value in weights['network_weights'].items() if '.seg_layers.' not in key)
    assert all(torch.equal(value,target.state_dict()[key]) for key,value in heads.items())
    transferred = transferred_plans(dict(plans=Path(trainer.output_folder_base)/'plans.json',
        dataset=Path(trainer.output_folder_base)/'dataset.json'),dataset,checkpoint,1)
    assert transferred['configurations']['3d_fullres']['architecture']==configuration['architecture']
    assert transferred['configurations']['3d_fullres']['spacing']==configuration['spacing']
    del target,heads,trainer
    gc.collect()
    resumed = ImageCHDTotalTrainer(dict(deepcopy(plan),continue_training=True),'3d_fullres',0,dataset,torch.device('cpu'))
    resumed.num_iterations_per_epoch = resumed.num_val_iterations_per_epoch = 1
    resumed.load_checkpoint(str(checkpoint))
    assert resumed.current_epoch==1 and tensors_equal(resumed.network.state_dict(),weights['network_weights'])
    assert tensors_equal(resumed.optimizer.state_dict(),weights['optimizer_state'])
    del weights
    gc.collect()
    resumed.on_train_start()
    second = one_epoch(resumed)
    resumed.on_train_end()
    assert (Path(resumed.output_folder)/'checkpoint_final.pth').is_file()
    training_seconds = perf_counter()-stage_started
    result = dict(state='passed',cases=exported,training_updates=2,training_processes=1,training_patch=[64]*3,
        initialized_network_plan=str(args.plans),official_301_weights_used=False,official_license_checked=False,
        generic_nnunet_weight_transfer_verified=True,target_heads_preserved=True,restored_model_optimizer=True,
        frozen_train=['ct_1003'],frozen_validation=['ct_1012'],first=first,second=second,
        seconds=perf_counter()-started,export_seconds=export_seconds,preprocess_seconds=preprocess_seconds,
        training_seconds=training_seconds,peak_rss_bytes=psutil.Process().memory_info().peak_wset,
        gpu_used=False,scope='real nnUNet initialized-network mechanism; 301 source and Linux/CUDA unverified')
    write_json(report/'acceptance.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def inspect_components(args):
    import ast
    import unittest
    import yaml

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / 'src'))
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.finetuning.total import memory_budget, options, status
    from chd_segmentation.training.host import host_capacity

    root, report = Path(args.work_root).resolve(strict=True), Path(args.report)
    original = read_json(args.manifest)
    rows = [dict(row, image=str(Path(args.data_root) / row['image']))
            for row in original['cases'] if row['case_id'] in ('ct_1003', 'ct_1012')]
    assert len(rows) == 2
    prepared = root / 'metadata-only.json'
    write_json(prepared, dict(manifest=dict(cases=rows)))
    config = yaml.safe_load((project / 'configs/finetuning/totalsegmentator.yaml').read_text(encoding='utf-8'))
    config.update(prepared=str(prepared), model_folder=str(Path(args.plans).parent),
                  work_root=str(root / 'course'), run_dir=str(root / 'course/run'),
                  data_root=str(root / 'data'), dataset_dir=str(root / 'data/inspect'),
                  result_root=str(root / 'result'), train_ram_bytes=2**30,
                  prepare_ram_bytes=2**30, prediction_ram_bytes=2**30)
    config_path = root / 'inspect.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    config = options(config_path)
    budgets = {stage: memory_budget(config, stage) for stage in ('prepare', 'train', 'predict')}
    assert all(amount >= 6 * 2**30 for amount in budgets.values())
    guard = unittest.TestCase()
    with guard.assertRaises(MemoryError):
        memory_budget(dict(config, prediction_ram_bytes=host_capacity()['total_bytes']), 'predict')
    for path in (project / 'src/chd_segmentation/finetuning').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), str(path))
    result = dict(state='passed', cases=[row['case_id'] for row in rows], metadata_only=True,
                  training_updates=0, budgets_with_50_percent_margin=budgets,
                  impossible_host_budget_rejected=True, status=status(config))
    write_json(report / 'components.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('work-root','report','manifest','data-root','plans'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    inspect_components(args) if args.inspect_only else run(args)
