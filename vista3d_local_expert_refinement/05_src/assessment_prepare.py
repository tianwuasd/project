"""Independent few-case preprocessing; full fit-only planning, no target labels."""
import importlib.metadata
from pathlib import Path
import time


def prepare(work,fold,gpu_memory):
    from assessment_policy import sample_scope,QUICK_BUDGET
    from data_pipeline import export_fold
    from prepare_nnunet import configure_paths
    from server_data import read_json,write_json,digest,freeze_source_labels
    configure_paths()
    work=Path(work)
    splits=read_json(work/'04_data/manifests/splits_v1.json')['folds']
    cases=read_json(work/'04_data/manifests/case_inventory.json')['cases']
    split=splits[fold];scope=sample_scope(cases,split)
    record=dict(job_kind='assessment',scope=scope,budget=dict(QUICK_BUDGET),
        formal_training=False,fold=fold,preprocessor='nnunet_default')
    marker=work/'00_admin/assessment.json'
    if marker.exists() and read_json(marker)!=record:raise ValueError('测评身份/子集已变化')
    write_json(marker,record)
    freeze_source_labels(work,fold)
    # The planner sees every original fit case, never calibration or target cases.
    # Export preserves the existing geometry/label/hash checks and full split.
    export_fold(work,fold,split,cases,resume=True)
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    name=f"Dataset{split['dataset_id']:03d}_{fold}"
    raw=work/'04_data/derived/nnUNet_raw'/name
    prep=work/'04_data/derived/nnUNet_preprocessed'/name
    started=time.monotonic()
    fingerprint=prep/'dataset_fingerprint.json'
    provenance=prep/'fingerprint_provenance.json'
    progress=work/f'00_admin/preparation_progress/{fold}.json'
    def update(phase,**fields):
        write_json(progress,dict(fold=fold,phase=phase,updated=time.time(),
            elapsed_seconds=time.monotonic()-started,assessment=True,**fields))
    if not fingerprint.exists() or not provenance.exists():
        update('完整fit来源规划指纹（8例；不含cal/target）')
        extractor=DatasetFingerprintExtractor(split['dataset_id'],num_processes=1,verbose=False)
        extractor.dataset={cid:extractor.dataset[cid] for cid in split['train']}
        extractor.dataset_json=dict(extractor.dataset_json,numTraining=len(split['train']))
        extractor.run(overwrite_existing=True)
        write_json(provenance,dict(fit_ids=split['train'],excluded_val=split['val'],
            excluded_target=split['target'],sha256=digest(fingerprint)))
    prov=read_json(provenance)
    if prov['fit_ids']!=split['train'] or prov['sha256']!=digest(fingerprint):
        raise ValueError('测评规划指纹来源校验失败')
    plan=prep/'WholeheartBaselinePlans.json'
    if not plan.exists():
        update('生成正式尺寸网络规划（CPU，不申请GPU）')
        planner=ExperimentPlanner(split['dataset_id'],gpu_memory_target_in_gb=gpu_memory,
                                  plans_name='WholeheartBaselinePlans')
        planner.dataset={cid:planner.dataset[cid] for cid in split['train']}
        planner.dataset_json=dict(planner.dataset_json,numTraining=len(split['train']))
        planner.plan_experiment()
    write_json(prep/'splits_final.json',[dict(train=split['train'],val=split['val'])])
    manager=PlansManager(read_json(plan));config=manager.get_configuration('3d_fullres')
    cache=prep/config.data_identifier;cache.mkdir(parents=True,exist_ok=True)
    selected=scope['train_ids']+scope['validation_ids']
    plan_hash=digest(plan)
    for i,cid in enumerate(selected):
        update('少量来源病例预处理',completed=i,total=len(selected),active_case=cid)
        done=prep/'assessment_case_completion'/f'{cid}.json'
        if not(done.exists() and read_json(done).get('plan_sha256')==plan_hash
                and all((cache/f'{cid}{suffix}').is_file() for suffix in ['.b2nd','_seg.b2nd','.pkl'])):
            config.preprocessor_class(verbose=False).run_case_save(str(cache/cid),
                [str(raw/'imagesTr'/f'{cid}_0000.nii.gz')],str(raw/'labelsTr'/f'{cid}.nii.gz'),
                manager,config,read_json(raw/'dataset.json'))
            write_json(done,dict(plan_sha256=plan_hash))
        print(f'独立测评预处理 {i+1}/{len(selected)}：{cid}',flush=True)
    update('测评预处理完成',completed=len(selected),total=len(selected))
    write_json(work/f'09_reports/{fold}_assessment_preparation.json',dict(status='passed',
        scope=scope,seconds=time.monotonic()-started,plan_sha256=plan_hash,
        nnunet_version=importlib.metadata.version('nnunetv2'),formal_cache_complete=False))
    return scope
