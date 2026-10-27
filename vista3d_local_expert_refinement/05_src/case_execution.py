"""Dedicated CPU case execution: scoped library settings, unchanged scientific data."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def leaf_threads(count):
    """Override explicit nnU-Net Blosc compression threads in this case process only.

    Do not use around concurrent threads. Codecs, clevel, chunks, blocks, filters
    and arrays pass through unchanged, including official filter selection.
    """
    if int(count)<1:raise ValueError('positive leaf thread count required')
    import blosc2
    original=blosc2.asarray
    previous=blosc2.set_nthreads(int(count))
    evidence=dict(blosc_calls=[],configured_threads=int(count),resampling_threads=1)
    def adapted(*args,**kwargs):
        params=dict(kwargs.get('cparams') or {})
        params['nthreads']=int(count)
        kwargs['cparams']=params
        row=dict(nthreads=count,clevel=params.get('clevel'),codec=str(params.get('codec')),filters=str(params.get('filters')))
        evidence['blosc_calls'].append(row);started=time.monotonic()
        try:return original(*args,**kwargs)
        finally:row['seconds']=time.monotonic()-started
    blosc2.asarray=adapted
    try:yield evidence
    finally:
        blosc2.asarray=original
        blosc2.set_nthreads(previous)


@contextmanager
def cpu_scope(count):
    import torch
    from task_resources import apply_training_cpu
    keys=['WHOLE_HEART_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','nnUNet_n_proc_DA']
    prior={k:os.environ.get(k) for k in keys};old=torch.get_num_threads()
    limiter=apply_training_cpu(dict(threads=count),torch)
    os.environ['nnUNet_n_proc_DA']=str(count)
    try:yield
    finally:
        limiter.restore_original_limits();torch.set_num_threads(old)
        for key,value in prior.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def planner_threads(work,raw,case_ids):
    from task_resources import case_budget
    value=case_budget(work,[Path(raw)/'imagesTr'/f'{cid}_0000.nii.gz' for cid in case_ids],maximum=1,phase='planner')
    if value['workers']<1:raise MemoryError('planner capacity: '+value['capacity_status'])
    return value['threads_per_worker']


def fingerprint_workers(work,raw,case_ids):
    from task_resources import case_budget
    budget=case_budget(work,[Path(raw)/'imagesTr'/f'{cid}_0000.nii.gz' for cid in case_ids],phase='fit_fingerprint',supported_threads=1)
    if budget['workers']<1:raise MemoryError('fit fingerprint capacity: '+budget['capacity_status'])
    from read_admission import create_gate
    os.environ['VISTA_READ_GATE']=str(create_gate(budget['read_concurrency']))
    return budget['workers']


def prepare_one(work,fold,case_id,assessment=False):
    from server_data import read_json,write_json,digest
    from prepare_nnunet import configure_paths
    os.environ['WHOLE_HEART_WORKSPACE']=str(work)
    configure_paths()
    import torch
    from task_resources import apply_training_cpu
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from runtime_observations import event,failure
    from threadpoolctl import threadpool_info
    work=Path(work);count=max(1,int(os.environ.get('WHOLE_HEART_CPU_THREADS','1')))
    limiter=apply_training_cpu(dict(threads=count),torch)
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    allowed=split['train']+split['val']
    if assessment:
        scope=read_json(work/'00_admin/assessment.json')['scope']
        allowed=scope['train_ids']+scope['validation_ids']
    if case_id not in allowed:raise ValueError('病例不在冻结来源范围')
    name=f'Dataset{split["dataset_id"]:03d}_{fold}'
    prep=work/'04_data/derived/nnUNet_preprocessed'/name;raw=work/'04_data/derived/nnUNet_raw'/name
    plan=prep/'WholeheartBaselinePlans.json';manager=PlansManager(read_json(plan));config=manager.get_configuration('3d_fullres')
    import tempfile
    staging_root=prep/'server_case_staging';staging_root.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='case-',dir=staging_root))
    output=staging/case_id
    if not output.resolve().is_relative_to(staging.resolve()):raise ValueError('病例写出路径超出暂存目录')
    started=time.monotonic()
    try:
        from read_admission import reader_scope
        with leaf_threads(count) as evidence,reader_scope(manager.image_reader_writer_class):
            config.preprocessor_class(verbose=False).run_case_save(str(output),
                [str(raw/'imagesTr'/f'{case_id}_0000.nii.gz')],str(raw/'labelsTr'/f'{case_id}.nii.gz'),
                manager,config,read_json(raw/'dataset.json'))
        # Official Blosc writer cannot overwrite an existing path. Commit a fresh
        # complete case, then the completion marker; retries keep prior cases.
        for suffix in ['.b2nd','_seg.b2nd','.pkl']:
            source=staging/(case_id+suffix);destination=prep/config.data_identifier/(case_id+suffix)
            os.replace(source,destination)
        from task_resources import process_observation
        import nibabel as nib
        from case_memory import compatibility
        evidence.update(process_observation(),seconds=time.monotonic()-started,torch_threads=torch.get_num_threads(),
            status='completed',observed_at=time.time(),
            compatibility=compatibility(nib.load(str(raw/'imagesTr'/f'{case_id}_0000.nii.gz')),list(reversed(config.spacing)),
                'assessment_prepare' if assessment else 'prepare'),
            output_bytes=sum((prep/config.data_identifier/(case_id+s)).stat().st_size for s in ['.b2nd','_seg.b2nd','.pkl']))
        done=prep/'assessment_case_completion'/f'{case_id}.json' if assessment else prep/'server_case_completion'/config.data_identifier/f'{case_id}.complete.json'
        write_json(done,dict(plan_sha256=digest(plan)))
        write_json(work/f'00_admin/case-observations/{fold}_{case_id}.json',evidence)
        event(work,'prepare.case_completed',fold=fold,case_id=case_id,observations=evidence)
    except BaseException as error:
        failure(work,'prepare_case',error,fold=fold,case_id=case_id);raise
    finally:
        limiter.restore_original_limits()
        # Only this newly created directory and its three known files are ours.
        if not staging.resolve().is_relative_to(staging_root.resolve()):raise ValueError('暂存目录身份变化')
        for suffix in ['.b2nd','_seg.b2nd','.pkl']:(staging/(case_id+suffix)).unlink(missing_ok=True)
        staging.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('work');parser.add_argument('fold');parser.add_argument('case')
    parser.add_argument('--assessment',action='store_true');args=parser.parse_args()
    prepare_one(args.work,args.fold,args.case,args.assessment)
