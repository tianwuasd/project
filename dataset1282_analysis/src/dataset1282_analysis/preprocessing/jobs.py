"""Isolated nnU-Net 2.8.1 jobs. Each commit is written only after artifact verification."""
from importlib.metadata import version
import os
from pathlib import Path
import sys
import threading
import time

from ..contracts import atomic_json, read_json, sha256_file
from ..vendor.vista_read_admission import read_slot, reader_scope, fingerprint_admission
from .workflow import DATASET, shared_geometry


def stage(output, request, name):
    import nibabel as nib
    import numpy as np
    record=request['cases'][name]
    folder=output/'staged'/name
    folder.mkdir(parents=True,exist_ok=True)
    with read_slot():
        image,label=[nib.load(record[k]) for k in ('image','label')]
        affine=shared_geometry(image,label,record['spacing'])
        data=image.get_fdata(dtype=np.float32,caching='unchanged')
        seg=np.asanyarray(label.dataobj)
    if not np.isfinite(data).all() or data.min()==data.max():
        raise ValueError('Nonfinite or constant CT')
    counts=np.zeros(11,dtype=np.int64)
    # Slab validation avoids whole-volume float temporaries and sorting label voxels.
    for z in range(0,seg.shape[2],8):
        slab=seg[:,:,z:z+8]
        if (not np.isfinite(slab).all() or slab.min()<0 or slab.max()>10
                or (slab.dtype.kind=='f' and not np.equal(slab,np.floor(slab)).all())):
            raise ValueError('Labels require finite integer foreground 0..10')
        counts+=np.bincount(slab.astype(np.uint8,copy=False).ravel(),minlength=11)
    present=set(map(int,np.flatnonzero(counts)))-{0}
    if not present:
        raise ValueError('Empty foreground')
    seg=seg.astype(np.uint8,copy=False)
    absent=set(record['known_absent'])
    if not absent.issubset(set(range(1,11))-present):
        raise ValueError('Reviewed absent classes conflict with observed labels')
    unknown=sorted(set(range(1,11))-present-absent)
    metadata=dict(case=name,role=record['role'],original_image_affine=image.affine.tolist(),
                  original_label_affine=label.affine.tolist(),working_affine=affine.tolist(),
                  source_spacing_xyz=record['spacing'],source_sha256=record['sha256'],
                  intensity=dict(assumed_unit='HU',additional_scale=1,additional_offset=0,verified=False),
                  supervision=dict(present_classes=sorted(present),known_absent_classes=sorted(absent),
                                   unknown_classes=unknown,ambiguous_background_allowed=[0]+unknown))
    paths=[]
    for kind,array in [('image',data),('label',seg.astype(np.uint8,copy=False))]:
        path=folder/(kind+request['staged_extension'])
        volume=nib.Nifti1Image(array,affine)
        volume.header.set_xyzt_units('mm')
        nib.save(volume,path)
        paths.append(path)
    atomic_json(folder/'metadata.json',metadata)
    return paths+[folder/'metadata.json']


def descriptor(request):
    return dict(channel_names={'0':'CT'}, labels={v:int(k) for k,v in request['labels'].items()},
                file_ending=request['staged_extension'],overwrite_image_reader_writer='NibabelIOWithReorient')


def fit(output, request):
    import numpy as np
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    raw=output/'raw'/DATASET
    dest=output/'preprocessed'/DATASET
    raw.mkdir(parents=True,exist_ok=True)
    dest.mkdir(parents=True,exist_ok=True)
    train=sorted(n for n,r in request['cases'].items() if r['role']=='train')
    ext=request['staged_extension']
    desc=dict(descriptor(request),numTraining=len(train),dataset={n:dict(
        images=[str(output/'staged'/n/('image'+ext))],label=str(output/'staged'/n/('label'+ext))) for n in train})
    atomic_json(raw/'dataset.json',desc)
    extractor=DatasetFingerprintExtractor(DATASET,num_processes=1)
    extractor.num_foreground_voxels_for_intensitystats=request['fingerprint_samples']
    with fingerprint_admission():
        extractor.run(overwrite_existing=True)
    plans=ExperimentPlanner(DATASET).plan_experiment()
    conf=PlansManager(plans).get_configuration('3d_fullres')
    atomic_json(dest/'dataset.json',dict(descriptor(request),numTraining=len(request['cases'])))
    result=dict(fit_cases=train,configuration='3d_fullres',data_identifier=conf.data_identifier,
                spacing_xyz=np.array(conf.spacing)[plans['transpose_backward']][::-1].tolist(),
                fingerprint_sha256=sha256_file(dest/'dataset_fingerprint.json'),
                plans_sha256=sha256_file(dest/'nnUNetPlans.json'),
                normalization=conf.normalization_schemes,fingerprint_samples=request['fingerprint_samples'])
    atomic_json(output/'fit.json',result)
    return [raw/'dataset.json',dest/'dataset.json',dest/'dataset_fingerprint.json',dest/'nnUNetPlans.json',output/'fit.json']


def process(output, request, name):
    import pickle
    import shutil
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from ..nnunet_prepare import roundtrip
    dest=output/'preprocessed'/DATASET
    manager=PlansManager(str(dest/'nnUNetPlans.json'))
    conf=manager.get_configuration('3d_fullres')
    target=dest/conf.data_identifier
    target.mkdir(exist_ok=True)
    attempt=output/'attempts'/name
    attempt.mkdir(parents=True,exist_ok=True)
    suffixes=('.b2nd','_seg.b2nd','.pkl')
    # Only this job's generated scratch files may be replaced after an interruption.
    for suffix in suffixes:
        path=attempt/(name+suffix)
        if output.resolve() not in path.resolve().parents:
            raise ValueError('Linked numerical scratch output')
        if path.exists(): path.unlink()
    ext=request['staged_extension']
    folder=output/'staged'/name
    processor=conf.preprocessor_class(verbose=True)
    with reader_scope(manager.image_reader_writer_class):
        processor.run_case_save(str(attempt/name),[str(folder/('image'+ext))],
                                str(folder/('label'+ext)),manager,conf,descriptor(request))
    metadata=read_json(folder/'metadata.json')
    # Retained with the case for external readers as well as the keyed Trainer adapter.
    with (attempt/(name+'.pkl')).open('rb') as stream:
        props=pickle.load(stream)
    props['partial_supervision']=metadata['supervision']
    with (attempt/(name+'.pkl')).open('wb') as stream:
        pickle.dump(props,stream)
    check=roundtrip(attempt,name,str(folder/('label'+ext)),manager.plans)
    if any(v['original_voxels']>0 and v['returned_voxels']==0 for v in check['classes'].values()):
        raise ValueError('An observed class disappeared after preprocessing')
    check['supervision']=metadata['supervision']
    for suffix in suffixes:
        destination=target/(name+suffix)
        if output.resolve() not in destination.resolve().parents:
            raise ValueError('Linked numerical output')
        os.replace(attempt/(name+suffix),destination)
    atomic_json(output/'qc'/(name+'.json'),check)
    paths=[target/(name+suffix) for suffix in ('.b2nd','_seg.b2nd','.pkl')]+[output/'qc'/(name+'.json')]
    if request['cases'][name]['role']=='development':
        gt=dest/'gt_segmentations'/(name+ext)
        gt.parent.mkdir(exist_ok=True)
        shutil.copyfile(folder/('label'+ext),gt)
        paths.append(gt)
    return paths


def main(path):
    from ..observability.status import owner_alive
    if version('nnunetv2')!='2.8.1':
        raise RuntimeError('Requires nnunetv2==2.8.1')
    job=read_json(path)
    if job['job']['kind']!='stage':
        import torch
        import blosc2
        torch.set_num_threads(1)
        blosc2.set_nthreads(1)
    output=Path(job['output'])
    request=read_json(output/'request.json')
    stopped=threading.Event()
    def watchdog():
        while not stopped.wait(1):
            if not owner_alive(Path(job['home']),job['owner']):
                import psutil
                for p in psutil.Process().children(recursive=True):
                    try: p.terminate()
                    except psutil.Error: pass
                os._exit(92)
    threading.Thread(target=watchdog,daemon=True).start()
    spec=job['job']
    started=time.monotonic()
    print('JOB_START',spec,flush=True)
    try:
        if spec['kind']=='fit':
            artifacts=fit(output,request)
        else:
            artifacts=globals()[spec['kind']](output,request,spec['case'])
        result=dict(job=spec,contract=job['contract'],seconds=time.monotonic()-started,
                    artifacts={p.relative_to(output).as_posix():sha256_file(p) for p in artifacts})
        atomic_json(output/'commits'/(spec['id']+'.json'),result)
        print('JOB_COMPLETE',result,flush=True)
    finally:
        stopped.set()


if __name__=='__main__':
    main(sys.argv[1])
