from math import prod
from pathlib import Path
import json
import time

import numpy as np

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..xu_fusion import _atomic_array, _refine_shape, _save_case, blood_grow, grid_prediction, vote
from ..xu_geometry import PreparedCache, _resize
from ..xu_roi import vote_box
from .heldout import HeldoutCache, prepare_case, verify_case
from .source import FINE, implementation, safe_path, verify_frozen


def verify_job(out):
    job = read_json(out/'run.json')
    if job['implementation'] != implementation() or job['source'] != digest(verify_frozen(out)):
        raise ValueError('Prediction implementation or frozen source changed')
    return job


def progress(out,stage,completed,total,started,**details):
    elapsed = time.monotonic()-started
    row = dict(stage=stage,completed=completed,total=total,seconds=elapsed,
               eta_seconds=elapsed*(total-completed)/completed if completed else None,
               updated_at=time.time(),**details)
    atomic_json(out/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def prepare(out):
    job = verify_job(out)
    started = time.monotonic()
    for i,name in enumerate(job['cases'],1):
        prepare_case(out,name)
        progress(out,'prepare',i,len(job['cases']),started,case=name)


def branch_record(out,branch,name,job):
    root = out/'branches'/branch
    marker = root/(name+'.json')
    if not marker.exists():
        return None
    record = read_json(safe_path(root,marker.name))
    if record['job'] != digest(job) or record['case'] != name or record['branch'] != branch:
        raise ValueError('Prediction branch identity changed')
    path = safe_path(root,name+'.npy')
    if sha256_file(path) != record['sha256']:
        raise ValueError('Prediction branch bytes changed')
    array = np.load(path,allow_pickle=False,mmap_mode='r')
    side = 128 if branch.endswith('128') else 64
    if array.shape != (side,)*3 or array.dtype != np.uint8 or array.max() > (6 if branch.startswith('init') else 10):
        raise ValueError('Invalid branch prediction grid')
    return array


def _branches(out,cache,training,source,job,device,branches,boxes=None):
    import torch
    from ..xu_workflow import frozen_predictor, input_id, side
    from ..xu_supply import BatchSupply
    for branch in branches:
        names = [name for name in job['cases'] if boxes is None or boxes[name]['quality']['accepted']]
        pending = [name for name in names if branch_record(out,branch,name,job) is None]
        if not pending:
            continue
        checkpoint = out/'source'/(branch+'.pt')
        if branch.startswith('init'):
            from ..xu_init import load_init_predictor
            predictor = load_init_predictor(checkpoint,cache=training,branch=branch,
                profile=source['run']['profile'],supervision=source['run']['supervision'],
                book=source['boxes'],device=device)
        else:
            training_boxes = None if branch.startswith('crop') else source['boxes']
            predictor,_ = frozen_predictor(checkpoint,branch=branch,profile=source['run']['profile'],
                supervision=source['run']['supervision'],data_identity=input_id(
                    training,source['run']['supervision'],training_boxes),device=device)
        n = side(branch)
        def read(batch):
            name = batch[0]
            box = None if boxes is None else boxes[name]['box']
            return (torch.from_numpy(cache.read_image(name,(n,)*3,box)),)
        supply = BatchSupply(pending,1,n,read,lambda name:prod(cache.geometry(name).cache_shape))
        started = time.monotonic()
        root = out/'branches'/branch
        root.mkdir(parents=True,exist_ok=True)
        with supply.batches() as batches:
            for i,(batch,values) in enumerate(batches,1):
                name = batch[0]
                label = predictor(supply.transfer(values,device)[0])
                path = root/(name+'.npy')
                _atomic_array(path,label)
                atomic_json(root/(name+'.json'),dict(job=digest(job),case=name,branch=branch,
                                                    sha256=sha256_file(path)))
                progress(out,branch,i,len(pending),started,case=name,supply=dict(supply.stats))
        del predictor


class BloodInputs:
    def __init__(self,training,images):
        self.training = training
        self.images = images

    def __getattr__(self,name):
        return getattr(self.training,name)

    def geometry(self,name):
        return self.images.geometry(name)

    def read_image(self,name,shape=None,box=None):
        return self.images.read_image(name,shape,box)


def case_record(out,name,job):
    marker = out/'cases'/(name+'.json')
    if not marker.exists():
        return None
    record = read_json(safe_path(out/'cases',marker.name))
    if record['job'] != digest(job) or record['case'] != name or record['role'] != job['cases'][name]['role']:
        raise ValueError('Completed case identity changed')
    if record['state'] == 'predicted':
        for suffix,key in (('.npy','label_sha256'),('.nii.gz','native_sha256')):
            if sha256_file(safe_path(out/'cases',name+suffix)) != record['result'][key]:
                raise ValueError('Completed prediction changed')
    elif record['state'] != 'roi_rejected' or record['roi']['quality']['accepted']:
        raise ValueError('Invalid terminal prediction state')
    return record


def predict(out,device):
    job = verify_job(out)
    source = verify_frozen(out)
    training = PreparedCache(job['prepared'],allow_technical_subset=job['technical_subset'])
    cache = HeldoutCache(out)
    for name in job['cases']:
        verify_case(out,name)
    _branches(out,cache,training,source,job,device,('crop64','crop128'))
    boxes = {name:vote_box(branch_record(out,'crop64',name,job),branch_record(out,'crop128',name,job),
                          cache.geometry(name),20) for name in job['cases']}
    atomic_json(out/'rois.json',dict(job=digest(job),cases=boxes))
    refiners = ['all64','all128'] + [b for b in ('init64','init128') if b in source['checkpoints']]
    _branches(out,cache,training,source,job,device,refiners,boxes)
    from ..xu_targets import anatomy
    mapping = anatomy(training)
    pending = [name for name in job['cases'] if case_record(out,name,job) is None]
    blood = None
    if job['mode'] != 'four-way' and any(boxes[n]['quality']['accepted'] for n in pending):
        from ..xu_blood import load_blood_predictor
        blood = load_blood_predictor(BloodInputs(training,cache),out/'source/blood.pt',device,
                                    source['run']['profile'],source['run']['supervision'])
    started = time.monotonic()
    (out/'cases').mkdir(exist_ok=True)
    for i,name in enumerate(pending,1):
        roi,geo = boxes[name],cache.geometry(name)
        record = dict(job=digest(job),case=name,role=job['cases'][name]['role'],roi=roi)
        if not roi['quality']['accepted']:
            record['state'] = 'roi_rejected'
        else:
            labels = {}
            for branch in (*FINE,*[b for b in refiners if b.startswith('init')]):
                labels[branch] = grid_prediction(branch_record(out,branch,name,job),geo.cache_shape,
                    None if branch.startswith('crop') else roi['box'],roi['box'],(128,)*3)
            small,qc = vote(labels,mapping)
            if blood is not None:
                shape = _refine_shape(roi['box'])
                small,details = blood_grow(_resize(small,shape,0),blood(name,roi['box'],shape),mapping['MYO'])
                qc['blood'] = details
            if job['mode'] == 'vessels':
                from ..xu_vessels import refine_vessels
                spacing = [geo.spacing_xyz[2-axis]*(stop-start)/count*(hi-lo)/size
                           for axis,(start,stop),count,(lo,hi),size in zip(
                               geo.transpose_forward,geo.bbox_transposed,geo.cache_shape,roi['box'],small.shape)]
                small,qc['vessels'] = refine_vessels(small,spacing,mapping)
            record.update(state='predicted',result=_save_case(out,name,geo,roi['box'],small,qc,cache,False))
        atomic_json(out/'cases'/(name+'.json'),record)
        progress(out,'prediction',i,len(pending),started,case=name,outcome=record['state'])
    records = {name:case_record(out,name,job) for name in job['cases']}
    verify_job(out)
    atomic_json(out/'complete.json',dict(job=digest(job),cases=records,
                coordinate_frame='local_array_excel_mm',rejected=sum(r['state']=='roi_rejected' for r in records.values())))
    return records
