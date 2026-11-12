from collections import defaultdict
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock, write_status
from ..xu_evaluation import class_metrics, summarize
from ..input_source import open_prepared as PreparedCache
from . import PROTOCOL


CLASSES = set(range(1,11))


def save_mask(output, relative, mask, row):
    if mask.shape != tuple(row['shape']) or mask.dtype != np.uint8:
        raise ValueError('预测掩膜与原 CT 网格不匹配')
    destination = output/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        path = Path(scratch)/destination.name
        image = nib.Nifti1Image(mask,np.diag([*row['spacing'],1.]))
        image.header.set_xyzt_units('mm')
        nib.save(image,path)
        sha = sha256_file(path)
        os.replace(path,destination)
    return dict(path=relative,sha256=sha)


def read_record(output, name, job):
    path = output/'records'/f'{name}.json'
    record = read_json(path)
    if record['job'] != digest(job) or record['case'] != name:
        raise ValueError('预测记录的来源不一致')
    for artifact in record['masks'].values():
        file = output/artifact['path']
        if file.resolve() != file or output not in file.parents or sha256_file(file) != artifact['sha256']:
            raise ValueError('已提交的预测掩膜发生变化')
    return record


def predict(cache, job, output, device):
    if job['model']=='stunet':
        from . import stunet
        from ..prediction.heldout import HeldoutCache, prepare_case
        training = Path(job['training'])
        if sha256_file(training/'training.json') != job['training_summary']:
            raise ValueError('STU-Net 训练摘要发生变化')
        model,patch,precision = stunet.load_predictor(training,cache,device)
        images = cache if job['development'] else HeldoutCache(output)
    else:
        from . import sam_med3d
        model = sam_med3d.load_model(job['weights'],device)
    records = {}
    for i,(name,row) in enumerate(job['cases'].items(),1):
        marker = output/'records'/f'{name}.json'
        if marker.exists():
            records[name] = read_record(output,name,job)
            continue
        write_status(output,state='running',stage=job['model']+'-predict',case=i,total=len(job['cases']))
        masks = {}
        if job['model']=='stunet':
            if not job['development']:
                prepare_case(output,name)
            predicted = stunet.predict(model,images.read_image(name),patch,device,precision)
            native = images.geometry(name).restore_label(predicted)
            masks['all'] = save_mask(output,f'cases/{name}.nii.gz',native,row)
        else:
            points = job['prompts'][name]
            if points:
                values = sam_med3d.load_ct(row)
                for label,point in points.items():
                    mask,info = sam_med3d.predict(model,values,row['spacing'],point,device)
                    masks[label] = dict(save_mask(output,f'cases/{name}/{label}.nii.gz',mask,row),**info)
                del values
        record = dict(job=digest(job),case=name,role=row['role'],masks=masks,
                      state='predicted' if masks else 'missing_prompt')
        atomic_json(marker,record)
        records[name] = record
        print(dict(case=i,total=len(job['cases']),state=record['state'],outputs=len(masks)),flush=True)
    atomic_json(output/'complete.json',dict(job=digest(job),records=records))


def native_reference(cache, name, row, development):
    if development:
        return cache.read_native_label(name),cache.supervision[name]
    if sha256_file(row['label']) != row['sha256']['label']:
        raise ValueError('参考掩膜的源文件发生变化')
    label = nib.load(row['label'])
    shifted = np.eye(4)
    shifted[:3,3] = 1
    if tuple(label.shape) != tuple(row['shape']) or not any(
            np.allclose(label.affine,affine,rtol=0,atol=1e-5) for affine in (np.eye(4),shifted)):
        raise ValueError('参考掩膜不符合已审核的原数组约定')
    values = np.asanyarray(label.dataobj)
    if (not np.isfinite(values).all() or values.min()<0 or values.max()>10
            or not np.equal(values,values.astype(np.uint8)).all()):
        raise ValueError('参考掩膜含无效类别')
    reference = values.astype(np.uint8,copy=False)
    present = set(map(int,np.unique(reference)))-{0}
    absent = set(row['known_absent'])
    return reference,dict(present_classes=sorted(present),known_absent_classes=sorted(absent),
                          unknown_classes=sorted(CLASSES-present-absent))


def read_mask(output, artifact, row, binary=False):
    image = nib.load(output/artifact['path'])
    if tuple(image.shape) != tuple(row['shape']) or not np.allclose(
            image.affine,np.diag([*row['spacing'],1.])):
        raise ValueError('预测掩膜的原图坐标不一致')
    mask = np.asanyarray(image.dataobj)
    if binary and (mask.dtype!=np.uint8 or mask.max()>1):
        raise ValueError('SAM-Med3D 输出必须为二值掩膜')
    return mask


def score_case(output, record, row, reference, supervision):
    masks = record['masks']
    if 'all' in masks:
        return class_metrics(reference,read_mask(output,masks['all'],row),row['spacing'],supervision)
    scores = {}
    for label in sorted(CLASSES):
        if label in supervision['unknown_classes']:
            scores[label] = dict(status='unannotated')
        elif str(label) not in masks:
            state = 'verified_absent' if label in supervision['known_absent_classes'] else 'missing_prompt'
            scores[label] = dict(status=state)
        else:
            binary = read_mask(output,masks[str(label)],row,binary=True)
            present = label in supervision['present_classes']
            single = dict(present_classes=[label] if present else [],
                          known_absent_classes=[] if present else [label],
                          unknown_classes=sorted(CLASSES-{label}))
            reference_target = (reference==label).astype(np.uint8)*label
            scores[label] = class_metrics(reference_target,binary*label,row['spacing'],single)[label]
    return scores


def evaluate(output, model):
    with operation_lock(output,PROTOCOL+'-evaluate',output):
        job = read_json(output/'run.json')
        if job['protocol']!=PROTOCOL or job['model']!=model or job['command']!='predict':
            raise ValueError('需要此模型已完成的独立预测目录')
        cache = PreparedCache(job['prepared'],allow_technical_subset=job['technical_subset'])
        if cache.request_digest != job['prepared_digest']:
            raise ValueError('原始划分或预处理发生变化')
        complete = read_json(output/'complete.json')
        records = {name:read_record(output,name,job) for name in job['cases']}
        if complete != dict(job=digest(job),records=records):
            raise ValueError('预测尚未完整提交')
        results,groups = {},defaultdict(list)
        for name,row in job['cases'].items():
            ref,supervision = native_reference(cache,name,row,job['development'])
            result = score_case(output,records[name],row,ref,supervision)
            results[name] = result
            groups[row['role']].append(result)
        reports = {}
        for group,rows in groups.items():
            summary = summarize(rows)
            for label,stats in summary['by_label'].items():
                stats['missing_prompt'] = sum(row[label]['status']=='missing_prompt' for row in rows)
            reports[group] = summary
        report = dict(job=digest(job),model=model,scope='technical' if job['technical_subset'] else 'held-out',
                      mode='automatic' if model=='stunet' else 'one-positive-point-per-structure',
                      groups=reports,cases=results)
        atomic_json(output/'evaluation.json',report)
        print(output/'evaluation.json',flush=True)
