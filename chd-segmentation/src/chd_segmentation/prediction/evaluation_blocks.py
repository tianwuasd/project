from contextlib import ExitStack
import gzip
import math
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from ..acceleration import ResourceDemand, ResourceSession, RuntimeRequest
from ..acceleration.legacy import SessionGuard
from ..models.stunet.streaming import tiles
from .workspace import PredictionWorkspace


def local_image(path, name, workspace):
    path = Path(path)
    if path.suffix != '.gz':
        return nib.load(path)
    image = nib.load(path)
    workspace.check_disk(math.prod(image.shape)*image.get_data_dtype().itemsize+16*1024**2)
    destination = workspace.path/(name+'.nii')
    with gzip.open(path, 'rb') as source, destination.open('xb') as target:
        shutil.copyfileobj(source, target, length=8*1024**2)
    return nib.load(destination)


def count_labels(prediction_path, reference_path, roi, classes, unknown, ignore, config):
    block_bytes = config['prediction'].get('max_block_bytes', 64*1024**2)
    resources = dict(config['resources'],device='cpu',cpu_processes=1,gpu_ids=None,
                     memory_policy='host_fraction',max_wait=config['resources']['wait_seconds'])
    demand = ResourceDemand(fixed_bytes=math.ceil((512*1024**2+4*block_bytes)*
        (1+resources.get('memory_margin_fraction',.5))),cpu_slots=1)
    request = RuntimeRequest(str(prediction_path),'evaluation',Path(config['work_root'])/'resources',
                             config['run_dir'],resources,demand)
    with ExitStack() as cleanup:
        session = cleanup.enter_context(ResourceSession(request))
        guard = cleanup.enter_context(SessionGuard(session.handle))
        workspace = cleanup.enter_context(PredictionWorkspace(config,
            dict(evaluation=str(prediction_path),reference=str(reference_path)),block_bytes,guard))
        prediction_image = local_image(prediction_path,'prediction',workspace)
        reference_image = local_image(reference_path,'reference',workspace)
        if prediction_image.shape != reference_image.shape:
            raise ValueError('评价标签与预测原图尺寸不同')
        if not np.allclose(prediction_image.affine,reference_image.affine,rtol=0,atol=1e-5):
            raise ValueError('评价标签与预测空间不同')
        shape = tuple(s.stop-s.start for s in roi)
        counts = {c:[0,0,0] for c in range(1,classes) if c not in unknown}
        parts = tiles(shape,max(1,block_bytes//64))
        for index, part in enumerate(parts,1):
            with workspace.boundary():
                section = tuple(slice(s.start+base.start,s.stop+base.start) for s,base in zip(part,roi))
                prediction = np.asanyarray(prediction_image.dataobj[section])
                truth = np.asanyarray(reference_image.dataobj[section])
                if not np.isfinite(truth).all() or not np.equal(truth,np.rint(truth)).all():
                    raise ValueError('评价标签不是有限整数')
                truth = truth.astype(np.int16)
                truth[np.isin(truth,ignore)] = -1
                if not ((truth>=-1)&(truth<classes)).all():
                    raise ValueError('评价标签超出协议范围')
                valid = truth>=0
                for c, value in counts.items():
                    target, found = (truth==c)&valid, (prediction==c)&valid
                    value[0] += int(target.sum())
                    value[1] += int(found.sum())
                    value[2] += int((target&found).sum())
                    del target, found
                del prediction, truth, valid
            workspace.progress('evaluation',index,None)
        del prediction_image, reference_image
    return {str(c):dict(reference_voxels=target,predicted_voxels=found,
        dice=float(2*overlap/(target+found)) if target+found else None)
        for c,(target,found,overlap) in counts.items()}
