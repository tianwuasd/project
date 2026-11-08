from collections import defaultdict

import nibabel as nib
import numpy as np

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock
from ..xu_evaluation import class_metrics, summarize
from .heldout import HeldoutCache
from .workflow import case_record,verify_job


def evaluate(out):
    with operation_lock(out,'xu-prediction-evaluate',out):
        job = verify_job(out)
        complete = read_json(out/'complete.json')
        cases = {name:case_record(out,name,job) for name in job['cases']}
        if complete['job'] != digest(job) or complete['cases'] != cases or any(r is None for r in cases.values()):
            raise ValueError('Evaluation requires the complete unchanged prediction set')
        cache = HeldoutCache(out)
        results = {}
        for name,record in cases.items():
            row = job['cases'][name]
            if record['state'] == 'roi_rejected':
                results[name] = dict(role=row['role'],device=row['device'],state='roi_rejected')
                continue
            if sha256_file(row['label']) != row['sha256']['label']:
                raise ValueError('Held-out reference changed')
            volume = nib.load(row['label'])
            geo = cache.geometry(name)
            shift = np.eye(4)
            shift[:3,3] = 1
            if volume.shape != geo.native_shape_xyz or not any(np.allclose(volume.affine,a,rtol=0,atol=1e-5)
                                                              for a in (np.eye(4),shift)):
                raise ValueError('Held-out reference grid differs from the reviewed array convention')
            reference = np.asanyarray(volume.dataobj)
            if (not np.isfinite(reference).all() or reference.min()<0 or reference.max()>10
                    or not np.equal(reference,np.floor(reference)).all()):
                raise ValueError('Invalid held-out reference labels')
            reference = reference.astype(np.uint8,copy=False)
            present = sorted(set(map(int,np.unique(reference)))-{0})
            absent = row['known_absent']
            if not present or set(absent)&set(present) or set(absent)-set(range(1,11)):
                raise ValueError('Invalid held-out supervision metadata')
            info = dict(present_classes=present,known_absent_classes=absent,
                        unknown_classes=sorted(set(range(1,11))-set(present)-set(absent)))
            image = nib.load(out/'cases'/(name+'.nii.gz'))
            if image.shape != geo.native_shape_xyz or not np.allclose(image.affine,np.diag([*geo.spacing_xyz,1.])):
                raise ValueError('Prediction export geometry changed')
            metrics = class_metrics(reference,np.asanyarray(image.dataobj),geo.spacing_xyz,info)
            results[name] = dict(role=row['role'],device=row['device'],state='evaluated',
                                 complete_annotation=len(present)==10,metrics=metrics)
        groups = defaultdict(list)
        devices = defaultdict(list)
        for row in results.values():
            groups[row['role']].append(row)
            devices[row['role']+'/'+row['device']].append(row)
        def summary(rows):
            good = [r for r in rows if r['state']=='evaluated']
            return dict(selected=len(rows),evaluated=len(good),roi_rejected=len(rows)-len(good),
                        observed_structure=summarize([r['metrics'] for r in good]),
                        full_annotation=summarize([r['metrics'] for r in good if r['complete_annotation']]),
                        metric_population='successfully_predicted_cases; rejected cases reported separately')
        report = dict(job=digest(job),cases=results,groups={k:summary(v) for k,v in groups.items()},
                      devices={k:summary(v) for k,v in devices.items()},
                      coordinate_frame='local_array_excel_mm',models=digest(read_json(out/'source/manifest.json')))
        atomic_json(out/'evaluation.json',report)
        print('Evaluation:',out/'evaluation.json',flush=True)
        return report
