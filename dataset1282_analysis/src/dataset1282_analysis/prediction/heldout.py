from dataclasses import asdict
from pathlib import Path
from math import ceil, prod
import os
import re
import tempfile

import numpy as np

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..xu_geometry import CaseGeometry, _box_in, _resize
from .source import safe_path
from .preprocessed import FORMATS


GROUPS = ('source_test','target_primary','target_secondary')


def select_cases(cohort, groups, cases=None):
    if not groups or set(groups)-set(GROUPS):
        raise ValueError('Prediction admits only frozen held-out groups')
    eligible = {name:row for name,row in cohort['cases'].items()
                if row['role'] in groups and row.get('included') and row.get('index_scan')}
    names = list(cases) if cases else sorted(eligible)
    if (not names or len(set(names)) != len(names) or set(names)-eligible.keys()
            or any(not re.fullmatch(r'[A-Za-z0-9_-]+',name) for name in names)):
        raise ValueError('Select unique included index scans from the requested held-out groups')
    training = {r['patient'] for r in cohort['cases'].values()
                if r['role'] in ('train','development') and r.get('patient')}
    patients = [eligible[name]['patient'] for name in names]
    if not all(patients) or len(patients) != len(set(patients)) or set(patients) & training:
        raise ValueError('Patient leakage across frozen roles')
    return {name:eligible[name] for name in names}


def plan_inputs(cache, config, policy_path, groups, cases=None):
    import nibabel as nib
    from ..inputs import load_cases
    from ..metadata import read_metadata
    from ..preparation import allocate, excel_spacing
    from ..preprocessing.workflow import derive_cohort, shared_geometry
    records = load_cases(config)
    rows = read_metadata(config['metadata_excel'],config.get('metadata_sheet','dicom2nii_info'))
    identities = dict(manifest=sha256_file(Path(config['data_root'])/'manifest.json'),
                      excel=sha256_file(config['metadata_excel']))
    original = dict(allocate(records,rows),inputs=identities)
    policy = read_json(policy_path)
    cohort, links = derive_cohort(records,rows,original,policy,identities)
    frozen = read_json(cache.run_dir/'cohort.json')
    selected = select_cases(frozen,groups,cases)
    for name,row in selected.items():
        if any(cohort['cases'][name].get(k) != row.get(k)
               for k in ('role','patient','included','index_scan')):
            raise ValueError('Held-out case membership or patient allocation changed')
    sources = {row.case_id:row for row in records}
    output = {}
    for name,row in selected.items():
        source = sources[name]
        if links[name] is None or set(source.expected_hashes) != {'image','label'}:
            raise ValueError('Missing frozen input identity')
        volume = nib.load(source.image)
        spacing = excel_spacing(rows[links[name]],volume.shape)
        shared_geometry(volume,volume,spacing)
        output[name] = dict(role=row['role'],device=row['device'],patient=row['patient'],
                            image=source.image,label=source.label,sha256=source.expected_hashes,
                            shape=list(volume.shape),spacing=spacing,
                            known_absent=policy.get('known_absent_classes',{}).get(name,[]))
    return output


def prepare_memory(job):
    plan = read_json(Path(job['prepared'])/'preprocessed/Dataset282_CHDSource/nnUNetPlans.json')
    spacing = plan['configurations']['3d_fullres']['spacing']
    permutation = plan['transpose_forward']
    demands = []
    for row in job['cases'].values():
        shape = [ceil(row['shape'][2-axis]*row['spacing'][2-axis]/target)
                 for axis,target in zip(permutation,spacing)]
        demands.append(prod(row['shape'])*32+prod(shape)*48)
    return max(demands)+2*1024**3


def prepare_case(out, name):
    import blosc2
    import nibabel as nib
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    job = read_json(out/'run.json')
    row = job['cases'][name]
    training = Path(job['prepared'])
    plans_path = training/'preprocessed/Dataset282_CHDSource/nnUNetPlans.json'
    dataset_path = training/'preprocessed/Dataset282_CHDSource/dataset.json'
    source = Path(row['image'])
    destination = out/'prepared'/name
    if destination.exists():
        return verify_case(out,name)
    destination.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.preparing-',dir=destination.parent) as scratch:
        root = Path(scratch)
        image = nib.load(source)
        if list(image.shape) != row['shape'] or not np.allclose(image.affine,np.eye(4),rtol=0,atol=1e-5):
            raise ValueError('Unexpected held-out native CT grid')
        values = image.get_fdata(dtype=np.float32,caching='unchanged')
        if not np.isfinite(values).all() or values.min() == values.max():
            raise ValueError('Nonfinite or constant held-out CT')
        working = nib.Nifti1Image(values,np.diag([*row['spacing'],1.]))
        working.header.set_xyzt_units('mm')
        staged = root/'image.nii'
        nib.save(working,staged)
        del working,values,image
        manager = PlansManager(str(plans_path))
        config = manager.get_configuration('3d_fullres')
        processor = config.preprocessor_class(verbose=False)
        # 仅应用训练计划，标签参数固定为空。
        data, _, props = processor.run_case([str(staged)],None,manager,config,str(dataset_path))
        if data.dtype != np.float32 or data.shape[0] != 1 or not np.isfinite(data).all():
            raise ValueError('Invalid held-out preprocessed CT')
        geometry = CaseGeometry(tuple(row['shape']),tuple(row['spacing']),
                                tuple(manager.plans['transpose_forward']),
                                tuple(map(tuple,props['bbox_used_for_cropping'])),tuple(data.shape[1:]))
        result = root/'result'
        result.mkdir()
        array = blosc2.asarray(data,urlpath=str(result/'image.b2nd'),mode='w')
        del array,data
        atomic_json(result/'geometry.json',asdict(geometry))
        marker = dict(job=digest(job),case=name,source_sha256=row['sha256']['image'],
                      artifacts={p.name:sha256_file(p) for p in result.iterdir()})
        atomic_json(result/'complete.json',marker)
        os.replace(result,destination)
    return verify_case(out,name)


def verify_case(out,name):
    job = read_json(out/'run.json')
    if job.get('input_format') in FORMATS:
        from .preprocessed import from_job
        from_job(job).geometry(name)
        return dict(case=name,job=digest(job),external_preprocessed=True)
    folder = out/'prepared'/name
    marker = read_json(safe_path(folder,'complete.json'))
    job = read_json(out/'run.json')
    if (marker['job'] != digest(job) or marker['case'] != name
            or marker['source_sha256'] != job['cases'][name]['sha256']['image']
            or set(marker['artifacts']) != {'image.b2nd','geometry.json'}):
        raise ValueError('Held-out prepared identity changed')
    for relative in marker['artifacts']:
        safe_path(folder,relative).stat()
    return marker


class HeldoutCache:
    def __init__(self,out):
        self.out = Path(out)
        self.job = read_json(self.out/'run.json')
        self._verified = {}
        self.preprocessed = None
        if self.job.get('input_format') in FORMATS:
            from .preprocessed import from_job
            self.preprocessed=from_job(self.job)

    def geometry(self,name):
        if self.preprocessed is not None:
            return self.preprocessed.geometry(name)
        if name not in self.job['cases']:
            raise ValueError('Case is outside the frozen prediction selection')
        folder = self.out/'prepared'/name
        identity = tuple((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
                         for p in (folder/'image.b2nd',folder/'geometry.json',folder/'complete.json'))
        if self._verified.get(name) != identity:
            verify_case(self.out,name)
            self._verified[name] = identity
        row = read_json(self.out/'prepared'/name/'geometry.json')
        row = {key:tuple(map(tuple,value)) if key=='bbox_transposed' else tuple(value)
               for key,value in row.items()}
        return CaseGeometry(**row)

    def read_image(self,name,shape=None,box=None):
        if self.preprocessed is not None:
            return self.preprocessed.read_image(name,shape,box)
        import blosc2
        geo = self.geometry(name)
        data = blosc2.open(str(self.out/'prepared'/name/'image.b2nd'),mode='r')
        if tuple(data.shape) != (1,*geo.cache_shape):
            raise ValueError('Prepared image shape mismatch')
        region = (slice(0,1),*(slice(lo,hi) for lo,hi in _box_in(box,geo.cache_shape)))
        array = np.asarray(data[region])[0]
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid prepared intensities')
        return _resize(array,shape,1)
