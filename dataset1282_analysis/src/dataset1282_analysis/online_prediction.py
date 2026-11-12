from contextlib import contextmanager
from math import ceil, prod
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .contracts import digest, read_json, sha256_file
from .resources import preprocessing_memory
from .xu_geometry import CaseGeometry, _box_in, _resize


def online_job(out):
    job = read_json(Path(out)/'run.json')
    return (Path(job['prepared'])/'online.json').is_file()


def prepare_memory(job):
    plan = read_json(Path(job['prepared'])/'preprocessed/Dataset282_CHDSource/nnUNetPlans.json')
    targets = plan['configurations']['3d_fullres']['spacing']
    demands = []
    for row in job['cases'].values():
        spacing = np.asarray(row['spacing'], dtype=np.float32).astype(float)
        # 使用未裁剪的重采样尺寸上界，覆盖顺序处理的数组与插值工作区。
        shape = [ceil(row['shape'][2-axis]*spacing[2-axis]/target)
                 for axis,target in zip(plan['transpose_forward'], targets)]
        demands.append(preprocessing_memory(prod(row['shape']), prod(shape)))
    return max(demands)


class OnlineHeldout:
    def __init__(self, out):
        from .online_memory import CaseMemory
        self.out = Path(out)
        self.job = read_json(self.out/'run.json')
        self.training = Path(self.job['prepared'])
        self.memory = CaseMemory(int(os.environ.get('CHD_RAM_CACHE_BYTES', str(2*1024**3))))
        self._verified = {}

    def verify(self, name):
        from .prediction.heldout import verify_plan
        verify_plan(self.training)
        base = self.training/'preprocessed/Dataset282_CHDSource'
        for file, key in (('nnUNetPlans.json', 'plans'), ('dataset.json', 'dataset')):
            if sha256_file(base/file) != self.job[key]:
                raise ValueError('留出集预测的训练计划发生变化')
        row = self.job['cases'][name]
        path = Path(row['image'])
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if name in self._verified and self._verified[name] != identity:
            raise ValueError('留出集 CT 在读取期间发生变化')
        if name not in self._verified:
            if sha256_file(path) != row['sha256']['image']:
                raise ValueError('留出集 CT 来源摘要不同')
            image = nib.load(path)
            if image.shape != tuple(row['shape']) or not np.allclose(image.affine, np.eye(4), rtol=0, atol=1e-5):
                raise ValueError('留出集 CT 原始空间信息不同')
            after = path.stat()
            if identity != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('验证过程中 CT 发生变化')
            self._verified[name] = identity
        return dict(job=digest(self.job), case=name, source_sha256=row['sha256']['image'], mode='online')

    def _transform(self, row):
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        name = row['name']
        source = self.job['cases'][name]
        values = nib.load(source['image']).get_fdata(dtype=np.float32, caching='unchanged')
        if not np.isfinite(values).all() or values.min() == values.max():
            raise ValueError('留出集 CT 数值无效')
        base = self.training/'preprocessed/Dataset282_CHDSource'
        manager = PlansManager(str(base/'nnUNetPlans.json'))
        config = manager.get_configuration('3d_fullres')
        spacing = np.asarray(source['spacing'], dtype=np.float32).astype(float)
        data, _, props = config.preprocessor_class(verbose=False).run_case_npy(
            values.transpose(2,1,0)[None], None, dict(spacing=spacing[::-1].tolist()),
            manager, config, read_json(base/'dataset.json'))
        geometry = CaseGeometry(tuple(source['shape']), tuple(source['spacing']),
            tuple(manager.plans['transpose_forward']), tuple(map(tuple,props['bbox_used_for_cropping'])),
            tuple(data.shape[1:]))
        self.verify(name)
        return dict(image=data[0], geometry=geometry)

    @contextmanager
    def _case(self, name):
        from monai.data import Dataset
        from monai.transforms import Compose
        self.verify(name)
        estimate = prepare_memory(dict(self.job, cases={name:self.job['cases'][name]}))
        dataset = Dataset([dict(name=name)], Compose([self._transform]))
        with self.memory.lease(name, estimate, lambda:dataset[0]) as value:
            yield value

    def geometry(self, name):
        with self._case(name) as value:
            return value['geometry']

    def read_image(self, name, shape=None, box=None):
        with self._case(name) as value:
            region = tuple(slice(a,b) for a,b in _box_in(box,value['geometry'].cache_shape))
            return np.array(_resize(value['image'][region], shape, 1), copy=True, order='C')
