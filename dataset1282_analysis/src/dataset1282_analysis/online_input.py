from contextlib import contextmanager
from importlib.metadata import version
from math import prod
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .contracts import read_json, sha256_file
from .input_source import LegacyInput
from .online_memory import CaseMemory
from .resources import preprocessing_memory
from .xu_geometry import _box_in, _resize


class OnlineInput(LegacyInput):
    online = True

    def __init__(self, run_dir, *, allow_technical_subset=False):
        super().__init__(run_dir, allow_technical_subset=allow_technical_subset)
        marker = read_json(self.run_dir/'online.json')
        if marker['schema'] != 'chd-online-1' or marker['request_digest'] != self.request_digest:
            raise ValueError('在线输入与预处理请求不一致')
        for relative, expected in marker['metadata'].items():
            self._check_file(relative, expected)
        if version('nnunetv2') != '2.8.1':
            raise ValueError('在线处理需要原预处理的 nnU-Net 2.8.1')
        from .prediction.heldout import verify_plan
        verify_plan(self.run_dir)
        self.memory = CaseMemory(int(os.environ.get('CHD_RAM_CACHE_BYTES', str(2*1024**3))))
        self.arrays = self.memory
        self._sources = {}
        self._dataset_reader = None
        self._indices = {name: i for i, name in enumerate(sorted(self._records))}

    def _source(self, name, kind):
        row = self._records[name]
        path = Path(row[kind]).resolve(strict=True)
        stat = path.stat()
        stamp = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (name, kind)
        previous = self._sources.get(key)
        if previous is not None and previous != stamp:
            raise ValueError('读取期间原始数据发生变化')
        if previous is None:
            if sha256_file(path) != row['sha256'][kind]:
                raise ValueError('原始数据摘要与冻结请求不同')
            after = path.stat()
            if stamp[1:] != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('摘要检查期间原始数据发生变化')
            self._sources[key] = stamp
        return path

    def verify_image(self, name):
        return self._source(name, 'image')

    def native_image_path(self, name):
        return self._source(name, 'image')

    def _native(self, name):
        row = self._records[name]
        image, label = (nib.load(self._source(name, kind)) for kind in ('image', 'label'))
        from .preprocessing.workflow import shared_geometry
        shared_geometry(image, label, row['spacing'])
        if tuple(row['shape']) != image.shape:
            raise ValueError('原始数组尺寸与冻结请求不同')
        data = image.get_fdata(dtype=np.float32, caching='unchanged')
        seg = np.asanyarray(label.dataobj)
        if (not np.isfinite(data).all() or data.min() == data.max() or not np.isfinite(seg).all()
                or seg.min() < 0 or seg.max() > 10 or not np.equal(seg, seg.astype(np.uint8)).all()):
            raise ValueError('原始 CT 或标签数值无效')
        present = sorted(int(c) for c in np.unique(seg) if c)
        if present != self.supervision[name]['present_classes']:
            raise ValueError('原始标签类别与已审核监督信息不同')
        return dict(image=data, label=seg.astype(np.uint8, copy=False))

    def _transform(self, row):
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        name = row['name']
        native = self._native(name)
        manager = PlansManager(str(self.run_dir/self._dataset/'nnUNetPlans.json'))
        config = manager.get_configuration('3d_fullres')
        # 复现旧 NIfTI-1 头的 float32 毫米精度，确保重采样数值一致。
        spacing = np.asarray(self._records[name]['spacing'], dtype=np.float32).astype(float)
        props = dict(spacing=spacing[::-1].tolist())
        image, label, props = config.preprocessor_class(verbose=False).run_case_npy(
            native['image'].transpose(2, 1, 0)[None],
            native['label'].astype(np.float32).transpose(2, 1, 0)[None],
            props, manager, config, read_json(self.run_dir/self._dataset/'dataset.json'))
        geo = self.geometry(name)
        if (image.shape != (1, *geo.cache_shape) or label.shape != image.shape
                or tuple(map(tuple, props['bbox_used_for_cropping'])) != geo.bbox_transposed
                or image.dtype != np.float32 or not np.isfinite(image).all()):
            raise ValueError('在线处理结果与已审核空间信息不同')
        self._source(name, 'image')
        self._source(name, 'label')
        return dict(image=image[0], label=label[0], properties=props)

    def _build(self, name):
        if self._dataset_reader is None:
            from monai.data import Dataset
            from monai.transforms import Compose
            self._dataset_reader = Dataset([dict(name=n) for n in sorted(self._records)],
                                           transform=Compose([self._transform]))
        return self._dataset_reader[self._indices[name]]

    @contextmanager
    def case_lease(self, name):
        self._source(name, 'image')
        self._source(name, 'label')
        geo = self.geometry(name)
        estimate = preprocessing_memory(prod(geo.native_shape_xyz), prod(geo.cache_shape))
        with self.memory.lease(('prepared', name), estimate, lambda: self._build(name)) as entry:
            yield entry

    def _volume(self, name, shape, box, label):
        with self.case_lease(name) as entry:
            region = tuple(slice(a, b) for a, b in _box_in(box, self.geometry(name).cache_shape))
            result = _resize(entry['label' if label else 'image'][region], shape, 0 if label else 1)
            return np.array(result, copy=True, order='C')

    def properties(self, name):
        with self.case_lease(name) as entry:
            return {key: {c:np.array(v, copy=True) for c,v in value.items()} if key=='class_locations'
                    else value for key,value in entry['properties'].items()}

    def _native_volume(self, name, kind):
        self._source(name, 'image')
        self._source(name, 'label')
        estimate = 256*1024**2 + 16*prod(self._records[name]['shape'])
        with self.memory.lease(('native', name), estimate, lambda:self._native(name)) as entry:
            return np.array(entry[kind], copy=True, order='C')

    def read_native_image(self, name):
        return self._native_volume(name, 'image')

    def read_native_label(self, name):
        return self._native_volume(name, 'label')

    def clear(self):
        self.memory.clear()
