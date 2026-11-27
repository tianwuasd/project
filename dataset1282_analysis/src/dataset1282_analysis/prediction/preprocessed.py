from pathlib import Path
import os
import sys

import numpy as np

from ..contracts import digest, read_json
from ..xu_geometry import CaseGeometry, _box_in, _resize


FORMAT='chd-offline-npy-v1'
B2ND_FORMAT='chd-offline-b2nd-v1'
FORMATS=(FORMAT,B2ND_FORMAT)
_OPEN={}


class PreprocessedInputs:
    def __init__(self, prepared, cht_code, names):
        code=Path(cht_code).resolve(strict=True)
        existing_snapshot=code.is_relative_to(Path('/data4/zhangruichen/course/chd-segmentation/code-snapshots')) and (code/'.snapshot.json').is_file()
        if os.name=='posix' and not code.is_relative_to(Path('/data4/zhangruichen/code')) and not existing_snapshot:
            raise ValueError('CHT读取代码必须位于data4/code')
        sys.path.insert(0,str(code/'src'))
        from chd_segmentation.preprocessing.inputs import resolve_preprocessed
        self.path=Path(prepared).resolve(strict=True)
        if os.name=='posix' and not self.path.is_relative_to(
                Path('/data4/zhangruichen/data/chd-segmentation/preprocessed')):
            raise ValueError('预处理描述必须位于正式data4/preprocessed目录')
        self.prepared=read_json(self.path)
        storage=self.prepared['storage']
        if storage['format'] not in FORMATS or not storage['complete']:
            raise ValueError('需要已完成的CHT预处理描述')
        self.source=resolve_preprocessed(self.path,dataset=self.prepared['dataset'],case_ids=names)
        if os.name=='posix' and not self.source.root.is_relative_to(
                Path('/data4/zhangruichen/data/chd-segmentation/preprocessed')):
            raise ValueError('预处理数组必须位于正式data4/preprocessed目录')
        self.plan=self.source.catalog['recipe']['plan']
        self.rows={r['case_id']:r for r in self.prepared['manifest']['cases']}
        self.identity=digest(dict(recipe=storage['recipe_identity'],cases={n:storage['cases'][n] for n in names}))

    def geometry(self,name):
        entry=self.source.open(name,with_label=False)
        props=entry['properties']
        shape=entry['source_shape_xyz']
        forward=self.plan['transpose_forward']
        bbox=props['bbox_used_for_cropping']
        if (list(props['shape_before_cropping']) != [shape[2-i] for i in forward]
                or list(props['shape_after_cropping_and_before_resampling']) != [b-a for a,b in bbox]
                or entry['image'].dtype!=np.float32 or entry['image'].shape[0]!=1):
            raise ValueError('预处理空间属性与实际数组不一致：'+name)
        return CaseGeometry(tuple(shape),tuple(entry['spacing_xyz']),tuple(forward),
            tuple(map(tuple,bbox)),tuple(entry['image'].shape[1:]))

    def read_image(self,name,shape=None,box=None):
        geo=self.geometry(name)
        image=self.source.open(name,with_label=False)['image']
        region=(0,*(slice(a,b) for a,b in _box_in(box,geo.cache_shape)))
        values=np.asarray(image[region])
        if not np.isfinite(values).all():
            raise ValueError('预处理图像包含非有限数值')
        return _resize(values,shape,1)


def from_job(job):
    descriptor=Path(job['preprocessed'])
    stamp=(descriptor.stat().st_mtime_ns,descriptor.stat().st_size)
    key=(str(descriptor),job['cht_code'],job['preprocessed_identity'],stamp)
    if key not in _OPEN:
        _OPEN.clear()
        _OPEN[key]=PreprocessedInputs(descriptor,job['cht_code'],job['cases'])
    source=_OPEN[key]
    if source.identity!=job['preprocessed_identity']:
        raise ValueError('当前数据条目与已登记预测输入不同')
    return source
