from pathlib import Path
import time

import numpy as np

from ..contracts import atomic_json, digest, read_json
from .preprocessed import B2ND_FORMAT, PreprocessedInputs


def accept_xu(source_descriptor,compressed_descriptor,cht_code,output):
    started=time.monotonic()
    description=read_json(source_descriptor)
    selected=list(description['storage']['cases'])[:1]
    original=PreprocessedInputs(source_descriptor,cht_code,selected)
    compressed=PreprocessedInputs(compressed_descriptor,cht_code,selected)
    if compressed.prepared['storage']['format']!=B2ND_FORMAT:
        raise ValueError('验收目标必须是B2ND')
    for name in selected:
        a,b=original.geometry(name),compressed.geometry(name)
        if a!=b:
            raise ValueError('压缩前后几何不同')
        for box in ([(0,min(32,n)) for n in a.cache_shape],[(max(0,n-32),n) for n in a.cache_shape]):
            if not np.array_equal(original.read_image(name,box=box),compressed.read_image(name,box=box)):
                raise ValueError('Xu实际读取的压缩区域与NPY不同')
    record=dict(protocol='chd-b2nd-read-acceptance-v1',acceptance_party='XU',passed=True,
        source=str(Path(source_descriptor).resolve()),source_identity=digest(description),
        prepared=str(Path(compressed_descriptor).resolve()),prepared_identity=digest(compressed.prepared),
        recipe_identity=compressed.prepared['storage']['recipe_identity'],cases=selected,
        seconds=time.monotonic()-started,reader='PreprocessedInputs.read_image')
    atomic_json(output,record)
    return record


def require_both(target,ready,paths):
    if len(paths)!=2:
        raise ValueError('删除NPY前必须同时提供CHT与XU的真实读取验收记录')
    parties=set()
    for path in paths:
        record=read_json(path)
        if (record.get('protocol')!='chd-b2nd-read-acceptance-v1' or not record.get('passed')
                or record.get('source_identity')!=ready['conversion']['source_identity']
                or record.get('prepared_identity')!=ready['prepared_identity']
                or Path(record['prepared']).resolve()!=target/'prepared.json'
                or record.get('recipe_identity')!=ready['conversion']['recipe_identity'] or not record.get('cases')):
            raise ValueError('读取验收未通过或对应的转换数据不同')
        parties.add(record['acceptance_party'])
    if parties!={'CHT','XU'}:
        raise ValueError('必须由CHT和XU两个接入入口分别验收')
