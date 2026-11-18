from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

from .contracts import read_json


def source_geometry(item, case):
    prepared = read_json(item['prepared'])
    root = Path(prepared['storage']['root']).resolve(strict=True)
    pointer = prepared['storage']['cases'][case]
    record_path = root / pointer['key'][:16] / 'case.json'
    entry = read_json(record_path)['entry']
    if entry['case_id'] != case:
        raise ValueError('实际预处理记录病例不符')
    catalog_path = root/'catalog.json'
    recipe = read_json(catalog_path)['recipe']
    if recipe['transform'].startswith('medsam2-native'):
        axes, processed_spacing = [0,1,2], entry['spacing_xyz'][::-1]
    else:
        plan = recipe['plan']
        axes = plan['transpose_forward']
        processed_spacing = plan['configurations']['3d_fullres']['spacing']
    return dict(record=str(record_path), catalog=str(catalog_path), roi_xyz=entry['roi_xyz'],
                axis_order=axes, processed_spacing=processed_spacing, transform=recipe['transform'],
                spacing_xyz=entry['spacing_xyz'], source_shape_xyz=entry['source_shape_xyz'])


def verify_output(source, output, metadata):
    started = perf_counter()
    original = nib.load(str(source))
    prediction = nib.load(str(output), keep_file_open=True)
    if len(prediction.shape) != 3 or original.shape[:3] != prediction.shape:
        raise ValueError('预测与真实原图尺寸不符')
    if list(prediction.shape) != metadata['source_shape_xyz'] or not np.allclose(
            original.affine,prediction.affine,atol=1e-5,rtol=0):
        raise ValueError('预测原图affine或来源尺寸不符')
    roi = metadata['roi_xyz']
    if len(roi) != 3 or any(len(pair)!=2 or any(type(v) is not int for v in pair) or
                            not 0 <= pair[0] < pair[1] <= prediction.shape[i] for i,pair in enumerate(roi)):
        raise ValueError('来源ROI无效')
    nonzero, peak = 0, 0
    (x0,x1),(y0,y1),(z0,z1) = roi
    # 只顺序读取预测标签的z切片，原始CT仅加载头部。
    for z in range(prediction.shape[2]):
        slab = np.asanyarray(prediction.dataobj[:,:,z])
        peak = max(peak,slab.nbytes)
        if z < z0 or z >= z1:
            nonzero += int(np.count_nonzero(slab))
        else:
            nonzero += sum(int(np.count_nonzero(region)) for region in
                           (slab[:x0,:],slab[x1:,:],slab[x0:x1,:y0],slab[x0:x1,y1:]))
    result = dict(original=str(source), prediction=str(output), shape=list(prediction.shape),
        affine=prediction.affine.tolist(),header_dtype=str(prediction.get_data_dtype()),
        observed_dtype=str(slab.dtype), roi_outside_nonzero=nonzero,
        max_prediction_slice_bytes=peak, metadata=metadata, seconds=perf_counter()-started,
        scope='原图头与ROI外零填充；实际计算仅ROI，不证明全体积预测效果')
    if nonzero:
        raise ValueError('预测ROI外存在非零标签：'+str(nonzero))
    return result
