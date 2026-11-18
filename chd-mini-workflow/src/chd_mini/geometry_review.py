from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

from .contracts import read_json


def source_geometry(item, case):
    from chd_segmentation.preprocessing.inputs import resolve_preprocessed
    prepared = read_json(item['prepared'])
    storage = prepared['storage']
    source = resolve_preprocessed(storage['root'],dataset=prepared['dataset'],
                                  case_ids=[case],storage=storage)
    entry = source.open(case,with_label=False)
    fields = ('roi_xyz','axis_order','processed_spacing','source_plan','source_directory',
              'source_shape_xyz','spacing_xyz','transform')
    metadata = {key:entry[key] for key in fields}
    metadata.update(prepared=str(Path(item['prepared']).resolve()),case_id=case,
                    recipe_identity=storage['recipe_identity'],selected_pointer=storage['cases'][case])
    # API按需映射所选数组；本函数只取元数据，不读取影像体素或标签。
    del entry
    return metadata


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
