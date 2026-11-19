import os
from pathlib import Path

import nibabel as nib
import numpy as np

from ..artifacts.store import file_digest, read_json, write_json
from ..observability.events import event


RAW_EXPORT = 'native-hu-author-spacing-mm-v1'


def export_case(reader, row, role, raw, run):
    with_label = role != 'test'
    image_dir = raw / ('imagesTr' if with_label else 'imagesTs')
    image_dir.mkdir(parents=True, exist_ok=True)
    if with_label:
        (raw / 'labelsTr').mkdir(exist_ok=True)
    kinds = ('image', 'label') if with_label else ('image',)
    stamps = {}
    for kind in kinds:
        stat = Path(row[kind]).stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
        if stamp != row['source_stamp'][kind]:
            raise ValueError('冻结的源文件已变化：' + row['case_id'] + ' ' + kind)
        stamps[kind] = stamp
    name = row['case_id']
    targets = dict(image=image_dir / f'{name}_0000.nii.gz')
    if with_label:
        targets['label'] = raw / 'labelsTr' / f'{name}.nii.gz'
    receipt_path = raw / '.completed' / (name + '.json')
    identity = dict(strategy=RAW_EXPORT, role=role, source_stamps=stamps,
                    source_files={kind: row[kind] for kind in kinds}, spacing_xyz=row['spacing_xyz'],
                    geometry_policy=row['geometry_policy'], ignore_label_ids=row.get('ignore_label_ids', []))
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt['input'] != identity:
            raise ValueError('已有病例导出使用了其他来源或规则：' + name)
        if all(path.is_file() and file_digest(path) == receipt['outputs'][kind] for kind, path in targets.items()):
            event(run, 'total_raw_case_reused', case_id=name, role=role)
            return targets
    values, target, native, spacing, roi, _ = reader.read_case(row)
    if any(a != 0 or b != n for (a, b), n in zip(roi, native.shape)):
        raise ValueError('Total HU导出需要完整原始病例')
    affine = native.affine.copy()
    affine[:3, :3] *= spacing / np.linalg.norm(affine[:3, :3], axis=0)
    arrays = dict(image=values)
    if with_label:
        target[target < 0] = 8
        arrays['label'] = target.astype(np.uint8)
    staging = raw / '.export' / name
    staging.mkdir(parents=True, exist_ok=True)
    for kind, path in targets.items():
        exported = nib.Nifti1Image(arrays[kind], affine)
        exported.header.set_xyzt_units('mm')
        pending = staging / f'{name}.{kind}.nii.gz'
        nib.save(exported, pending)
        os.replace(pending, path)
    staging.rmdir()
    write_json(receipt_path, dict(input=identity, outputs={kind: file_digest(path) for kind, path in targets.items()}))
    event(run, 'total_raw_case', case_id=name, role=role)
    return targets
