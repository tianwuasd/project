from pathlib import Path

import nibabel as nib
import numpy as np

from ...artifacts.store import read_json


def selected_cases(config, names):
    manifest = read_json(config['manifest'])
    if manifest['dataset'] != 'chd1282' or not names or len(set(names)) != len(names):
        raise ValueError('需要不重复的CHD1282显式病例名单')
    selected = {r['case_id']: r for r in manifest['cases'] if r['case_id'] in names}
    if set(selected) != set(names):
        raise ValueError('所选病例未登记于来源清单')
    cohort = read_json(config['cohort'])['cases']
    root = Path(config['data_root']).resolve(strict=True)
    results = []
    for name in names:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('病例编号不能包含目录分隔符')
        row, record = selected[name], cohort[name]
        if not record.get('included', True):
            raise ValueError('所选病例已被冻结划分排除')
        image = (root / row['image']).resolve(strict=True)
        if not image.is_relative_to(root):
            raise ValueError('病例路径越出显式数据根')
        spacing = row['spacing_xyz']
        if (len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing) or
                row.get('spacing_evidence', {}).get('source') != 'author_excel'):
            raise ValueError('缺少逐例作者间距证据')
        results.append(dict(
            case_id=name, image=str(image), spacing_xyz=spacing,
            spacing_evidence=row['spacing_evidence'], role=record['role'], patient=record['patient'],
            unknown_classes=row.get('unknown_classes', []), ignore_label_ids=row.get('ignore_label_ids', [])))
    return results


def open_image(row):
    image = nib.load(row['image'], keep_file_open=True)
    if len(image.shape) != 3:
        raise ValueError('MedSAM3需要三维单通道NIfTI')
    return image


def source_record(row, image, config):
    stat = Path(row['image']).stat()
    return dict(path=row['image'], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
        shape=list(image.shape), dtype=str(image.get_data_dtype()), original_affine=image.affine.tolist(),
        original_header_hex=image.header.binaryblock.hex(),
        research_affine=np.diag([*row['spacing_xyz'], 1.]).tolist(),
        research_origin='first_voxel_center_zero', orientation='RAS_research_assumption',
        orientation_evidence=config['orientation_evidence'], geometry_policy='shared_index_excel',
        spacing_xyz=row['spacing_xyz'], spacing_evidence=row['spacing_evidence'],
        intensity_evidence=config['intensity_evidence'], role=row['role'])


def slices_for(config, name, depth):
    indices = list(range(depth)) if config['coverage'] == 'all' else config['slices'].get(name)
    if (not indices or any(type(k) is not int or not 0 <= k < depth for k in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError('切片覆盖索引缺失、越界或重复')
    return sorted(indices)


def rgb_slice(image, index, config):
    plane = np.asarray(image.dataobj[:, :, index]).T
    if config['intensity_semantics'] == 'windowed_uint8':
        if plane.dtype != np.uint8:
            raise ValueError('windowed_uint8需要真实uint8输入')
        gray = plane
    else:
        values = np.asarray(plane, dtype=np.float32)
        if config['intensity_semantics'] == 'zscore_clipped_hu':
            values = values * config['source_std'] + config['source_mean']
        if not np.isfinite(values).all():
            raise ValueError('CT切片包含非有限值')
        low, high = config['gray_window']
        gray = np.rint((np.clip(values, low, high) - low) * (255. / (high - low))).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)
