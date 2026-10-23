"""Read a completed paired AAOCA preprocessing artifact without importing its code.

Normalized pixels are never inverted into alleged original HU. The producer
must have published an unclipped HU companion on exactly the same grid.
"""
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from .schema import CASE_ID
from .service_state import read_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def contained(root, relative):
    relative = Path(str(relative).replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('预处理清单路径越界')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('预处理文件不存在或链接越界')
    return path


@contextmanager
def source_guard(root):
    """Shared reader lock compatible with the producer's exclusive RunLock."""
    root = Path(root).expanduser().resolve()
    runtime = root.with_name(root.name + '.runtime')
    worker = read_json(runtime / 'worker.json', {})
    if not isinstance(worker, dict) or worker.get('state') in {'submitted', 'checking', 'running'}:
        raise ValueError('预处理后台尚无结束记录，请先查看或恢复预处理任务')
    try:
        stream = (root / '.preprocess.lock').open('rb')
    except OSError as error:
        raise ValueError('需要完整的预处理任务目录，缺少可访问的运行锁') from error
    try:
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBRLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('预处理源目录正被写入，请稍后重试') from error
        try:
            yield root
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def candidates(path):
    """Small directory listing only; invalid neighbors remain selectable for diagnostics."""
    path = Path(path).expanduser().resolve()
    if (path / 'dataset.json').is_file():
        return [path]
    if not path.is_dir():
        raise ValueError('预处理任务或结果根目录不存在')
    return sorted((p.parent for p in path.glob('*/dataset.json')), reverse=True)


def load(root):
    """Validate receipts and freeze pair identities, without hashing image payloads."""
    with source_guard(root) as root:
        dataset = read_json(root / 'dataset.json', {})
        if not isinstance(dataset, dict) or not isinstance(dataset.get('config'), dict):
            raise ValueError('预处理任务配置无效')
        config = dataset.get('config', {})
        if (config.get('hu_companion') != 'resampled_unclipped_HU_v1'
                or config.get('formula') != 'clip(HU, -1000, 1000) / 1000'
                or config.get('input_units') != 'HU'):
            raise ValueError('该预处理结果没有未截断 HU 配套影像；请用1.3或新版从原始HU新建预处理任务，不能将[-1,1]乘1000冒充原始HU')
        progress = read_json(root / 'progress.json', {})
        manifest = read_json(root / 'manifest.json', [])
        if (not isinstance(progress, dict) or progress.get('state') != 'completed' or not isinstance(manifest, list) or not manifest
                or progress.get('total') != len(manifest) or progress.get('failed') or progress.get('pending')):
            raise ValueError('请选择全部完成的预处理任务；尚未完成或有失败病例')
        rows, seen = [], set()
        dataset_hash = digest(root / 'dataset.json')
        for item in manifest:
            if not isinstance(item, dict):
                raise ValueError('预处理清单记录无效')
            key = item.get('case_id', '')
            if not isinstance(key, str) or not CASE_ID.fullmatch(key) or key in seen or item.get('state') not in {'completed', 'reused'}:
                raise ValueError('预处理清单存在重复/非法病例或未完成记录')
            seen.add(key)
            record_path = contained(root, f'cases/{key}/record.json')
            record = read_json(record_path, {})
            relative = f'cases/{key}/image.nii.gz'
            hu_relative = f'cases/{key}/image_hu.nii.gz'
            if (not isinstance(record, dict) or record.get('status') != 'complete' or record.get('case_id') != key
                    or record.get('config') != config or record.get('config_sha256') != dataset.get('config_sha256')
                    or record.get('output') != relative or record.get('hu_output') != hu_relative
                    or item.get('output') != relative or item.get('hu_output') != hu_relative):
                raise ValueError('预处理病例的完成记录、配置或配套路径不一致')
            for field in ('output_sha256', 'hu_output_sha256'):
                value = record.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('预处理病例缺少完整的影像SHA256')
            image = contained(root, hu_relative)
            normalized = contained(root, relative)
            rows.append({'case_id': key, 'image': str(image), 'preprocessing': {
                'root': str(root), 'dataset_sha256': dataset_hash,
                'record': str(record_path), 'record_sha256': digest(record_path),
                'normalized_image': str(normalized), 'normalized_sha256': record['output_sha256'],
                'hu_sha256': record['hu_output_sha256'], 'config_sha256': dataset['config_sha256'],
                'contract': 'resampled_unclipped_HU_v1'}})
        return rows


def validate_case(case, full=True):
    """Recheck frozen receipts and paired geometry; full includes both file hashes."""
    proof = case.get('preprocessing')
    if not proof:
        return
    with source_guard(proof['root']) as root:
        if digest(root / 'dataset.json') != proof['dataset_sha256']:
            raise ValueError('来源预处理任务配置已改变')
        record_path = contained(root, f"cases/{case['case_id']}/record.json")
        if str(record_path) != proof['record'] or digest(record_path) != proof['record_sha256']:
            raise ValueError('来源预处理病例记录已改变')
        image = contained(root, f"cases/{case['case_id']}/image_hu.nii.gz")
        normalized = contained(root, f"cases/{case['case_id']}/image.nii.gz")
        if str(image) != case['image'] or str(normalized) != proof['normalized_image']:
            raise ValueError('冻结的预处理路径不符')
        if full:
            if digest(image) != proof['hu_sha256'] or digest(normalized) != proof['normalized_sha256']:
                raise ValueError('预处理配套影像SHA256校验失败')
        # Numerical imports belong only to the execution environment, never menu queries.
        import nibabel as nib
        import numpy as np
        hu, norm = nib.load(image), nib.load(normalized)
        record = read_json(record_path)
        if (hu.shape != norm.shape or list(hu.shape) != record.get('output_shape')
                or not np.allclose(hu.affine, norm.affine, rtol=0, atol=1e-5)
                or not np.allclose(hu.affine, record['output_affine'], rtol=0, atol=1e-4)
                or hu.header.get_xyzt_units()[0] != 'mm' or norm.header.get_xyzt_units()[0] != 'mm'
                or hu.get_data_dtype() != np.dtype('float32') or norm.get_data_dtype() != np.dtype('float32')
                or b'aaoca_preprocessing HU;' not in bytes(hu.header['descrip'])
                or b'normalized [-1,1]; NOT HU' not in bytes(norm.header['descrip'])):
            raise ValueError('HU影像与归一化影像网格或单位不匹配')
