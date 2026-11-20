from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from time import perf_counter

import nibabel as nib
import numpy as np

from ...artifacts.store import read_json, write_json
from ...observability.events import event
from ..stunet.streaming import local_image
from . import PROTOCOL


def atomic_array(path, values):
    pending = path.with_name(path.name + '.pending')
    with pending.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


class ImageWorkspace:
    def __init__(self, root, guard, log_root):
        self.path = root
        self.guard = guard
        self.log_root = log_root
        self.streams = []
        self.config = {'technical': {'roi_xyz': None}}
        self.last_log = 0.

    @contextmanager
    def allocation(self, amount, label):
        with self.guard.admit(amount, label=label) as ticket:
            if ticket is None:
                raise InterruptedError('读片资源等待被取消')
            yield

    def report(self, phase, count):
        now = perf_counter()
        if now - self.last_log >= 15:
            event(self.log_root, 'medsam3_decompress', phase=phase, blocks=count)
            self.last_log = now


def mapped_image(image, source, cache, identity, guard, log_root, name='image'):
    from ...artifacts.store import require_output
    cache = require_output(cache, log_root)
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / (name + '-source.json')
    target = cache / (name + '.nii')
    if Path(source).suffix.lower() != '.gz':
        return image
    if marker.exists():
        if read_json(marker) != identity or not target.is_file() or target.is_symlink():
            raise ValueError('映射缓存来源改变或文件丢失')
        return nib.load(target, mmap='r', keep_file_open=True)
    # 缓存目录由本任务登记；无完成标志的解压文件可以重新生成。
    if target.exists():
        if target.is_symlink():
            raise ValueError('映射缓存不能是链接')
        target.unlink()
    amount = int(np.prod(image.shape)) * image.get_data_dtype().itemsize + 16 * 2**20
    if shutil.disk_usage(cache).free < amount:
        raise OSError('一次解压所需磁盘空间不足')
    result = local_image(image, Path(source), name, ImageWorkspace(cache, guard, log_root))
    write_json(marker, identity)
    return result


def export_case(root, original, state):
    destination = root / 'prediction.nii.gz'
    pending = root / 'prediction.pending.nii.gz'
    mapping = root / 'export.npy'
    values = np.lib.format.open_memmap(mapping, mode='w+', dtype=np.uint8, shape=original.shape)
    try:
        values[:] = 255
        for index in state['completed_slices']:
            values[:, :, int(index)] = np.load(root / f'slice-{index}.npy', allow_pickle=False)
        values.flush()
        header = original.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(1., 0.)
        result = original.__class__(values, original.affine, header)
        nib.save(result, pending)
        with pending.open('rb') as stream:
            os.fsync(stream.fileno())
        pending.replace(destination)
        del result
    finally:
        values._mmap.close()
        mapping.unlink()
    state['output'] = str(destination)
    state['uncomputed_label'] = 255
    state['protocol'] = PROTOCOL
    state['complete_volume'] = len(state['completed_slices']) == original.shape[2]
    write_json(root / 'case.json', state)
    return destination
