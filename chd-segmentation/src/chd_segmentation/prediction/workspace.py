from contextlib import contextmanager
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from ..artifacts.cache import release_private_mapping
from ..artifacts.store import digest, read_json, require_output, write_json
from ..observability.events import event


class PredictionWorkspace:
    def __init__(self, config, identity, block_bytes, guard=None):
        self.config, self.guard, self.block_bytes = config, guard, int(block_bytes)
        self.path = require_output(Path(config['work_root'])/'prediction-work'/digest(identity)[:24], config['work_root'])
        self.owner = dict(protocol='prediction-blocks-v1', identity=identity)
        self.maps, self.last_log = {}, 0.

    def __enter__(self):
        if self.path.exists():
            if self.path.is_symlink() or read_json(self.path/'owner.json') != self.owner:
                raise ValueError('预测临时目录归属不符')
            self._remove_files()
        self.path.mkdir(parents=True, exist_ok=True)
        write_json(self.path/'owner.json', self.owner)
        return self

    def _remove_files(self):
        allowed = {'owner.json', 'logits.npy', 'counts.npy', 'native.npy', 'prediction.nii', 'reference.nii'}
        files = list(self.path.iterdir())
        if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in files):
            raise ValueError('预测临时目录存在未登记文件，停止清理')
        for path in files:
            path.unlink()
        self.path.rmdir()

    def check_disk(self, amount):
        available = shutil.disk_usage(self.path).free
        if available < amount + 1024**3:
            raise OSError(f'预测分块临时磁盘不足：需要{amount}字节，剩余{available}字节')

    def create(self, name, shape, dtype, fortran=False):
        if name not in ('logits', 'counts', 'native') or name in self.maps:
            raise ValueError('预测临时数组名称重复或无效')
        value = np.lib.format.open_memmap(self.path/(name+'.npy'), mode='w+', dtype=dtype,
                                         shape=tuple(shape), fortran_order=fortran)
        self.maps[name] = value
        return value

    def release(self, *extra):
        for array in (*self.maps.values(), *extra):
            if isinstance(array, np.memmap):
                release_private_mapping(array)

    @contextmanager
    def boundary(self):
        if self.guard is None:
            yield
        else:
            with self.guard.admit(0, label='prediction_block', release=self.release) as ready:
                if ready is None:
                    raise InterruptedError('分块预测等待已取消')
                yield

    def progress(self, phase, completed, total, **fields):
        now = perf_counter()
        if now-self.last_log >= self.config['logging']['heartbeat_seconds'] or completed == total:
            event(self.config['run_dir'], 'prediction_blocks', phase=phase, completed=completed, total=total,
                  block_bytes=self.block_bytes, **fields)
            self.last_log = now

    def __exit__(self, *_):
        for array in self.maps.values():
            array.flush()
            array._mmap.close()
        self.maps.clear()
        self._remove_files()
        return False
