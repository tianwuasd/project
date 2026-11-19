from pathlib import Path

import numpy as np

from ...prediction.workspace import PredictionWorkspace


class ValidationWorkspace(PredictionWorkspace):
    def __init__(self, config, identity, block_bytes, guard=None):
        applied = dict(config, work_root=config['work_root'] or str(Path(config['run_dir']).parent))
        super().__init__(applied, dict(task='q-validation', **identity), block_bytes, guard)
        self.slots = set()

    def create(self, name, shape, dtype, fortran=False):
        if name in ('logits', 'counts', 'native'):
            return super().create(name, shape, dtype, fortran)
        if name != 'layer':
            raise ValueError('验证数组名称无效')
        slot = next(i for i in range(64) if 'layer_%02d' % i not in self.maps)
        key = 'layer_%02d' % slot
        self.check_disk(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        value = np.lib.format.open_memmap(self.path / (key + '.npy'), mode='w+',
                                         dtype=dtype, shape=tuple(shape))
        self.maps[key] = value
        self.slots.add(key)
        return value

    def drop(self, array):
        for key, value in list(self.maps.items()):
            if value is array:
                value.flush()
                value._mmap.close()
                del self.maps[key]
                return

    def _remove_files(self):
        files = list(self.path.iterdir())
        allowed = {'owner.json', 'logits.npy', 'counts.npy', 'native.npy'}
        allowed.update('layer_%02d.npy' % i for i in range(64))
        if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in files):
            raise ValueError('验证临时目录包含未登记内容')
        for path in files:
            path.unlink()
        self.path.rmdir()
