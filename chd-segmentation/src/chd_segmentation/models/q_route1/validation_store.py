import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...prediction.workspace import PredictionWorkspace
from ...artifacts.store import write_json
from ...observability.events import event


class ValidationWorkspace(PredictionWorkspace):
    def __init__(self, config, identity, block_bytes, guard=None, *, storage='disk', array_budget_bytes=0, rank=0):
        applied = dict(config, work_root=config['work_root'] or str(Path(config['run_dir']).parent))
        super().__init__(applied, dict(task='q-validation', **identity), block_bytes, guard)
        self.slots = set()
        self.storage, self.array_budget_bytes, self.rank = storage, int(array_budget_bytes), rank
        self.started = perf_counter()
        self.identity = identity
        if storage not in ('memory', 'disk'):
            raise ValueError('验证存储方式必须是memory或disk')

    def __enter__(self):
        super().__enter__()
        self.progress('case_start', 0, None)
        return self

    def _array(self, key, shape, dtype, fortran):
        amount = math.prod(shape) * np.dtype(dtype).itemsize
        if self.storage == 'memory':
            if sum(value.nbytes for value in self.maps.values()) + amount > self.array_budget_bytes:
                raise MemoryError('验证数组超过实际已申请的内存额度')
            value = np.zeros(shape, dtype=dtype, order='F' if fortran else 'C')
        else:
            self.check_disk(amount)
            value = np.lib.format.open_memmap(self.path / (key + '.npy'), mode='w+', dtype=dtype,
                                              shape=tuple(shape), fortran_order=fortran)
        self.maps[key] = value
        return value

    def check_disk(self, amount):
        if self.storage == 'disk':
            return super().check_disk(amount)

    def create(self, name, shape, dtype, fortran=False):
        if name in ('logits', 'counts', 'native'):
            if name in self.maps:
                raise ValueError('验证数组名称重复')
            return self._array(name, shape, dtype, fortran)
        if name != 'layer':
            raise ValueError('验证数组名称无效')
        slot = next(i for i in range(64) if 'layer_%02d' % i not in self.maps)
        key = 'layer_%02d' % slot
        value = self._array(key, shape, dtype, fortran)
        self.slots.add(key)
        return value

    def drop(self, array):
        for key, value in list(self.maps.items()):
            if value is array:
                if isinstance(value, np.memmap):
                    value.flush()
                    value._mmap.close()
                    (self.path / (key + '.npy')).unlink()
                del self.maps[key]
                return

    @property
    def compute_block_bytes(self):
        device = torch.device(self.config.get('device', 'cpu'))
        if device.type != 'cuda':
            return self.block_bytes
        free, total = torch.cuda.mem_get_info(device)
        settings = self.config.get('execution', {})
        spare = max(settings.get('vram_reserve_bytes', 1024**3),
                    total * settings.get('vram_reserve_fraction', .1))
        amount = int(max(0, free - spare) / 1.5)
        if amount < 1024**2:
            raise MemoryError('定位器验证缺少最小GPU块显存')
        return min(self.block_bytes, amount)

    def progress(self, phase, completed, total, **fields):
        now = perf_counter()
        if now - self.last_log < self.config['logging']['heartbeat_seconds'] and completed != total:
            return
        record = dict(state='running', rank=self.rank, case_id=self.identity.get('case_id'),
                      phase=phase, completed=completed, total=total, block_bytes=self.block_bytes,
                      storage=self.storage, seconds=now-self.started, started_monotonic=self.started, **fields)
        write_json(Path(self.config['run_dir']) / f'validation-rank{self.rank}.json', record, durable=False)
        event(self.config['run_dir'], 'prediction_blocks', **record)
        self.last_log = now

    def __exit__(self, *error):
        for value in self.maps.values():
            if isinstance(value, np.memmap):
                value.flush()
                value._mmap.close()
        self.maps.clear()
        self._remove_files()
        write_json(Path(self.config['run_dir']) / f'validation-rank{self.rank}.json',
                   dict(state='failed' if error[0] else 'completed', rank=self.rank,
                        case_id=self.identity.get('case_id'), seconds=perf_counter()-self.started), durable=False)
        return False

    def _remove_files(self):
        files = list(self.path.iterdir())
        allowed = {'owner.json', 'logits.npy', 'counts.npy', 'native.npy'}
        allowed.update('layer_%02d.npy' % i for i in range(64))
        if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in files):
            raise ValueError('验证临时目录包含未登记内容')
        for path in files:
            path.unlink()
        self.path.rmdir()
