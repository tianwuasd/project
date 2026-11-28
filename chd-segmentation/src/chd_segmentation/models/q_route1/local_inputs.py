from dataclasses import asdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter
import os

import numpy as np
import torch

from ...acceleration.accounting import available
from ...acceleration.host import host_capacity
from ...prediction.blocks import window_probabilities
from .local_cache import LocalCache, disk_plan
from .preparation import preparation
from .validation import validity
from .validation_candidates import bounded_candidates
from .validation_network import LayerInference, localizer_forward
from .validation_store import ValidationWorkspace


class LocalWorkspace(ValidationWorkspace):
    def __init__(self, *args, directory, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = directory/'work'
        self.buffered_bytes = 0
        self.flush_seconds = 0.

    def release(self, *extra, force=False):
        self.buffered_bytes += self.block_bytes
        if self.storage == 'disk' and (force or self.buffered_bytes >= 4*self.block_bytes):
            began = perf_counter()
            super().release(*extra)
            self.flush_seconds += perf_counter()-began
            self.buffered_bytes = 0

    @contextmanager
    def boundary(self):
        if self.guard is not None:
            if self.guard.client.boundary().action == 'wait':
                self.release(force=True)
            with self.guard.admit(0, label='q_local_block'):
                yield
        else:
            yield

    def progress(self, phase, completed, total, **fields):
        from ...observability.events import event
        from time import perf_counter
        now = perf_counter()
        if now-self.last_log >= self.config['logging']['heartbeat_seconds'] or completed == total:
            event(self.config['run_dir'], 'q_local_input_blocks', phase=phase, completed=completed,
                  total=total, case_id=self.identity['case'], storage=self.storage,
                  seconds=now-self.started, flush_seconds=self.flush_seconds,
                  cpu_threads=torch.get_num_threads(), block_bytes=self.block_bytes, **fields)
            self.last_log = now

    def take(self, array, destination):
        key = next(key for key, value in self.maps.items() if value is array)
        if isinstance(array, np.memmap):
            array.flush()
            array._mmap.close()
            os.replace(self.path/(key+'.npy'), destination)
        else:
            self.check_disk(array.nbytes)
            np.save(destination, array, allow_pickle=False)
        del self.maps[key]


@contextmanager
def local_resources(bundle, entry, guard):
    block = bundle.config['execution'].get('validation_block_bytes', 64*2**20)
    arrays = disk_plan(entry['image'].shape[1:], bundle.spec['patch'][0])['peak_bytes']
    storage, amount = 'disk', 0
    if guard is not None:
        h = guard.client.handle
        with h.condition:
            state = dict(h.state)
        world = max(1, state['world'])
        free = available(state, protect_pool=True)
        extra = min(state['project_limit_bytes']-state['limited_project_bytes'],
                    state['system_headroom_bytes']-state['pending_bytes'])
        potential = max(0, free+extra)//world
        block = max(block, min(512*2**20, potential//16))
        if arrays+8*block+256*2**20 <= potential:
            storage = 'memory'
        amount = (arrays if storage == 'memory' else 0)+8*block
    claim = guard.admit(amount, label='q_local_frozen_workspace') if guard is not None else nullcontext(True)
    with claim as ticket, preparation(bundle.config, guard, entry, stage='local') as runtime:
        if ticket is None:
            raise InterruptedError('局部输入工作区等待已取消')
        previous = torch.get_num_threads()
        prefetch = False
        if runtime is not None:
            limit = runtime.executor.settings['max_case_threads'] or min(8, host_capacity()['cores'])
            threads = runtime.executor.cpu.request(limit)
            prefetch = threads>1
            torch.set_num_threads(max(1, threads-int(prefetch)))
        try:
            yield storage, block, arrays, prefetch
        finally:
            torch.set_num_threads(previous)


def frozen_inputs(bundle, entry):
    cache = LocalCache(bundle.training_cache)
    guard = getattr(bundle, 'training_guard', None)
    identity = dict(purpose='q-local-inputs', cache=cache.identity, case=entry['case_id'], pid=os.getpid())
    config = dict(bundle.config, run_dir=str(Path(bundle.config['run_dir'])/'local'),
                  work_root=str(cache.path.parent))

    def build(destination, key):
        with local_resources(bundle, entry, guard) as (storage, block, arrays, prefetch), LocalWorkspace(
                config, dict(identity, key=key), block, guard, directory=destination,
                storage=storage, array_budget_bytes=arrays) as workspace:
            probability = window_probabilities(entry['image'], bundle.base, bundle.spec['patch'], 11,
                dict(overlap=.5, blend='gaussian', window_batch=1), bundle.device, workspace,
                prefetch=prefetch)
            shape = list(entry['image'].shape[1:])
            padded = list(workspace.maps['logits'].shape[1:])
            before = [(b-a)//2 for a, b in zip(shape, padded)]
            workspace.drop(workspace.maps['counts'])
            valid = validity(entry, workspace, False)
            logits, features = localizer_forward(bundle.locator, entry['image'], valid, bundle.device, workspace)
            workspace.drop(valid)
            heat = LayerInference(workspace, bundle.device).unary(logits, torch.sigmoid)
            workspace.drop(logits)
            candidates = bounded_candidates(heat, shape, workspace, (4, 4, 8), .2, .8)
            workspace.drop(heat)
            record = dict(shape=shape, before=before,
                          shapes=dict(probability=[11, *padded], features=list(features.shape)),
                          candidates=[asdict(value) for value in candidates])
            del probability
            workspace.take(workspace.maps['logits'], destination/'probability.npy')
            workspace.take(features, destination/'features.npy')
            workspace.release(entry['image'])
            return record

    return cache.acquire(entry, build, guard)
