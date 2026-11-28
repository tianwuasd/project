from dataclasses import asdict
from pathlib import Path
import os

import torch

from ...prediction.blocks import window_probabilities
from .local_cache import LocalCache
from .validation import validity
from .validation_candidates import bounded_candidates
from .validation_network import LayerInference, localizer_forward
from .validation_store import ValidationWorkspace


class LocalWorkspace(ValidationWorkspace):
    def __init__(self, *args, directory, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = directory/'work'

    def progress(self, phase, completed, total, **fields):
        from ...observability.events import event
        from time import perf_counter
        now = perf_counter()
        if now-self.last_log >= self.config['logging']['heartbeat_seconds'] or completed == total:
            event(self.config['run_dir'], 'q_local_input_blocks', phase=phase, completed=completed,
                  total=total, block_bytes=self.block_bytes, **fields)
            self.last_log = now

    def take(self, array, destination):
        key = next(key for key, value in self.maps.items() if value is array)
        array.flush()
        array._mmap.close()
        del self.maps[key]
        os.replace(self.path/(key+'.npy'), destination)


def frozen_inputs(bundle, entry):
    cache = LocalCache(bundle.training_cache)
    settings = bundle.config['execution']
    block = settings.get('validation_block_bytes', 64*2**20)
    guard = getattr(bundle, 'training_guard', None)
    identity = dict(purpose='q-local-inputs', cache=cache.identity, case=entry['case_id'], pid=os.getpid())
    config = dict(bundle.config, run_dir=str(Path(bundle.config['run_dir'])/'local'),
                  work_root=str(cache.path.parent))

    def build(destination, key):
        with LocalWorkspace(config, dict(identity, key=key), block, guard,
                            directory=destination, storage='disk') as workspace:
            probability = window_probabilities(entry['image'], bundle.base, bundle.spec['patch'], 11,
                dict(overlap=.5, blend='gaussian', window_batch=1), bundle.device, workspace)
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

    return cache.acquire(entry, build)
