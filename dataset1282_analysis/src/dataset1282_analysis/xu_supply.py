"""One speculative CPU batch at most. GPU transfers, augmentation and Adam stay with the consumer."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from time import perf_counter

import torch

from .resources import snapshot


def _budget():
    try:
        return max(0, int(os.environ.get('XU_PREFETCH_BYTES', '0')))
    except ValueError:
        return 0


class BatchSupply:
    def __init__(self, ids, batch_size, side, prepare, source_voxels):
        self.ids = ids
        self.size = batch_size
        self.side = side
        self.prepare = prepare
        self.source_voxels = source_voxels
        self.stats = dict(requested=os.environ.get('XU_PREFETCH') == '1', applied=False,
                          budget_bytes=_budget(), estimated_peak_bytes=0, prefetch_batches=0,
                          minimum_live_budget_bytes=None, memory_denied_batches=0,
                          cpu_prepare_seconds=0., cpu_wait_seconds=0., transfer_call_seconds=0.,
                          pinned_memory=False, non_blocking=False)

    def _reserve(self, batch):
        # One source scan at a time; include decompression/resize workspace, hash buffers,
        # per-case CT and long labels and their stacked copies. Budget is for the extra batch.
        largest = max(self.source_voxels(name) for name in batch)
        if largest <= 0:
            raise ValueError('Expected a positive source voxel count')
        return 64 * 1024**2 + 32 * largest + 48 * len(batch) * self.side**3

    def _prepare(self, batch):
        start = perf_counter()
        with torch.no_grad():
            values = self.prepare(batch)
        return values, perf_counter() - start

    @contextmanager
    def batches(self):
        stats = self.stats
        enabled = stats['requested'] and stats['budget_bytes'] > 0 and self.source_voxels is not None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='xu-cpu') if enabled else None

        def iterate():
            pending = None
            refresh_at = 0.
            live_budget = 0
            for start in range(0, len(self.ids), self.size):
                batch = self.ids[start:start+self.size]
                wait = perf_counter()
                values, elapsed = pending.result() if pending is not None else self._prepare(batch)
                pending = None
                next_batch = self.ids[start+self.size:start+2*self.size]
                if executor is not None and next_batch:
                    if perf_counter() >= refresh_at:
                        available = snapshot().get('available_memory_bytes') or 0
                        live_budget = min(stats['budget_bytes'], max(0, available//4))
                        previous = stats['minimum_live_budget_bytes']
                        stats['minimum_live_budget_bytes'] = live_budget if previous is None else min(live_budget, previous)
                        refresh_at = perf_counter() + 30.
                    estimate = self._reserve(next_batch)
                    stats['estimated_peak_bytes'] = max(stats['estimated_peak_bytes'], estimate)
                    if estimate <= live_budget:
                        pending = executor.submit(self._prepare, next_batch)
                        stats['prefetch_batches'] += 1
                        stats['applied'] = True
                    else:
                        stats['memory_denied_batches'] += 1
                stats['cpu_prepare_seconds'] += elapsed
                stats['cpu_wait_seconds'] += perf_counter() - wait
                yield batch, values
                del values
        batches = iterate()
        try:
            yield batches
        finally:
            batches.close()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    def transfer(self, pair, device):
        start = perf_counter()
        result = tuple(t.to(device) for t in pair)
        # Wall time spent in the blocking .to() calls, not measured CUDA kernel duration.
        self.stats['transfer_call_seconds'] += perf_counter() - start
        return result


def check_gradients(model):
    # All Xu parameters share one device. Reduce scalar flags there before the host read.
    flags = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not flags or not torch.stack(flags).all().item():
        raise FloatingPointError('Missing or non-finite branch gradient')
