from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import threading
from time import perf_counter

import torch

from .resources import snapshot


def _budget():
    value = int(os.environ.get('XU_PREFETCH_BYTES', '0'))
    if value < 0:
        raise ValueError('预取内存预算必须非负')
    return value


class BatchSupply:
    def __init__(self, ids, batch_size, side, prepare, source_voxels):
        self.ids = ids
        self.size = batch_size
        self.side = side
        self.prepare = prepare
        self.source_voxels = source_voxels
        self.workers = int(os.environ.get('XU_PREFETCH_WORKERS', '1'))
        if self.workers < 1 or batch_size < 1:
            raise ValueError('供数 worker 和 batch 必须为正整数')
        self.copies = deque()
        self.copy_stream = None
        self.stats = dict(requested=os.environ.get('XU_PREFETCH') == '1', applied=False,
                          budget_bytes=_budget(), estimated_peak_bytes=0, prefetch_batches=0,
                          minimum_live_budget_bytes=None, memory_denied_batches=0,
                          workers_requested=self.workers, workers_applied=0, queue_peak=0,
                          reserved_peak_bytes=0, pin_seconds=0., pinned_peak_bytes=0,
                          cpu_prepare_seconds=0., cpu_wait_seconds=0., transfer_call_seconds=0.,
                          pinned_memory=False, non_blocking=False)
        self.stats_lock = threading.Lock()
        self.worker_names = set()

    def _reserve(self, batch):
        # 包含每个并发批次的解压、插值和输入数组工作内存。
        largest = max(self.source_voxels(name) for name in batch)
        if largest <= 0:
            raise ValueError('Expected a positive source voxel count')
        return 64 * 1024**2 + 32 * largest + 48 * len(batch) * self.side**3

    def _prepare(self, batch):
        start = perf_counter()
        with torch.no_grad():
            values = self.prepare(batch)
        with self.stats_lock:
            self.worker_names.add(threading.current_thread().name)
            self.stats['cpu_prepare_seconds'] += perf_counter()-start
            self.stats['workers_applied'] = len([n for n in self.worker_names if n.startswith('xu-cpu')])
        return values

    @contextmanager
    def batches(self):
        stats = self.stats
        requested = stats['requested']
        if requested and (stats['budget_bytes'] <= 0 or self.source_voxels is None):
            raise ValueError('启用预取需要明确的内存预算和源病例尺寸')
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='xu-cpu') if requested else None

        def iterate():
            batches = [self.ids[i:i+self.size] for i in range(0,len(self.ids),self.size)]
            if executor is None:
                for batch in batches:
                    started = perf_counter()
                    values = self._prepare(batch)
                    stats['cpu_wait_seconds'] += perf_counter()-started
                    yield batch, values
                return
            pending = deque()
            next_index = reserved = 0
            refreshed = 0.
            live_budget = 0
            estimates = {}

            def fill():
                nonlocal next_index, reserved, refreshed, live_budget
                if perf_counter() >= refreshed:
                    available = snapshot().get('available_memory_bytes')
                    if available is None:
                        raise RuntimeError('无法核实预取的实时主机内存')
                    live_budget = min(stats['budget_bytes'],max(0,available-max(4*1024**3,available//5)))
                    old = stats['minimum_live_budget_bytes']
                    stats['minimum_live_budget_bytes'] = live_budget if old is None else min(old,live_budget)
                    refreshed = perf_counter()+1
                while next_index < len(batches) and len(pending) < 2*self.workers:
                    if next_index not in estimates:
                        estimates[next_index] = self._reserve(batches[next_index])
                    estimate = estimates[next_index]
                    stats['estimated_peak_bytes'] = max(stats['estimated_peak_bytes'], estimate)
                    if reserved+estimate > live_budget:
                        stats['memory_denied_batches'] += 1
                        if not pending:
                            raise MemoryError('预取预算不足以准备下一批；保留已有检查点')
                        break
                    future = executor.submit(self._prepare,batches[next_index])
                    pending.append((next_index,estimate,future))
                    next_index += 1
                    reserved += estimate
                    stats['prefetch_batches'] += 1
                    stats['applied'] = True
                    stats['queue_peak'] = max(stats['queue_peak'],len(pending))
                    stats['reserved_peak_bytes'] = max(stats['reserved_peak_bytes'],reserved)

            fill()
            while pending:
                index, estimate, future = pending.popleft()
                started = perf_counter()
                values = future.result()
                stats['cpu_wait_seconds'] += perf_counter()-started
                reserved -= estimate
                # 当前批次已经准备完成；仅派发后续工作，不等待后续计算。
                fill()
                yield batches[index],values
        batches = iterate()
        try:
            yield batches
        finally:
            batches.close()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            for event, _ in self.copies:
                event.synchronize()
            self.copies.clear()
            self.copy_stream = None

    def transfer(self, pair, device):
        start = perf_counter()
        target = torch.device(device)
        if target.type == 'cuda' and self.stats['requested']:
            while self.copies and self.copies[0][0].query():
                self.copies.popleft()
            if len(self.copies) >= 2:
                event, retained = self.copies.popleft()
                event.synchronize()
                del retained
            pin_start = perf_counter()
            host = tuple(t.pin_memory() for t in pair)
            self.stats['pin_seconds'] += perf_counter()-pin_start
            self.stats['pinned_memory'] = self.stats['non_blocking'] = True
            if self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(device=target)
            with torch.cuda.stream(self.copy_stream):
                result = tuple(t.to(target,non_blocking=True) for t in host)
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
            current = torch.cuda.current_stream(target)
            current.wait_event(ready)
            for tensor in result:
                tensor.record_stream(current)
            self.copies.append((ready,host))
            pinned = sum(t.numel()*t.element_size() for _, row in self.copies for t in row)
            self.stats['pinned_peak_bytes'] = max(self.stats['pinned_peak_bytes'],pinned)
        else:
            result = tuple(t.to(target) for t in pair)
        self.stats['transfer_call_seconds'] += perf_counter() - start
        return result


def check_gradients(model):
    flags = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not flags or not torch.stack(flags).all().item():
        raise FloatingPointError('Missing or non-finite branch gradient')
