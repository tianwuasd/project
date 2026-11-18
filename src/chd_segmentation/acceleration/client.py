from contextlib import contextmanager
from datetime import datetime, timezone
from time import monotonic

from .contracts import BoundaryDecision, Reservation, ResourceSnapshot


class ResourceClient:
    def __init__(self, handle):
        self.handle, self.wait_started = handle, None

    def snapshot(self):
        h = self.handle
        with h.condition:
            state = dict(h.state)
        if state.get('monitor_failed'):
            raise RuntimeError('资源监控异常结束，停止新增分配')
        return ResourceSnapshot(state['sampled_at'], state['sampled_monotonic'],
            monotonic() - state['sampled_monotonic'], state['host'], state['project_bytes'],
            max(0,min(state['headroom'],state['capacity']-state.get('fixed_rss',0)-state['reserved'])),
            state['paused'], h.cancel.is_set(), state['sampling_seconds'])

    def boundary(self):
        value = self.snapshot()
        if value.cancelled:
            return BoundaryDecision('stop', 'cancelled')
        with self.handle.condition:
            quota_full = self.handle.state['reserved'] >= self.handle.state['capacity']-self.handle.state.get('fixed_rss',0)
        if value.paused or quota_full:
            self.wait_started = monotonic() if self.wait_started is None else self.wait_started
            elapsed = monotonic() - self.wait_started
            limit = self.handle.resources.get('max_wait')
            return BoundaryDecision('stop' if limit is not None and elapsed >= limit else 'wait',
                                    'resource_pressure', elapsed)
        self.wait_started = None
        return BoundaryDecision('continue', 'eligible')

    @contextmanager
    def reserve(self, amount_bytes, *, purpose, wait=True):
        if type(amount_bytes) is not int or amount_bytes < 0:
            raise ValueError('资源申请字节数必须为非负整数')
        h, began, acquired = self.handle, monotonic(), False
        with h.condition:
            while True:
                if h.state.get('monitor_failed'):
                    raise RuntimeError('资源监控异常结束，停止新增分配')
                if h.cancel.is_set():
                    raise InterruptedError('资源等待已取消')
                free = h.state['capacity']-h.state.get('fixed_rss',0)-h.state['reserved']
                if not h.state['paused'] and amount_bytes <= free and (amount_bytes > 0 or free > 0):
                    h.state['reserved'] += amount_bytes
                    acquired = True
                    break
                limit = h.resources.get('max_wait')
                if limit is not None and monotonic() - began >= limit:
                    raise TimeoutError('已达到显式资源等待时限')
                if not wait:
                    break
                h.condition.wait(timeout=1.)
        try:
            yield Reservation(amount_bytes, purpose) if acquired else None
        finally:
            if acquired:
                with h.condition:
                    h.state['reserved'] -= amount_bytes
                    h.condition.notify_all()

    def close(self):
        # 进程结束前继续计入RSS，避免关闭一个调用句柄就漏记驻留内存。
        return None
