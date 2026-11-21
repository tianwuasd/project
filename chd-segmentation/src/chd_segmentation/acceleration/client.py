from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4
import os

import psutil

from .contracts import BoundaryDecision, Reservation, ResourceSnapshot
from .accounting import add_claim, remove_claim, available


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
            available(state),
            state['paused'] or state.get('replanning',False), h.cancel.is_set(), state['sampling_seconds'])

    def boundary(self):
        value = self.snapshot()
        if value.cancelled:
            return BoundaryDecision('stop', 'cancelled')
        with self.handle.condition:
            quota_full = available(self.handle.state) <= 0
        if value.paused or quota_full:
            self.wait_started = monotonic() if self.wait_started is None else self.wait_started
            elapsed = monotonic() - self.wait_started
            limit = self.handle.resources.get('max_wait')
            return BoundaryDecision('stop' if limit is not None and elapsed >= limit else 'wait',
                                    'resource_pressure', elapsed)
        self.wait_started = None
        return BoundaryDecision('continue', 'eligible')

    def wait_active(self):
        began = monotonic()
        while True:
            value = self.snapshot()
            if value.cancelled:
                raise InterruptedError('已持有额度的计算已取消')
            if not value.paused:
                return
            limit = self.handle.resources.get('max_wait')
            if limit is not None and monotonic()-began >= limit:
                raise TimeoutError('已持有额度的计算等待达到显式时限')
            if self.handle.cancel.wait(.2):
                raise InterruptedError('已持有额度的计算已取消')

    @contextmanager
    def _pending(self, token, amount, purpose, kind):
        h = self.handle
        def register():
            rows = dict(h.state.get('waiting', {}))
            rows[token] = dict(bytes=amount, purpose=purpose, kind=kind,
                pid=os.getpid(), birth=psutil.Process().create_time())
            h.state['waiting'] = rows
            h.condition.notify_all()
        try:
            yield register
        finally:
            with h.condition:
                rows = dict(h.state.get('waiting', {}))
                rows.pop(token, None)
                h.state['waiting'] = rows
                h.condition.notify_all()

    @contextmanager
    def reserve(self, amount_bytes, *, purpose, wait=True):
        if type(amount_bytes) is not int or amount_bytes < 0:
            raise ValueError('资源申请字节数必须为非负整数')
        h, began, acquired, token = self.handle, monotonic(), False, uuid4().hex
        with self._pending(token, amount_bytes, purpose, 'ordinary') as pending, h.condition:
            while True:
                if h.state.get('monitor_failed'):
                    raise RuntimeError('资源监控异常结束，停止新增分配')
                if h.cancel.is_set():
                    raise InterruptedError('资源等待已取消')
                free = available(h.state,protect_pool=amount_bytes>0)
                if not h.state['paused'] and not h.state.get('replanning',False) and amount_bytes <= free and (amount_bytes > 0 or free > 0):
                    h.state['reserved'] += amount_bytes
                    if amount_bytes:
                        add_claim(h.state,token,amount_bytes,'ordinary')
                    acquired = True
                    break
                limit = h.resources.get('max_wait')
                if limit is not None and monotonic() - began >= limit:
                    raise TimeoutError('已达到显式资源等待时限')
                if not wait:
                    break
                pending()
                self._wait(began)
        try:
            yield Reservation(amount_bytes, purpose) if acquired else None
        finally:
            if acquired:
                with h.condition:
                    h.state['reserved'] -= amount_bytes
                    if amount_bytes:
                        remove_claim(h.state,token)
                    h.condition.notify_all()

    @contextmanager
    def case(self, amount_bytes, *, purpose, wait=True):
        if type(amount_bytes) is not int or amount_bytes < 1:
            raise ValueError('病例工作空间必须为正整数字节数')
        h, began, acquired, token = self.handle, monotonic(), False, uuid4().hex
        with self._pending(token, amount_bytes, purpose, 'case') as pending, h.condition:
            while True:
                if h.state.get('monitor_failed'):
                    raise RuntimeError('资源监控异常结束，停止病例构建')
                if h.cancel.is_set():
                    raise InterruptedError('病例额度等待已取消')
                capacity = h.state.get('workspace_capacity',0)
                if not h.state.get('reader_slots',0):
                    raise RuntimeError('此会话尚未配置病例构建额度池')
                if amount_bytes > capacity:
                    raise MemoryError('单例工作空间超过当前病例额度池')
                if (not h.state['paused'] and not h.state.get('replanning',False) and
                        h.state['active_readers'] < h.state['reader_slots'] and
                        h.state['workspace_reserved'] + amount_bytes <= capacity and
                        amount_bytes <= available(h.state)):
                    h.state['active_readers'] += 1
                    h.state['workspace_reserved'] += amount_bytes
                    add_claim(h.state,token,amount_bytes,'case')
                    acquired = True
                    break
                limit = h.resources.get('max_wait')
                if limit is not None and monotonic()-began >= limit:
                    raise TimeoutError('已达到显式病例额度等待时限')
                if not wait:
                    break
                pending()
                self._wait(began)
        try:
            yield Reservation(amount_bytes,purpose) if acquired else None
        finally:
            if acquired:
                with h.condition:
                    h.state['workspace_reserved'] -= amount_bytes
                    h.state['active_readers'] -= 1
                    remove_claim(h.state,token)
                    h.condition.notify_all()

    def request_replan(self, demand, *, safe_boundary, layout=None):
        h, token, began = self.handle, uuid4().hex, monotonic()
        completed = False
        with h.condition:
            if h.state.get('replan_request') is not None or h.state.get('replanning',False):
                raise RuntimeError('已有资源重规划正在执行')
            if h.state.get('active_readers',0):
                raise RuntimeError('病例构建结束后才能请求重规划')
            h.state['replan_request'] = dict(token=token,demand=asdict(demand),boundary=asdict(safe_boundary),
                                           layout=layout,abandoned=False)
            h.state['replan_result'] = None
            h.state['replanning'] = True
            h.condition.notify_all()
        try:
            with h.condition:
                while True:
                    result = h.state.get('replan_result')
                    if result is not None and result['token']==token:
                        if result.get('failed'):
                            raise RuntimeError('会话所有者重规划失败，请查看所有者异常')
                        completed = True
                        return result['allocation']
                    if h.state.get('monitor_failed'):
                        raise RuntimeError('资源监控异常结束，重规划已停止')
                    if h.cancel.is_set():
                        raise InterruptedError('资源重规划等待已取消')
                    limit = h.resources.get('max_wait')
                    if limit is not None and monotonic()-began >= limit:
                        raise TimeoutError('已达到显式资源重规划等待时限')
                    self._wait(began)
        finally:
            if not completed:
                with h.condition:
                    request = h.state.get('replan_request')
                    if request is not None and request['token']==token:
                        h.state['replan_request'] = dict(request,abandoned=True)
                    h.condition.notify_all()

    def _wait(self,began):
        limit = self.handle.resources.get('max_wait')
        timeout = .2 if limit is None else max(.001,min(.2,limit-(monotonic()-began)))
        self.handle.condition.wait(timeout=timeout)

    def close(self):
        # 进程结束前继续计入RSS，避免关闭一个调用句柄就漏记驻留内存。
        return None
