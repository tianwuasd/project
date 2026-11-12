from contextlib import contextmanager
import os
from threading import RLock
from time import perf_counter

from cachetools import LRUCache
import numpy as np
import psutil

from .resources import available_memory_bytes
from .xu_resources import HostResourcesChanged, applied_host_budget


def array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(array_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(v) for v in value)
    return 0


class CaseMemory:
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('病例缓存字节上限必须为非负整数')
        self.limit = max_bytes
        self.cache = LRUCache(maxsize=max(1, max_bytes), getsizeof=array_bytes)
        self.lock = RLock()
        self.stats = dict(hits=0, builds=0, evictions=0, build_seconds=0., peak_bytes=0,
                          limit_bytes=max_bytes, disk_bytes=0, memory_denials=0)
        self.owner = os.getpid()

    def clear(self):
        with self.lock:
            self.stats['evictions'] += len(self.cache)
            self.cache.clear()

    def _headroom(self):
        available = available_memory_bytes()
        if available is None:
            raise HostResourcesChanged('在线处理缺少主机/cgroup 内存余量证据')
        live = max(0, int(available-max(4*1024**3, available*.2)))
        stage = applied_host_budget()
        if stage is None and os.environ.get('XU_HOST_BUDGET_BYTES'):
            stage = int(os.environ['XU_HOST_BUDGET_BYTES'])
        if stage is not None:
            live = min(live, max(0, stage-psutil.Process().memory_info().rss))
        return live

    @contextmanager
    def lease(self, key, estimate, build):
        if self.owner != os.getpid():
            raise RuntimeError('在线缓存不能继承到另一个进程')
        # 持有锁期间独占整例准备和借用，避免驱逐在用数组或重复预留峰值。
        with self.lock:
            if key in self.cache:
                self.stats['hits'] += 1
                yield self.cache[key]
                return
            if estimate <= 0:
                raise ValueError('在线处理需要明确的单例内存估算')
            if self._headroom() < estimate:
                self.clear()
            if self._headroom() < estimate:
                self.stats['memory_denials'] += 1
                raise HostResourcesChanged(f'在线准备需要 {estimate} 字节；等待新资源许可')
            started = perf_counter()
            value = build()
            size = array_bytes(value)
            self.stats['builds'] += 1
            self.stats['build_seconds'] += perf_counter()-started
            if size > estimate:
                raise HostResourcesChanged('病例实际数组超过已预留内存')
            cap = min(self.limit, max(0, self._headroom()+size))
            while self.cache and self.cache.currsize+size > cap:
                self.cache.popitem()
                self.stats['evictions'] += 1
            if size <= cap:
                self.cache[key] = value
                self.stats['peak_bytes'] = max(self.stats['peak_bytes'], self.cache.currsize)
            yield value
