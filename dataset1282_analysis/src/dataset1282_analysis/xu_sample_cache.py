import os
import json
from contextlib import contextmanager, ExitStack
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4

import numpy as np

from .contracts import atomic_json, digest, read_json, sha256_file
from .artifact_retention import cache_lock_home
from .observability.status import operation_lock, lock_active


_ROOT_LOCKS = {}
_ENTRY_LOCKS = {}
_ROOT_LOCKS_GUARD = threading.Lock()


class ArrayCache:
    def __init__(self, root, max_bytes=None):
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root:
            raise ValueError('输入缓存不接受符号链接')
        self.max_bytes = int(os.environ.get('XU_INPUT_CACHE_MAX_BYTES', 8*1024**3)
                             if max_bytes is None else max_bytes)
        if self.max_bytes < 0:
            raise ValueError('输入缓存容量必须为非负整数')
        self.root.mkdir(parents=True,exist_ok=True)
        with _ROOT_LOCKS_GUARD:
            self.root_lock = _ROOT_LOCKS.setdefault(str(self.root), threading.RLock())
        self.lock = threading.Lock()
        self.entries = {}
        self.verified = {}
        self.stats = dict(hits=0,created=0,read_seconds=0.,build_seconds=0.,
                          max_bytes=self.max_bytes, oversized_uncached=0,
                          busy_uncached=0,
                          evicted_arrays=0, evicted_bytes=0)

    def get(self, identity, build):
        key = digest(identity)
        with _ROOT_LOCKS_GUARD:
            entry_lock = _ENTRY_LOCKS.setdefault((str(self.root),key),threading.Lock())
        with entry_lock, ExitStack() as lease:
            # 仅在登记租约和分配容量期间使用共享锁，读取和生成数组可并行。
            with self._budget():
                reservations = self._reservations()
                self._evict(self.max_bytes,reservations)
                lease.enter_context(operation_lock(self._lease_home(key), 'input-cache-entry'))
            return self._get(identity, build)

    def _lease_home(self, key):
        return cache_lock_home(self.root)/'leases'/key[:32]

    @contextmanager
    def _budget(self):
        with self.root_lock, operation_lock(cache_lock_home(self.root), 'input-cache-budget'):
            if self.root.resolve() != self.root:
                raise ValueError('输入缓存不接受符号链接')
            self.root.mkdir(parents=True, exist_ok=True)
            yield

    def _reservations(self):
        path = cache_lock_home(self.root)/'reservations.json'
        rows = read_json(path) if path.exists() else {}
        stale = [key for key in rows
                 if not lock_active(self._lease_home(key)/'.runtime/operation.lock')]
        for key in stale:
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('输入缓存容量记录无效')
            for target in self.root.glob(key+'.*'):
                if target.resolve() != target or not target.is_file():
                    raise ValueError('输入缓存不接受符号链接')
                target.unlink()
            del rows[key]
        if stale:
            atomic_json(path,rows)
        return rows

    def _evict(self, limit, reservations):
        paths = [path for path in self.root.iterdir() if path.name.split('.')[0] not in reservations]
        if any(path.resolve() != path or not path.is_file()
               or path.suffix not in ('.npy', '.json', '.pending') for path in paths):
            raise ValueError('输入缓存包含未知文件')
        sizes = {path: path.stat().st_size for path in paths}
        total = sum(size for path,size in sizes.items() if path.name.split('.')[0] not in reservations)
        total += sum(reservations.values())
        arrays = sorted((path for path in paths if path.suffix == '.npy'),
                        key=lambda path: (path.with_suffix('.json') if
                            path.with_suffix('.json').exists() else path).stat().st_mtime_ns)
        for path in arrays:
            if total <= limit:
                break
            if lock_active(self._lease_home(path.stem)/'.runtime/operation.lock'):
                continue
            for target in (path.with_suffix('.json'), path):
                if target in sizes:
                    target.unlink()
                    total -= sizes[target]
                    self.stats['evicted_bytes'] += sizes[target]
            self.verified.pop(path.stem, None)
            self.stats['evicted_arrays'] += 1
        return total

    def _get(self, identity, build):
        key = digest(identity)
        with self.lock:
            entry_lock = self.entries.setdefault(key,threading.Lock())
        with entry_lock:
            path = self.root/(key+'.npy')
            receipt = path.with_suffix('.json')
            if path.is_symlink() or receipt.is_symlink():
                raise ValueError('输入缓存不接受符号链接')
            if receipt.exists():
                started = time.perf_counter()
                record = read_json(receipt)
                if record['identity'] != identity or record['schema'] != 1:
                    raise ValueError('输入缓存来源不一致')
                stat = path.stat()
                stamp = (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,record['sha256'])
                if self.verified.get(key) != stamp:
                    if sha256_file(path) != record['sha256']:
                        raise ValueError('输入缓存内容校验失败')
                    self.verified[key] = stamp
                array = np.load(path,allow_pickle=False)
                if list(array.shape) != record['shape'] or array.dtype.str != record['dtype']:
                    raise ValueError('输入缓存形状或类型不一致')
                with self.lock:
                    self.stats['hits'] += 1
                    self.stats['read_seconds'] += time.perf_counter()-started
                os.utime(receipt, None)
                return array
            started = time.perf_counter()
            array = np.ascontiguousarray(build())
            if array.dtype.hasobject or array.ndim != 3:
                raise ValueError('输入缓存只接受三维数值数组')
            allocation = array.nbytes+4096+2*len(json.dumps(identity,ensure_ascii=False,default=str).encode('utf-8'))
            if allocation > self.max_bytes:
                with self.lock:
                    self.stats['created'] += 1
                    self.stats['build_seconds'] += time.perf_counter()-started
                    self.stats['oversized_uncached'] += 1
                return array
            with self._budget():
                reservations = self._reservations()
                used = self._evict(self.max_bytes-allocation,reservations)
                if used+allocation > self.max_bytes:
                    with self.lock:
                        self.stats['created'] += 1
                        self.stats['build_seconds'] += time.perf_counter()-started
                        self.stats['busy_uncached'] += 1
                    return array
                reservations[key] = allocation
                atomic_json(cache_lock_home(self.root)/'reservations.json',reservations)
            if shutil.disk_usage(self.root).free < array.nbytes+256*1024**2:
                raise OSError('输入缓存磁盘空间不足')
            pending = self.root/(key+'.'+uuid4().hex+'.pending')
            with pending.open('xb') as stream:
                np.save(stream,array,allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending,path)
            record = dict(schema=1,identity=identity,sha256=sha256_file(path),
                          shape=list(array.shape),dtype=array.dtype.str)
            with self._budget():
                atomic_json(receipt,record)
                reservations = self._reservations()
                del reservations[key]
                atomic_json(cache_lock_home(self.root)/'reservations.json',reservations)
            with self.lock:
                self.stats['created'] += 1
                self.stats['build_seconds'] += time.perf_counter()-started
            return array


class CachedPrepared:
    def __init__(self, original, output):
        self.original = original
        self.arrays = ArrayCache(output)
        self.reader_sha256 = sha256_file(Path(__file__).with_name('xu_geometry.py'))

    def __getattr__(self, name):
        return getattr(self.original,name)

    def _read(self, name, shape, box, label):
        suffix = '_seg.b2nd' if label else '.b2nd'
        relative = f'{self.original._folder}/{name}{suffix}'
        source = self.original._artifact(f'process-{name}',relative)
        marker = read_json(self.original.run_dir/f'commits/process-{name}.json')
        geometry = self.original.geometry(name)
        identity = dict(protocol='xu-exact-grid-cache-1',request=self.original.request_digest,
                        reader_sha256=self.reader_sha256,
                        case=name,source=marker['artifacts'][relative],label=label,
                        shape=list(shape) if shape else None,
                        box=[list(pair) for pair in box] if box else None,
                        cache_shape=list(geometry.cache_shape))
        stat = source.stat()
        before = (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        def build():
            reader = self.original.read_label if label else self.original.read_image
            result = reader(name,shape,box)
            after = source.stat()
            if before != (after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('生成网络输入缓存时源文件发生变化')
            return result
        return self.arrays.get(identity,build)

    def read_image(self, name, shape=None, box=None):
        return self._read(name,shape,box,False)

    def read_label(self, name, shape=None, box=None):
        return self._read(name,shape,box,True)
