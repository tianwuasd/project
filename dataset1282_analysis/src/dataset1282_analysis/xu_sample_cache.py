import os
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4

import numpy as np

from .contracts import atomic_json, digest, read_json, sha256_file


class ArrayCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = threading.Lock()
        self.entries = {}
        self.verified = {}
        self.stats = dict(hits=0,created=0,read_seconds=0.,build_seconds=0.)

    def get(self, identity, build):
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
                return array
            started = time.perf_counter()
            array = np.ascontiguousarray(build())
            if array.dtype.hasobject or array.ndim != 3:
                raise ValueError('输入缓存只接受三维数值数组')
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
            atomic_json(receipt,record)
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
