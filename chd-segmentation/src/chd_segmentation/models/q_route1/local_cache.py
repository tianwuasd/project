from pathlib import Path
from time import time
from uuid import uuid4
import math
import os
import shutil

from filelock import FileLock
import numpy as np
import psutil
import torch

from ...acceleration.memory import alive
from ...artifacts.cache import release_private_mapping
from ...artifacts.store import digest, read_json, write_json
from ...observability.events import event
from .candidates import Candidate


def disk_plan(shape, patch=32):
    padded = tuple(max(int(patch), math.ceil(int(n)/32)*32) for n in shape)
    voxels = math.prod(padded)
    return dict(padded=padded, retained_bytes=voxels*44+math.prod(n//16 for n in padded)*256+128*2**10,
                peak_bytes=voxels*80+64*2**20)


def _readers(path):
    active = []
    for receipt in path.glob('reader-*.json.pending'):
        if receipt.is_symlink():
            raise ValueError('局部输入未完成读者登记不能为链接')
        if psutil.pid_exists(int(receipt.name.split('-')[1])):
            active.append(receipt)
        else:
            receipt.unlink()
    for receipt in path.glob('reader-*.json'):
        if receipt.is_symlink():
            raise ValueError('局部输入读者登记不能为链接')
        if alive(read_json(receipt)):
            active.append(receipt)
        else:
            receipt.unlink()
    return active


def _remove(path, key):
    if path.is_symlink() or read_json(path/'owner.json')['key'] != key:
        raise ValueError('局部输入目录归属不符')
    allowed = {'owner.json', 'entry.json', 'entry.json.pending', 'probability.npy', 'features.npy'}
    work = path/'work'
    if work.exists():
        if work.is_symlink():
            raise ValueError('局部输入计算工作区归属不符')
        work_owner = work/'owner.json'
        children = list(work.iterdir())
        if work_owner.exists():
            if read_json(work_owner)['identity'].get('key') != key:
                raise ValueError('局部输入计算工作区身份不同')
        elif any(p.name != 'owner.json.pending' for p in children):
            raise ValueError('未提交归属的工作区包含计算文件')
        work_allowed = {'owner.json', 'owner.json.pending', 'logits.npy', 'counts.npy', 'native.npy'}
        work_allowed.update('layer_%02d.npy' % i for i in range(64))
        if any(p.is_symlink() or not p.is_file() or p.name not in work_allowed for p in children):
            raise ValueError('局部计算工作区包含未登记文件')
        for child in children:
            child.unlink()
        work.rmdir()
    files = list(path.iterdir())
    if any(p.is_symlink() or not p.is_file() or p.name not in allowed for p in files):
        raise ValueError('局部输入目录包含未登记文件')
    for file in files:
        file.unlink()
    path.rmdir()


class FrozenInputs:
    def __init__(self, store, path, receipt, temporary):
        self.store, self.path, self.receipt, self.temporary = store, path, receipt, temporary
        self.maps = {}
        self.probability, self.features = None, None
        record = read_json(path/'entry.json')
        self.key = record['key']
        try:
            for name in ('probability', 'features'):
                array = np.load(path/(name+'.npy'), mmap_mode='c', allow_pickle=False)
                self.maps[name] = array
                if array.dtype != np.float32 or list(array.shape) != record['shapes'][name]:
                    raise ValueError('局部输入数组尺寸或类型与登记不同')
            crop = tuple(slice(a, a+n) for a, n in zip(record['before'], record['shape']))
            self.probability = torch.from_numpy(self.maps['probability'][(slice(None), *crop)])[None]
            self.features = torch.from_numpy(self.maps['features'])[None]
            self.candidates = [Candidate(tuple(c['start']), c['source'], c['task'], c['score'])
                               for c in record['candidates']]
        finally:
            if self.probability is None or self.features is None:
                self.close()

    def release(self):
        for array in self.maps.values():
            release_private_mapping(array)

    def close(self):
        self.probability, self.features = None, None
        for array in self.maps.values():
            array._mmap.close()
        self.maps.clear()
        with self.store.lock():
            if self.receipt.exists():
                self.receipt.unlink()
            if self.temporary and self.path.exists() and not _readers(self.path):
                _remove(self.path, self.key)


class LocalCache:
    def __init__(self, cache):
        self.path = cache.path.parent/'local-blocks'
        self.identity = dict(cache.identity, protocol='q-local-blocks-v1')
        self.enabled = cache.settings['enabled']
        self.limit = cache.limit if self.enabled else 0
        self.timeout = cache.settings['lock_timeout_seconds']
        self.log_dir = cache.log_dir

    def lock(self):
        return FileLock(self.path/'cache.lock', timeout=self.timeout)

    def _reader(self, path):
        process = psutil.Process()
        receipt = path/('reader-'+str(process.pid)+'-'+uuid4().hex+'.json')
        write_json(receipt, dict(pid=process.pid, birth=process.create_time()))
        os.utime(path/'entry.json', None)
        return receipt

    def _existing(self, key):
        if not self.enabled or self.limit == 0:
            return None
        for path in [self.path/(key+'.case'), *sorted(self.path.glob(key+'-*.temporary'))]:
            if path.exists():
                if path.is_symlink() or read_json(path/'entry.json')['key'] != key:
                    raise ValueError('局部输入缓存身份不同')
                if path.suffix == '.case' and read_json(path/'entry.json')['bytes'] > self.limit:
                    continue
                return path, self._reader(path), path.suffix == '.temporary'
        return None

    def _reserve(self, key, plan):
        for path in self.path.glob('*.temporary'):
            if path.is_symlink():
                raise ValueError('局部输入临时目录不能为链接')
            if not _readers(path):
                _remove(path, read_json(path/'entry.json')['key'])
        reservations = []
        for receipt in self.path.glob('*.reserve.json'):
            value = read_json(receipt)
            if alive(value):
                reservations.append(value)
            else:
                pending = self.path/value['pending']
                if pending.exists():
                    _remove(pending, value['key'])
                receipt.unlink()
        cached = sorted(self.path.glob('*.case'), key=lambda p:(p/'entry.json').stat().st_mtime)
        amounts = {p:read_json(p/'entry.json')['bytes'] for p in cached}
        used = sum(amounts.values())+sum(r['retained_bytes'] for r in reservations if r['cached'])
        retain = plan['retained_bytes'] <= self.limit
        if retain:
            for path in cached:
                if used+plan['retained_bytes'] <= self.limit:
                    break
                if not _readers(path):
                    used -= amounts[path]
                    _remove(path, read_json(path/'entry.json')['key'])
            retain = used+plan['retained_bytes'] <= self.limit
        pending = self.path/(key+'-'+uuid4().hex+'.pending')
        reserved = sum(r['peak_bytes'] for r in reservations)
        if shutil.disk_usage(self.path).free < reserved+plan['peak_bytes']+2**30:
            raise OSError('局部输入磁盘余量不能容纳已登记工作区和当前病例')
        pending.mkdir()
        write_json(pending/'owner.json', dict(protocol='q-local-blocks-v1', key=key))
        receipt = self.path/(pending.name+'.reserve.json')
        write_json(receipt, dict(pid=os.getpid(), birth=psutil.Process().create_time(), key=key,
            pending=pending.name, cached=retain, **plan))
        return pending, receipt, retain

    def acquire(self, entry, build):
        key = digest(dict(identity=self.identity, case=entry['case_id'], shape=list(entry['image'].shape),
            spacing=entry['processed_spacing'], unknown_classes=entry['unknown_classes']))
        self.path.mkdir(parents=True, exist_ok=True)
        plan = disk_plan(entry['image'].shape[1:], self.identity['patch'])
        with FileLock(self.path/(key+'.lock'), timeout=self.timeout):
            with self.lock():
                existing = self._existing(key)
                if existing is None:
                    pending, reservation, cached = self._reserve(key, plan)
            if existing is not None:
                event(self.log_dir, 'q_local_cache_hit', case_id=entry['case_id'], storage='disk')
                return FrozenInputs(self, *existing)
            committed = False
            try:
                record = build(pending, key)
                amount = sum(p.stat().st_size for p in pending.iterdir())+65536
                if amount > plan['retained_bytes']:
                    raise ValueError('局部输入实际磁盘容量超过登记容量')
                write_json(pending/'entry.json', dict(record, key=key, bytes=amount, created_at=time()))
                with self.lock():
                    target = self.path/(key+'.case') if cached else pending.with_suffix('.temporary')
                    os.replace(pending, target)
                    reader = self._reader(target)
                    reservation.unlink()
                    committed = True
                event(self.log_dir, 'q_local_cache_saved', case_id=entry['case_id'], bytes=amount,
                      retained=cached, storage='disk')
                return FrozenInputs(self, target, reader, not cached)
            finally:
                if not committed:
                    with self.lock():
                        if pending.exists():
                            _remove(pending, key)
                        if reservation.exists():
                            reservation.unlink()
