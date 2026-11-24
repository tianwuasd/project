import os
from pathlib import Path
from time import time

from filelock import FileLock
import psutil
import torch

from ...artifacts.store import digest,read_json,write_json
from ...observability.events import event


def cache_options(values):
    defaults = dict(enabled=True, localizer_bytes=2*2**30, local_bytes=16*2**30,lock_timeout_seconds=3600)
    if not isinstance(values,dict) or set(values)-defaults.keys():
        raise ValueError('Q训练缓存配置字段无效')
    result = {**defaults,**values}
    if type(result['enabled']) is not bool or any(type(result[key]) is not int or result[key]<0
                                                  for key in ('localizer_bytes','local_bytes')) or (
            type(result['lock_timeout_seconds']) is not int or result['lock_timeout_seconds']<1):
        raise ValueError('Q缓存开关或容量无效')
    return result


class TrainingCache:
    def __init__(self,config,prepared,stage,parents=None):
        self.settings = cache_options(config.get('training_cache',{}))
        self.limit = self.settings['localizer_bytes' if stage=='localizer' else 'local_bytes']
        self.log_dir = Path(config['run_dir'])/stage
        self.path = Path(config.get('work_root') or config['run_dir'])/'q-training-cache'/stage
        self.identity = dict(protocol='q-training-cache-v1',dataset=prepared['dataset'],
            storage=prepared['storage'],plan=prepared['plan'],parents=parents,
            prepared_identity=prepared['identity'],technical=prepared['technical'],
            input_source=config.get('input_source',{}),model=config['model'],method=config['method'],
            patch=config['patch'],local_size=config['local_size'])

    def _directory_lock(self):
        return FileLock(self.path/'cache.lock',timeout=self.settings['lock_timeout_seconds'])

    def _reserve(self,key,amount):
        reserved = 0
        for receipt in self.path.glob('*.reserve.json'):
            owner = read_json(receipt)
            active = psutil.pid_exists(owner['pid']) and psutil.Process(owner['pid']).create_time()==owner['birth']
            if active:
                reserved += owner['bytes']
            else:
                partial = self.path/(receipt.name.removesuffix('.reserve.json')+'.pending')
                if partial.exists():
                    partial.unlink()
                receipt.unlink()
        files = sorted(self.path.glob('*.pt'),key=lambda p:p.stat().st_mtime)
        used = sum(p.stat().st_size for p in files)
        while files and used+reserved+amount>self.limit:
            victim = files.pop(0)
            used -= victim.stat().st_size
            victim.unlink()
        if used+reserved+amount>self.limit:
            return False
        write_json(self.path/(key+'.reserve.json'),dict(pid=os.getpid(),
            birth=psutil.Process().create_time(),bytes=amount))
        return True

    def get(self,entry,produce,estimated_bytes):
        if not self.settings['enabled'] or self.limit==0 or estimated_bytes>self.limit:
            return produce()
        key = digest(dict(identity=self.identity,case=entry['case_id'],shape=list(entry['image'].shape),
            spacing=entry['processed_spacing'],unknown_classes=entry['unknown_classes']))
        self.path.mkdir(parents=True,exist_ok=True)
        target = self.path/(key+'.pt')
        # 病例生成各用一把锁；容量锁只用于登记、读盘与提交，不包围模型计算。
        with FileLock(self.path/(key+'.lock'),timeout=self.settings['lock_timeout_seconds']):
            with self._directory_lock():
                if target.exists():
                    record = torch.load(target,map_location='cpu',weights_only=True)
                    if record['key']!=key:
                        raise ValueError('Q训练缓存身份不同')
                    os.utime(target,None)
                    event(self.log_dir,'q_training_cache_hit',case_id=entry['case_id'],bytes=target.stat().st_size)
                    return record['value']
                reserved = self._reserve(key,estimated_bytes)
            if not reserved:
                return produce()
            pending = self.path/(key+'.pending')
            try:
                value = produce()
                payload = dict(key=key,value=value,created_at=time())
                def tensor_bytes(item):
                    if isinstance(item,torch.Tensor):
                        return item.untyped_storage().nbytes()
                    if isinstance(item,dict):
                        return sum(tensor_bytes(v) for v in item.values())
                    if isinstance(item,(tuple,list)):
                        return sum(tensor_bytes(v) for v in item)
                    return 0
                if tensor_bytes(value)+32768>estimated_bytes:
                    raise ValueError('Q缓存容量预估小于实际张量，停止写入')
                with pending.open('wb') as stream:
                    torch.save(payload,stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                amount = pending.stat().st_size
                if amount>estimated_bytes:
                    raise ValueError('Q缓存序列化大小超过已登记容量')
                with self._directory_lock():
                    os.replace(pending,target)
                    (self.path/(key+'.reserve.json')).unlink()
                    event(self.log_dir,'q_training_cache_saved',case_id=entry['case_id'],bytes=amount,
                          limit_bytes=self.limit)
            finally:
                with self._directory_lock():
                    if pending.exists():
                        pending.unlink()
                    reservation = self.path/(key+'.reserve.json')
                    if reservation.exists():
                        reservation.unlink()
            return value
