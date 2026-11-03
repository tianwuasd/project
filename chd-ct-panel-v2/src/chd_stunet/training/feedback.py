"""Thread-safe progress, one-second RAM feedback and safe-point pressure signal."""
import json
import os
import threading
import time
import psutil
from ..acceleration.preprocess_cpu import snapshot,GIB
from ..observability.runlog import RunLog


class TrainingLog:
    def __init__(self,output):
        self.base=RunLog(output);self.output=output
        self.lock=threading.RLock();self.context=dict(state='running',stage='initializing')

    def event(self,name,message,**fields):
        with self.lock:self.base.event(name,message,**fields)

    def status(self,state=None,**fields):
        with self.lock:
            if state is not None:self.context['state']=state
            self.context.update(fields)
            self.base.status(**self.context)


class MemoryMonitor:
    def __init__(self,output,log,reserve_bytes,device):
        self.output,self.log,self.reserve,self.device=output,log,reserve_bytes,device
        self.stop_event=threading.Event();self.pressure=threading.Event()
        self.error=None;self.samples=0;self.peak_group_rss=0
        self.thread=threading.Thread(target=self._run,name='training-memory-monitor',daemon=True)

    def start(self):self.thread.start()

    def _run(self):
        last_console=0
        while not self.stop_event.is_set():
            try:
                facts=snapshot();free=facts.get('available_memory_bytes')
                if free is None:raise MemoryError('Missing host/cgroup memory evidence')
                self.samples=self.samples+1 if free<self.reserve else 0
                if self.samples>=3:self.pressure.set()
                process=psutil.Process(os.getpid());rss=process.memory_info().rss
                for child in process.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.Error:pass
                self.peak_group_rss=max(self.peak_group_rss,rss)
                record=dict(time=time.time(),free_bytes=free,reserve_bytes=self.reserve,
                            process_group_rss_bytes=rss,pressure_samples=self.samples)
                with (self.output/'logs/training-resources.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record)+'\n')
                self.log.status(resources=record)
                if time.monotonic()-last_console>=20:
                    self.log.event('training.heartbeat',f"阶段{self.log.context.get('stage')}；可用RAM {free/GIB:.1f}GiB；进程组RSS {rss/GIB:.1f}GiB")
                    last_console=time.monotonic()
            except BaseException as error:
                self.error=error;self.pressure.set()
            self.stop_event.wait(1)

    def close(self):self.stop_event.set();self.thread.join(timeout=3)
