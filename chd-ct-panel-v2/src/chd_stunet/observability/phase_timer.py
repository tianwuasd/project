"""Per-case stage timings; progress has a real numerator, or explicitly none."""
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import time
import psutil
from .runlog import write_json


class PhaseTimer:
    def __init__(self,result_path):
        self.path=Path(result_path).with_suffix('.progress.json') if result_path else None
        self.events=self.path.with_suffix('.jsonl') if self.path else None
        self.current=None;self.started=time.monotonic();self.durations={};self.last_write=0
        self.born=psutil.Process().create_time()

    def record(self,event,**fields):
        if self.path is None:return
        row=dict(event=event,stage=self.current,worker_pid=os.getpid(),worker_born=self.born,
                 time=time.time(),timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 elapsed_seconds=time.monotonic()-self.started,**fields)
        write_json(self.path,row)
        with self.events.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        self.last_write=time.monotonic()

    @contextmanager
    def phase(self,name):
        previous=self.current;self.current=name;began=time.monotonic();self.record('stage.start')
        print('阶段开始：'+name,flush=True)
        try:yield
        except BaseException as error:
            self.record('stage.failed',error=str(error),stage_seconds=time.monotonic()-began);raise
        else:
            seconds=time.monotonic()-began;self.durations[name]=self.durations.get(name,0)+seconds
            self.record('stage.done',stage_seconds=seconds)
            print(f'阶段完成：{name}；耗时{seconds:.3f}秒',flush=True)
        finally:self.current=previous

    def wrap(self,name,function):
        def call(*args,**kwargs):
            with self.phase(name):return function(*args,**kwargs)
        return call

    def progress(self,done,total):
        if done==total or time.monotonic()-self.last_write>=2:
            # A class-wise label resize can contain many zoom calls. This is
            # the current call/array's real progress, never a whole-case ETA.
            self.record('stage.progress',completed=done,total=total,progress_scope='current_call_or_array')


class TimedConfiguration:
    """Read-only proxy: native ConfigurationManager properties have no setters."""
    def __init__(self,configuration,timer):
        self._configuration=configuration
        self.resampling_fn_data=timer.wrap('image_resampling',configuration.resampling_fn_data)
        self.resampling_fn_seg=timer.wrap('label_resampling',configuration.resampling_fn_seg)

    def __getattr__(self,name):return getattr(self._configuration,name)
