from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import psutil

from ..contracts import atomic_json


class EvaluationLog:
    def __init__(self, out=None, heartbeat_seconds=15):
        self.out = Path(out) if out is not None else None
        self.interval = heartbeat_seconds
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.current = dict(state='running', phase='evaluation', stage='starting',
                            completed=0, total=0, last_progress_at=None)
        self.future = None
        self.stream = None

    def __enter__(self):
        if self.out is not None:
            self.stream = (self.out/'evaluation.log').open('a', encoding='utf-8')
        self.emit('evaluation_started')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='evaluation-heartbeat')
        self.future = self.pool.submit(self._heartbeat)
        return self

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit('heartbeat')

    def emit(self, event, **fields):
        with self.lock:
            if self.future is not None and self.future.done():
                self.future.result()
            now = time.time()
            if event != 'heartbeat':
                if fields.get('stage') != self.current.get('stage'):
                    self.stage_started = time.monotonic()
                self.current.update(fields)
                if event.endswith('_completed'):
                    self.current['last_progress_at'] = now
            elapsed = time.monotonic()-self.started
            cpu = self.process.cpu_times()
            completed = self.current['completed']
            total = self.current['total']
            measured_completed=max(0,completed-self.current.get('reused_cases',0))
            row = dict(self.current, event=event, timestamp=datetime.now(timezone.utc).isoformat(),
                       updated_at=now, elapsed_seconds=elapsed,
                       stage_seconds=time.monotonic()-getattr(self,'stage_started',self.started),
                       cpu_seconds=cpu.user+cpu.system, rss_bytes=self.process.memory_info().rss,
                       eta_seconds=elapsed*(total-completed)/measured_completed if measured_completed else None)
            text = json.dumps(row, ensure_ascii=False, allow_nan=False)
            print(text, flush=True)
            if self.stream is not None:
                self.stream.write(text+'\n')
                self.stream.flush()
            # 高频结构阶段只追加日志；进度快照每15秒及病例边界更新。
            if self.out is not None and (event == 'heartbeat' or not event.startswith('metric_')):
                atomic_json(self.out/'evaluation-progress.json',row)
            return row

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        self.pool.shutdown(wait=True)
        try:
            self.future.result()
            if error is not None:
                self.emit('evaluation_failed',state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                          error_type=error_type.__name__,error=str(error))
        finally:
            if self.stream is not None:
                self.stream.close()
        return False
