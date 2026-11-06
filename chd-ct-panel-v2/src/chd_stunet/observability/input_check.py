"""Visible, bounded input-check feedback before a training run is created.

The training contract must be verified before writing to an existing output.
Consequently startup progress goes to stdout (and the caller's launcher log),
not into a possibly incompatible training run. No background data work or
invented byte percentages: only the actual case, phase and elapsed duration.
"""
import threading
import time


class InputCheckProgress:
    def __init__(self,label,heartbeat=10,emit=None):
        self.label=label;self.heartbeat=heartbeat
        self.emit=emit or (lambda value:print(value,flush=True))
        self.lock=threading.RLock();self.stop=threading.Event()
        self.current=dict(case='清单/角色/分折计划',done=0,total='?',phase='读取来源合同')
        self.began=self.phase_began=time.monotonic()
        self.thread=threading.Thread(target=self._heartbeat,name='input-check-progress',daemon=True)

    def __enter__(self):
        self.emit(f'开始输入核验：{self.label}；逐项核验旧结果，源文件只读')
        self.thread.start();return self

    def update(self,case,done,total,phase):
        with self.lock:
            self.current=dict(case=case,done=done,total=total,phase=phase)
            self.phase_began=time.monotonic()
            self._print('核验')

    def _print(self,prefix):
        row=self.current
        self.emit(f"{prefix}：{self.label} 已完成{row['done']}/{row['total']}；"
            f"{row['case']}；{row['phase']}；该阶段{time.monotonic()-self.phase_began:.1f}秒")

    def _heartbeat(self):
        while not self.stop.wait(self.heartbeat):
            with self.lock:self._print('等待核验')

    def __exit__(self,kind,error,traceback):
        self.stop.set();self.thread.join()
        self.emit(f"输入核验{'失败' if error else '完成'}：{self.label}；总耗时{time.monotonic()-self.began:.2f}秒"
                  +(f'；{type(error).__name__}: {error}' if error else ''))
