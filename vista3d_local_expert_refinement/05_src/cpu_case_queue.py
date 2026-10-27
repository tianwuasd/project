"""Bounded independent CPU commands with inherited workflow lock and owned cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from gpu_queue import stop_owned_process


def run(commands,workers,logdir,completed,*,refresh=None):
    if int(workers)<1 and commands:raise MemoryError('CPU case capacity is zero; 未启动病例')
    pending=list(commands);active=[];logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True)
    stop=[False];old=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        while pending or active:
            if stop[0]:raise KeyboardInterrupt('CPU病例队列收到停止请求')
            for item in list(active):
                process,stream,name,assigned=item;rc=process.poll()
                if rc is None:continue
                active.remove(item);stream.close()
                if rc:raise RuntimeError(f'CPU病例失败：{name}，日志 {logdir/name}.log')
                completed(name)
            while pending:
                budget=refresh() if refresh else dict(workers=workers,threads_per_worker=1,cpu_budget=workers)
                threads=min(int(budget['threads_per_worker']),max(0,int(budget['cpu_budget'])-sum(r[3] for r in active)))
                if len(active)>=int(budget['workers']) or threads<1:
                    if not active:raise MemoryError('CPU case capacity is zero: '+str(budget.get('capacity_status')))
                    break
                name,command=pending.pop(0);env=os.environ.copy()
                from task_resources import thread_environment
                env.update(thread_environment(threads,env));env['CUDA_VISIBLE_DEVICES']=''
                stream=(logdir/f'{name}.log').open('a',encoding='utf-8')
                try:
                    extra={}
                    if env.get('WHOLE_HEART_MANAGED')=='1':
                        from task_worker import child_options,register_child
                        extra=child_options()
                    process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,**extra)
                    active.append((process,stream,name,threads))
                    if extra:register_child(process,'prepare_case')
                except BaseException:stream.close();raise
            if pending or active:time.sleep(.2)
    finally:
        for process,stream,_,_ in active:
            try:stop_owned_process(process)
            finally:stream.close()
        signal.signal(signal.SIGTERM,old)
