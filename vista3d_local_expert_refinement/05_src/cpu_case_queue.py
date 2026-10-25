"""Bounded independent CPU commands with inherited workflow lock and owned cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from gpu_queue import stop_owned_process


def run(commands,workers,logdir,completed):
    pending=list(commands);active=[];logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True)
    stop=[False];old=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        while pending or active:
            if stop[0]:raise KeyboardInterrupt('CPU病例队列收到停止请求')
            for item in list(active):
                process,stream,name=item;rc=process.poll()
                if rc is None:continue
                active.remove(item);stream.close()
                if rc:raise RuntimeError(f'CPU病例失败：{name}，日志 {logdir/name}.log')
                completed(name)
            while pending and len(active)<workers:
                name,command=pending.pop(0);env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES='',WHOLE_HEART_CPU_THREADS='1')
                for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
                stream=(logdir/f'{name}.log').open('a',encoding='utf-8')
                try:
                    extra={}
                    if env.get('WHOLE_HEART_MANAGED')=='1':
                        from task_worker import child_options,register_child
                        extra=child_options()
                    process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,**extra)
                    active.append((process,stream,name))
                    if extra:register_child(process,'prepare_case')
                except BaseException:stream.close();raise
            if pending or active:time.sleep(.2)
    finally:
        for process,stream,_ in active:
            try:stop_owned_process(process)
            finally:stream.close()
        signal.signal(signal.SIGTERM,old)
