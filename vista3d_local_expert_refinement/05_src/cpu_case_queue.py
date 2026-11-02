"""Bounded independent CPU commands with inherited workflow lock and owned cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from gpu_queue import stop_owned_process


def private_memory_bytes(process):
    """Credit only resident private bytes of the still-owned live child.

    Shared RSS and unobserved descendant allocations are not credited. Missing
    USS keeps the full future reservation, rather than guessing it is resident.
    """
    try:
        import psutil
        child=psutil.Process(process.pid)
        if process.poll() is not None:return 0
        born=child.create_time()
        value=max(0,int(child.memory_full_info().uss))
        if process.poll() is not None or psutil.Process(process.pid).create_time()!=born:return 0
        return value
    except (ImportError,AttributeError,OSError,ValueError):return 0
    except Exception:
        # psutil's AccessDenied/NoSuchProcess are optional-runtime exceptions.
        return 0


def run(commands,workers,logdir,completed,*,refresh=None):
    if int(workers)<1 and commands:raise MemoryError('CPU case capacity is zero; 未启动病例')
    pending=list(commands);active=[];logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True)
    from read_admission import create_gate,update_gate
    gate=create_gate(workers)
    stop=[False];old=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        stage=refresh() if refresh else dict(workers=workers,threads_per_worker=1,cpu_budget=workers)
        stage_cpu=int(stage['cpu_budget'])
        stage_memory=stage.get('memory_budget_bytes')
        stage_workers=min(int(workers),int(stage['workers']))
        while pending or active:
            if stop[0]:raise KeyboardInterrupt('CPU病例队列收到停止请求')
            for item in list(active):
                process,stream,name,assigned,reserved=item;rc=process.poll()
                if rc is None:continue
                active.remove(item);stream.close()
                if rc:raise RuntimeError(f'CPU病例失败：{name}，日志 {logdir/name}.log')
                completed(name)
            while pending:
                budget=refresh() if refresh else dict(workers=workers,threads_per_worker=1,cpu_budget=workers)
                update_gate(gate,budget.get('read_concurrency',budget['workers']))
                estimate=int(budget.get('estimated_bytes_per_case',0))
                memory=budget.get('memory_budget_bytes')
                reserved=sum(max(r[4],estimate) for r in active)
                # Fresh available RAM already excludes resident allocations.
                # Reserve only their not-yet-resident remainder a second time;
                # the original stage ceiling still accounts for every case.
                outstanding=sum(max(0,max(r[4],estimate)-private_memory_bytes(r[0])) for r in active)
                if ((stage_memory is not None and reserved+estimate>stage_memory) or
                        (memory is not None and outstanding+estimate>memory)):
                    if not active:raise MemoryError('CPU case RAM reservation exceeds refreshed capacity')
                    break
                threads=min(int(budget['threads_per_worker']),int(budget['cpu_budget']),
                    max(0,stage_cpu-sum(r[3] for r in active)))
                unknown=budget.get('capacity_status')=='unknown_resources_bounded_fallback'
                current_limit=min(stage_workers,int(budget['workers'])) if unknown else stage_workers
                if len(active)>=current_limit or int(budget['workers'])<1 or threads<1:
                    if not active:raise MemoryError('CPU case capacity is zero: '+str(budget.get('capacity_status')))
                    break
                name,command=pending.pop(0);env=os.environ.copy()
                from task_resources import thread_environment
                env.update(thread_environment(threads,env));env['CUDA_VISIBLE_DEVICES']='';env['VISTA_READ_GATE']=str(gate)
                stream=(logdir/f'{name}.log').open('a',encoding='utf-8')
                try:
                    extra={}
                    if env.get('WHOLE_HEART_MANAGED')=='1':
                        from task_worker import child_options,register_child
                        extra=child_options()
                    process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,**extra)
                    active.append((process,stream,name,threads,estimate))
                    if extra:register_child(process,'prepare_case')
                except BaseException:stream.close();raise
            if pending or active:time.sleep(.2)
    finally:
        for process,stream,_,_,_ in active:
            try:stop_owned_process(process)
            finally:stream.close()
        signal.signal(signal.SIGTERM,old)
