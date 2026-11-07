import copy
import time

from .contracts import atomic_json, read_json
from .observability.status import write_status
from .xu_resources import GpuResourcesChanged, ResourceUnavailable


def run_with_reallocation(operation, args, out):
    attempt = 0
    running = copy.copy(args)
    recovery = out/'resources/recovery.json'
    if running.resume and recovery.exists():
        previous = read_json(recovery)
        for key in ('force_serial','prefetch_workers','prefetch_depth'):
            setattr(running,key,previous[key])
        attempt = previous['attempt']
    while True:
        try:
            return operation(running)
        except ResourceUnavailable as error:
            attempt += 1
            if not getattr(args,'resource_recovery',False):
                if not isinstance(error,GpuResourcesChanged):
                    raise
                record_reallocation(out, error, attempt, getattr(args,'max_reallocations',3))
                running.resume = True
            else:
                # 资源压力只调整执行并发；保持模型、batch 和最近完整 epoch。
                host_pressure = not isinstance(error,GpuResourcesChanged)
                running.force_serial = getattr(running,'force_serial',False) or host_pressure
                workers = getattr(running,'prefetch_workers',4)
                running.prefetch_workers = max(1,workers//2) if host_pressure else workers
                running.prefetch_depth = 1 if host_pressure else getattr(running,'prefetch_depth',2*workers)
                running.resume = (out/'run.json').exists()
                slowdown = 2**min(3,max(0,attempt-getattr(args,'max_reallocations',3)))
                delay = max(.05,min(300,getattr(args,'resource_poll_seconds',60)*slowdown))
                row = dict(attempt=attempt,force_serial=running.force_serial,prefetch_workers=running.prefetch_workers,
                           prefetch_depth=running.prefetch_depth,checked_at=time.time(),next_check_seconds=delay,
                           **error.evidence())
                atomic_json(recovery,row)
                atomic_json(out/'reallocations'/f'{attempt}-{time.time_ns()}.json',row)
                write_status(out,state='waiting_resources',stage=getattr(error,'xu_stage','admission'),
                             log=getattr(error,'xu_log',None),**row)
                print(f'资源不足，已释放工作进程；{delay:g} 秒后重新准入: {error}',flush=True)
                try:
                    time.sleep(delay)
                except KeyboardInterrupt:
                    write_status(out,state='interrupted',stage='resource-wait',reason='操作员停止等待')
                    raise
            running.reallocate = True


def record_reallocation(out, error, attempt, maximum):
    if attempt > maximum:
        raise error
    unfinished = {}
    for file in sorted((out/'assignments').glob('*.json')):
        if (out/'receipts'/file.name).exists():
            continue
        stage = file.stem
        checkpoints = [str(p) for p in (out/stage).glob('*last.pt')]
        unfinished[stage] = dict(assignment=read_json(file), checkpoints=checkpoints)
    row = dict(attempt=attempt, maximum=maximum, changed_at=time.time(),
               **error.evidence(), unfinished=unfinished,
               recovery='restart from last complete epoch; replay incomplete work')
    atomic_json(out/'reallocations'/f'{attempt}-{time.time_ns()}.json', row)
    write_status(out, state='reallocating', stage='gpu-allocation', **row)
    print(f'GPU 资源变化，重新分配 {attempt}/{maximum}: {error}', flush=True)
