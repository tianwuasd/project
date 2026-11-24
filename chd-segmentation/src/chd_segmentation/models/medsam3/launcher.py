from contextlib import ExitStack
import multiprocessing as mp
from multiprocessing import spawn
import os
from pathlib import Path
import socket
from time import time,monotonic

import psutil

from ...acceleration.memory import identity
from ...acceleration.sync import HostCoordinator
from ...observability.events import event


def launch(config, environment, session, target, arguments, world, deadline):
    context = mp.get_context('spawn')
    coordinator = HostCoordinator(context,world,session.cancel)
    workers = []
    previous = spawn.get_executable(),spawn.get_preparation_data
    keys = ('PYTHONPATH','PYTHONHOME','PYTHONNOUSERSITE','PYTHONSAFEPATH','PYTHONDONTWRITEBYTECODE','USE_LIBUV')
    saved = {key:os.environ.get(key) for key in keys}
    project = Path(__file__).resolve().parents[4]
    def prepare(name):
        data = previous[1](name)
        data['sys_path'] = [str(project/'src'),str(project),*environment['paths']]
        return data
    def restore():
        mp.set_executable(previous[0])
        spawn.get_preparation_data = previous[1]
        for key,value in saved.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key] = value
    def stop():
        session.cancel.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
            if worker.is_alive():
                raise RuntimeError('MedSAM3子进程未退出，禁止清理')
            event(session.request.run_dir,'medsam3_worker_exited',pid=worker.pid,exit_code=worker.exitcode)
            worker.close()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
    with ExitStack() as cleanup:
        cleanup.callback(stop)
        cleanup.callback(restore)
        for key in ('PYTHONPATH','PYTHONHOME'):
            os.environ.pop(key,None)
        os.environ.update(PYTHONNOUSERSITE='1',PYTHONSAFEPATH='1',PYTHONDONTWRITEBYTECODE='1',USE_LIBUV='0')
        spawn.get_preparation_data = prepare
        mp.set_executable(config['python'])
        for rank in range(world):
            worker = context.Process(target=target,args=(*arguments,rank,world,coordinator,port))
            worker.start()
            workers.append(worker)
            member = identity(psutil.Process(worker.pid))
            member['baseline_rss'] = 0
            session.members[member['pid']] = member
            event(session.request.run_dir,'medsam3_worker_started',rank=rank,**member)
        restore()
        while any(worker.is_alive() for worker in workers):
            if any(worker.exitcode not in (None,0) for worker in workers):
                raise RuntimeError('MedSAM3某rank执行失败，停止本组；见逐rank日志')
            if time() >= deadline:
                raise TimeoutError('MedSAM3达到固定总截止')
            if session.cancel.is_set() or session.state.get('monitor_failed'):
                raise InterruptedError('公共资源监督器已要求停止')
            if session.state['task_rss'] > config.get('training',{}).get('ram_limit_bytes',config['resources']['memory_limit_bytes']):
                raise MemoryError('MedSAM3超过显式进程组内存上限')
            if session.state['sampled_monotonic'] and monotonic()-session.state['sampled_monotonic'] > 30:
                raise InterruptedError('公共资源采样过期')
            for worker in workers:
                worker.join(timeout=.1)
        if any(worker.exitcode != 0 for worker in workers):
            raise RuntimeError('MedSAM3工作进程失败，见逐rank日志')
