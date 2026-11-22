from copy import deepcopy
from contextlib import ExitStack
import multiprocessing as mp
from multiprocessing import spawn
import os
from pathlib import Path
import sys
import time

import psutil

from ...acceleration import ResourceDemand, ResourceSession, RuntimeRequest
from ...acceleration.memory import identity
from ...acceleration.legacy import SessionGuard
from ...acceleration.limits import reserve_margin
from ...acceleration.registry import inventory as gpu_inventory
from ...artifacts.store import read_json, require_output, write_json
from ..medsam3.input import open_image, selected_cases, source_record
from .materials import FILES, inventory, environment


def stop_worker(worker, cancel, state):
    cancel.set()
    if worker.pid is not None:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)
        if worker.is_alive():
            raise RuntimeError('SAM3 worker未退出，禁止清理其工作文件')
        state['worker_exit_code'] = worker.exitcode
    worker.close()


def inspect_inputs(config, case, slices):
    if not 1 <= len(slices) <= 2 or len(set(slices)) != len(slices):
        raise ValueError('只允许一至两个不重复切片')
    for name, size in FILES.items():
        if (Path(config['weights']) / name).stat().st_size != size:
            raise ValueError('本地固定版本材料缺失或大小不同：' + name)
    row = selected_cases(config, [case])[0]
    image = open_image(row)
    if any(type(k) is not int or not 0 <= k < image.shape[2] for k in slices):
        raise ValueError('切片索引越界')
    return dict(row=row, slices=sorted(slices), source=source_record(row, image, config),
                model_loaded=False, voxels_read=False, training_updates=0)


def predict(config, case, slices, output, deadline):
    from .worker import run
    if not 0 < deadline - time.time() <= 900 or len(slices) != 1:
        raise ValueError('本批只接受一片及15分钟内的固定截止')
    inspected = inspect_inputs(config, case, slices)
    root = require_output(output, config['work_root'],
                          [config['data_root'], config['weights'], Path(__file__).parents[4]])
    root.mkdir(parents=True, exist_ok=False)
    resources = deepcopy(config['resources'])
    resources['max_wait'] = min(resources['max_wait'], max(0, deadline-time.time()-120))
    resources['vram_reserve_bytes'] += config['estimated_vram_bytes']
    demand = ResourceDemand(fixed_bytes=config['estimated_ram_bytes'], cpu_slots=resources['cpu_threads'],
                            gpu_count=1, gpu_ids=resources['gpu_ids'], local_batches=[1])
    request = RuntimeRequest(str(root), 'sam3_selected_slices', config['registry_root'], str(root), resources, demand)

    def planner(*, world, shares, memory_budget_bytes, cpu_budget_slots, facts):
        plan = reserve_margin(demand.memory_bytes, min(memory_budget_bytes, config['ram_limit_bytes']),
                              resources['memory_margin_fraction'])
        result = dict(admitted=plan['admitted'], minimum_bytes=plan['memory_bytes'],
                      raw_minimum_bytes=demand.memory_bytes, reason='SAM3完整图像模型逐张量加载工作集')
        if plan['admitted']:
            result.update(demand=ResourceDemand(fixed_bytes=plan['memory_bytes'],
                cpu_slots=demand.cpu_slots, gpu_count=world, gpu_ids=demand.gpu_ids, local_batches=shares),
                layout=dict(workers=[0], prefetch=1),
                supply=dict(plan, estimated_ram_bytes=demand.memory_bytes))
        return result
    state = dict(state='waiting_resources', started_epoch=time.time(), deadline_epoch=deadline,
                 training_updates=0, prompt=config['prompt'], requested_slices=inspected['slices'])
    write_json(root / 'execution.json', state)
    success = False
    try:
        with ResourceSession(request, planner=planner) as session, ExitStack() as cleanup:
            gpu = session.allocation.gpus[0]
            cap = min(config['vram_limit_bytes'], gpu['free'] - 2**30)
            if cap <= 0:
                raise MemoryError('显卡没有扣除1GiB预留后的可用空间')
            worker_config = dict(config, selected=inspected, effective_vram_limit_bytes=cap, gpu_uuid=gpu['uuid'])
            write_json(root / 'worker.json', worker_config)
            env = environment(config['work_root'], config['dependencies'], direct=True)
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONPATH=os.pathsep.join([config['dependencies'], str(Path(__file__).parents[3])]),
                       CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS=str(resources['cpu_threads']),
                       MKL_NUM_THREADS=str(resources['cpu_threads']))
            env.pop('PYTHONHOME', None)
            context = mp.get_context('spawn')
            child = context.Process(target=run, args=(str(root), session.handle))
            cleanup.callback(stop_worker, child, session.cancel, state)
            previous_environment = {key: os.environ.get(key) for key in {*env, 'PYTHONHOME'}}
            previous_executable = spawn.get_executable()
            previous_preparation = spawn.get_preparation_data

            def isolated_preparation(name):
                data = previous_preparation(name)
                data['sys_path'] = [config['dependencies'], str(Path(__file__).parents[3]),
                                    *[p for p in sys.path if p != config['dependencies']]]
                return data

            try:
                os.environ.update(env)
                os.environ.pop('PYTHONHOME', None)
                spawn.get_preparation_data = isolated_preparation
                mp.set_executable(sys.executable)
                child.start()
            finally:
                spawn.get_preparation_data = previous_preparation
                mp.set_executable(previous_executable)
                for key, value in previous_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            with SessionGuard(session.handle) as guard:
                member = identity(psutil.Process(child.pid))
                member['baseline_rss'] = 0
                session.members[member['pid']] = member
                cutoff = min(deadline-120, time.time() + config['worker_timeout_seconds'])
                state.update(state='running', worker_pid=child.pid, effective_vram_limit_bytes=cap,
                             admitted_epoch=time.time(), worker_deadline_epoch=cutoff)
                write_json(root / 'execution.json', state)
                try:
                    while child.is_alive():
                        if time.time() >= cutoff:
                            raise TimeoutError('SAM3达到固定检查截止时间')
                        if session.cancel.is_set():
                            raise InterruptedError('公共资源监督器已停止')
                        snapshot = guard.client.snapshot()
                        if snapshot.cancelled:
                            raise InterruptedError('公共资源监督器取消本次检查')
                        rss = session.state['task_rss']
                        state['observed_peak_group_rss_bytes'] = max(state.get('observed_peak_group_rss_bytes', 0), rss)
                        if rss > config['ram_limit_bytes'] or psutil.virtual_memory().percent > 80:
                            raise MemoryError('SAM3超过进程组8GiB或整机80%内存上限')
                        if time.monotonic() >= state.get('next_gpu_sample', 0):
                            query_resources = dict(resources, gpu_query_timeout_seconds=min(2., max(.1, cutoff-time.time())),
                                                   gpu_query_retries=0)
                            current = next(g for g in gpu_inventory([child.pid], resources=query_resources,
                                cancel=session.cancel, log_dir=root) if g['uuid'] == gpu['uuid'])
                            state['next_gpu_sample'] = time.monotonic() + 5
                            state['observed_min_gpu_free_bytes'] = min(state.get('observed_min_gpu_free_bytes', current['free']), current['free'])
                            if current['free'] < 2**30:
                                raise MemoryError('当前显卡空闲空间低于1GiB')
                        entries = inventory(root)
                        if len(entries) > 32 or sum(f['bytes'] for f in entries) > 128 * 2**20:
                            raise RuntimeError('切片结果超过批准数量或容量')
                        time.sleep(1)
                    child.join()
                    if child.exitcode != 0:
                        raise RuntimeError(f'SAM3工作进程失败，退出码 {child.exitcode}，详见工作日志')
                finally:
                    state['worker_exit_code'] = child.exitcode
            result = read_json(root / 'summary.json')
            success = True
            return result
    finally:
        state.update(state='completed' if success else 'failed', finished_epoch=time.time())
        write_json(root / 'execution.json', state)
