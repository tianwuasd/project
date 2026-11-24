from copy import deepcopy
import multiprocessing as mp
from multiprocessing import spawn
import os
from pathlib import Path
from time import monotonic

from ...acceleration import ResourceDemand, ResourceSession, RuntimeRequest
from ...artifacts.store import operation_lock, read_json, require_output
from ...observability.events import event
from .input import selected_cases
from .environment import inspect_environment
from .weights import dependencies


def request(config, output, evaluation=False):
    resources = deepcopy(config['resources'])
    resources['device'] = 'cpu' if evaluation else config['device']
    resources['memory_policy'] = 'host_fraction'
    resources['gpu_spec'] = dict(resource_model='medsam3', resolution=1008,
        queries=len(config['training']['classes']) if 'training' in config else len(config['prompts']),
        stage='training' if 'training' in config else 'prediction',
        dependencies={key:[Path(config[key]).stat().st_size,Path(config[key]).stat().st_mtime_ns]
                      for key in ('base_weights','lora_weights','official_config','bpe_path')}
        if not evaluation else {})
    if not evaluation and config.get('capacity_probe_bytes', 0):
        resources['gpu_spec']['capacity_probe_bytes'] = config['capacity_probe_bytes']
    if not evaluation and config['device'] == 'cuda' and not config.get('capacity_probe_bytes', 0):
        resources['vram_reserve_bytes'] += config['estimated_vram_bytes']
    gpu_count = int(resources['device'] == 'cuda')
    demand = ResourceDemand(
        fixed_bytes=512 * 2**20 if evaluation else config['fixed_ram_bytes'],
        workspace_bytes=config['workspace_bytes'],
        cpu_slots=resources['cpu_threads'], gpu_count=gpu_count,
        gpu_ids=resources['gpu_ids'] if gpu_count else None, local_batches=[1])
    return RuntimeRequest(str(output), 'medsam3_evaluation' if evaluation else 'medsam3_prediction',
        config['registry_root'], str(output), resources, demand)


def predict(config, names, output, stop_after=None):
    from .worker import run
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('切片停止数需要正整数')
    rows = selected_cases(config, names)
    protected = [config['data_root'], config['official_root'], Path(__file__).parents[3]]
    root = require_output(output, config['work_root'], protected)
    dependency_record = dependencies(config)
    environment = inspect_environment(config['python'])
    config = dict(config, environment=environment)
    with operation_lock(root), ResourceSession(request(config, root)) as session:
        context = mp.get_context('spawn')
        worker = context.Process(target=run, args=(config, rows, str(root), session.handle,
            session.allocation.gpus, dependency_record, stop_after))
        previous = spawn.get_executable()
        prepare = spawn.get_preparation_data
        previous_environment = {key: os.environ.get(key) for key in
            ('PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONSAFEPATH')}
        project = Path(__file__).resolve().parents[4]

        def isolated_preparation(name):
            data = prepare(name)
            if 'sys_path' not in data:
                raise RuntimeError('当前Python spawn启动合同不受支持')
            data['sys_path'] = [str(project / 'src'), str(project), *environment['paths']]
            return data

        try:
            for key in ('PYTHONPATH', 'PYTHONHOME'):
                os.environ.pop(key, None)
            os.environ.update(PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1')
            spawn.get_preparation_data = isolated_preparation
            mp.set_executable(config['python'])
            worker.start()
        finally:
            mp.set_executable(previous)
            spawn.get_preparation_data = prepare
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        began = monotonic()
        try:
            while worker.is_alive():
                worker.join(timeout=1)
                if monotonic() - began > config['worker_timeout_seconds']:
                    session.cancel.set()
                    raise TimeoutError('MedSAM3达到显式入口运行时限')
                if session.cancel.is_set():
                    raise InterruptedError('公共资源监督器已要求停止')
            if worker.exitcode != 0:
                raise RuntimeError(f'MedSAM3工作进程失败，退出码{worker.exitcode}')
        finally:
            if worker.is_alive():
                session.cancel.set()
                worker.terminate()
                worker.join()
            event(root, 'medsam3_worker_exited', pid=worker.pid, exit_code=worker.exitcode)
            worker.close()
        return read_json(root / 'summary.json')
