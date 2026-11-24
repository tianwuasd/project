from copy import deepcopy
from pathlib import Path
from time import time

from ...acceleration import ResourceDemand, ResourceSession, RuntimeRequest
from ...artifacts.store import operation_lock, read_json, require_output, write_json
from .input import selected_cases
from .environment import inspect_environment
from .weights import dependencies
from .distributed import topology


def request(config, output, evaluation=False):
    resources = deepcopy(config['resources'])
    resources['device'] = 'cpu' if evaluation else config['device']
    resources['memory_policy'] = 'host_fraction'
    plan = dict(world=1,batch=1,shares=[1]) if evaluation else topology(config)
    resources['batch_layout'] = plan['shares']
    resources['gpu_spec'] = dict(resource_model='medsam3', resolution=1008,
        queries=len(config['training']['classes']) if 'training' in config else len(config['prompts']),
        stage='training' if 'training' in config else 'prediction',
        accumulation='per_sample_global_mean',
        dependencies={key:[Path(config[key]).stat().st_size,Path(config[key]).stat().st_mtime_ns]
                      for key in ('base_weights','lora_weights','official_config','bpe_path')}
        if not evaluation else {})
    if not evaluation and config.get('capacity_probe_bytes', 0):
        resources['gpu_spec']['capacity_probe_bytes'] = config['capacity_probe_bytes']
    if not evaluation and config['device'] == 'cuda' and not config.get('capacity_probe_bytes', 0):
        resources['vram_reserve_bytes'] += config['estimated_vram_bytes']
    gpu_count = plan['world'] if resources['device'] == 'cuda' else 0
    demand = ResourceDemand(
        fixed_bytes=512 * 2**20 if evaluation else plan['world']*config['fixed_ram_bytes'],
        workspace_bytes=plan['world']*config['workspace_bytes'],
        cpu_slots=resources['cpu_threads'], gpu_count=gpu_count,
        gpu_ids=resources['gpu_ids'] if gpu_count else None, local_batches=plan['shares'])
    return RuntimeRequest(str(output), 'medsam3_evaluation' if evaluation else 'medsam3_prediction',
        config['registry_root'], str(output), resources, demand)


def predict(config, names, output, stop_after=None):
    from .worker import execute
    from .launcher import launch
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('切片停止数需要正整数')
    rows = selected_cases(config, names)
    plan = topology(config)
    if len(rows) < plan['world'] or config['resources']['cpu_threads'] < plan['world']:
        raise ValueError('并行预测至少需要每卡一例和每rank一个CPU线程')
    protected = [config['data_root'], config['official_root'], Path(__file__).parents[3]]
    root = require_output(output, config['work_root'], protected)
    dependency_record = dependencies(config)
    environment = inspect_environment(config['python'])
    config = dict(config, environment=environment)
    with operation_lock(root), ResourceSession(request(config, root)) as session:
        session.state['medsam3_prediction_claimed'] = 0
        launch(config,environment,session,execute,(config,rows,str(root),session.handle,
            session.allocation.gpus,dependency_record,stop_after),plan['world'],time()+config['worker_timeout_seconds'])
        reports = [read_json(root/f'summary.rank{rank}.json') for rank in range(plan['world'])]
        result = dict(protocol=reports[0]['protocol'],state='completed' if all(r['state']=='completed' for r in reports)
            else 'interrupted_at_slice_boundary',cases=[name for r in reports for name in r.get('cases',[])],
            new_slices=sum(r['new_slices'] for r in reports),reused_slices=sum(r['reused_slices'] for r in reports),
            seconds=max(r['seconds'] for r in reports),world=plan['world'],ranks=reports,
            coverage=config['coverage'],scope='technical_slices' if config['coverage']=='selected' else 'whole_volume',training_updates=0)
        write_json(root/'summary.json',result)
        return result
