from contextlib import contextmanager
from pathlib import Path

from ..acceleration import ResourceSession, RuntimeRequest, ResourceDemand
from .budget import cpu_plan
from .resources import layout


def supply_layout(supply):
    return dict(workers=supply['workers'], cache_bytes=supply['cache_bytes'], prefetch=supply['prefetch'])


def supply_demand(supply, world, resources, shares, gpus=None):
    amount = supply['ram_bytes'] if supply['admitted'] else supply['minimum_bytes']
    workspace = supply['workspace_capacity_bytes']
    return ResourceDemand(fixed_bytes=int(amount-workspace), workspace_bytes=int(workspace),
        reader_slots=int(supply['reader_slots']), cpu_slots=int(supply['cpu_slots']),
        gpu_count=world if resources['device']=='cuda' else 0,
        gpu_ids=(None if resources['device']=='cpu' else resources['gpu_ids'] if gpus is None
            else [gpu['uuid'] for gpu in gpus]),local_batches=shares)


def supply_planner(config,prepared,stage,estimates):
    def plan(*,world,shares,memory_budget_bytes,cpu_budget_slots,facts):
        from copy import deepcopy
        candidate = deepcopy(config)
        candidate['resources']['batch_layout'] = shares
        supply = cpu_plan(candidate,prepared,world,stage=stage,estimates=estimates,
            bounds=dict(memory_budget_bytes=memory_budget_bytes,cpu_budget_slots=cpu_budget_slots,facts=facts))
        result = dict(admitted=supply['admitted'],minimum_bytes=supply['minimum_bytes'],
                      raw_minimum_bytes=supply['raw_minimum_bytes'],reason=supply['reason'])
        if supply['admitted']:
            result.update(demand=supply_demand(supply,world,candidate['resources'],shares),
                layout=supply_layout(supply),supply=supply)
        return result
    return plan


@contextmanager
def admitted_training(config, prepared, spec, previous_world=None, stage='train'):
    resources = config['resources']
    world = resources['gpu_count'] if resources['device'] == 'cuda' else resources['cpu_processes']
    shares = layout(config['training']['batch'], world, resources['batch_layout'])
    supply = cpu_plan(config, prepared, world, stage=stage)
    amount = supply['ram_bytes'] if supply['admitted'] else supply.get('minimum_bytes', 0)
    if amount <= 0:
        raise RuntimeError('训练输入无法建立有效内存预算：'+supply.get('reason', '未知原因'))
    demand = supply_demand(supply, world, resources, shares)
    request = RuntimeRequest(str(Path(config['run_dir']).resolve()), stage,
        Path(config['work_root'])/'resources', config['run_dir'],
        dict(resources,gpu_spec=dict(spec,operation='prediction') if stage=='prediction' else spec,
             max_wait=resources['wait_seconds'],
             precision=config['training']['precision'] if stage!='prediction' else 'fp32',
             previous_world=previous_world), demand)
    with ResourceSession(request, layout=supply_layout(supply),
            planner=supply_planner(config,prepared,stage,supply['case_estimates'])) as session:
        applied = session.allocation
        yield dict(owner=session.handle.owner, resource_handle=session.handle,
            world=applied.world, shares=applied.shares,gpus=applied.gpus,supply=applied.supply),session


def memory_for(job):
    from ..acceleration.legacy import SessionGuard
    value = SessionGuard(job['resource_handle'])
    value.config = job['config']
    return value
