from contextlib import contextmanager
from pathlib import Path

from ..acceleration import ResourceSession, RuntimeRequest, ResourceDemand, SafeBoundary
from .budget import cpu_plan
from .resources import layout


def supply_layout(supply):
    return dict(workers=supply['workers'], cache_bytes=supply['cache_bytes'], prefetch=supply['prefetch'])


def supply_demand(supply, world, resources, shares):
    amount = supply['ram_bytes'] if supply['admitted'] else supply['minimum_bytes']
    workspace = supply['workspace_capacity_bytes']
    return ResourceDemand(fixed_bytes=int(amount-workspace), workspace_bytes=int(workspace),
        reader_slots=int(supply['reader_slots']), cpu_slots=int(supply['cpu_slots']),
        gpu_count=world if resources['device']=='cuda' else 0, gpu_ids=resources['gpu_ids'], local_batches=shares)


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
        dict(resources,gpu_spec=spec,max_wait=resources['wait_seconds'],precision=config['training']['precision'],
             previous_world=previous_world), demand)
    with ResourceSession(request, layout=supply_layout(supply)) as session:
        if not supply['admitted']:
            supply = cpu_plan(config, prepared, world, stage=stage, allocation=session.allocation,
                              estimates=supply['case_estimates'])
            if not supply['admitted']:
                raise RuntimeError('公共准入后训练供数规划仍不可用：'+supply['reason'])
            session.replan(supply_demand(supply,world,resources,shares),
                safe_boundary=SafeBoundary('before_case'), layout=supply_layout(supply))
        yield dict(owner=session.handle.owner, resource_handle=session.handle,
            world=world, gpus=session.allocation.gpus, supply=supply), session


def memory_for(job):
    from ..acceleration.legacy import SessionGuard
    value = SessionGuard(job['resource_handle'])
    value.config = job['config']
    return value
