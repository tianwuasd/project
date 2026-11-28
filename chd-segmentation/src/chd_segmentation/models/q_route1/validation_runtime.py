from dataclasses import replace
import gc
import math
from pathlib import Path
from time import perf_counter

from filelock import FileLock
import numpy as np
import torch

from ...acceleration import ResourceDemand, SafeBoundary
from ...acceleration.host import host_capacity
from ...acceleration.limits import host_fraction, memory_ceiling, project_usage, reserve_margin
from ...acceleration.memory import alive, system_headroom
from ...artifacts.store import read_json, write_json
from ...observability.events import event


MIB = 1024**2


def assignments(names, shapes, world):
    if world < 1 or len(set(names)) != len(names) or not names:
        raise ValueError('验证需要非空、无重复病例及正数进程数量')
    groups, loads = [[] for _ in range(world)], [0] * world
    for name in sorted(names, key=lambda name: (-math.prod(shapes[name]), name)):
        rank = min(range(world), key=lambda rank: (loads[rank], rank))
        groups[rank].append(name)
        loads[rank] += math.prod(shapes[name])
    return groups


def aggregate(records, names, rows):
    indexed = {row['case_id']: row['loss'] for row in records}
    if len(indexed) != len(records) or set(indexed) != set(names):
        raise ValueError('验证结果缺少病例、存在重复或混入其他病例')
    patients = {}
    by_patient = all(rows[name].get('patient') for name in names)
    for name in names:
        key = rows[name]['patient'] if by_patient else name
        patients.setdefault(key, []).append(indexed[name])
    value = float(np.mean([np.mean(values) for values in patients.values()]))
    if not np.isfinite(value):
        raise FloatingPointError('验证归约结果非有限')
    return value


def case_demand(stage, shape, patch, settings, spacing=None):
    voxels = math.prod(shape)
    padded = math.prod(max(n, p) for n, p in zip(shape, patch))
    arrays = (48 * padded if stage == 'base' else (96 if stage == 'localizer' else 160) * padded)
    transient = voxels * (320 if stage == 'base' else 512)
    maximum = settings.get('validation_max_block_bytes')
    if maximum is not None:
        transient = min(transient, maximum)
    if not settings.get('validation_auto_blocks', True):
        transient = min(transient, settings.get('validation_block_bytes', 64*MIB))
    minimum = min(max(MIB, transient), settings.get('validation_block_bytes', 64*MIB))
    if stage == 'localizer':
        from .training_inputs import input_plan
        if spacing is None:
            raise ValueError('定位器验证预算需要实际毫米间距')
        target_minimum = input_plan(shape,spacing,100_000_000_000)['minimum_workspace_bytes']
        if maximum is not None and maximum < target_minimum:
            raise MemoryError('验证最大块预算无法容纳定位监督halo')
        minimum = max(minimum,target_minimum)
        transient = max(transient,minimum)
    return dict(arrays=arrays, transient=max(MIB, transient), minimum=minimum,
                desired=arrays + max(MIB, transient))


def divide_budget(desires, minimums, amount):
    if sum(minimums) > amount:
        raise MemoryError('验证预算无法容纳每个活动进程的最小计算块')
    values = list(minimums)
    remaining = amount - sum(values)
    pending = {i for i, want in enumerate(desires) if values[i] < want}
    while remaining and pending:
        share = max(1, remaining // len(pending))
        for index in sorted(pending):
            take = min(share, desires[index] - values[index], remaining)
            values[index] += take
            remaining -= take
        pending = {i for i in pending if values[i] < desires[i]}
    return values


def available_budget(common, handle):
    settings = common['resources']
    facts = host_capacity()
    spare, _ = system_headroom(facts, host_fraction(settings), settings['reserve_ram_bytes'])
    root = Path(common['work_root']) / 'resources'
    with FileLock(str(root / '.resource.lock')), handle.condition:
        own = handle.owner
        peers = [row for row in read_json(root / 'active.json') if alive(row) and
                 (row['pid'], row['birth']) != (own['pid'], own['birth'])]
        pending = sum(max(0, row['ram_bytes'] - row.get('rss_bytes', 0)) for row in peers)
        used = handle.state['task_rss']
    return max(0, int(min(memory_ceiling(settings, facts) - project_usage(peers, settings), used + spare - pending)))


def request_budget(common, allocation, memory, demands, update):
    original = ResourceDemand(**allocation['resource_demand'])
    base = original.memory_bytes
    settings = common['resources']
    preferred = settings['memory_margin_fraction']
    desires = [row['desired'] if row else 0 for row in demands]
    minimums = [row['minimum'] if row else 0 for row in demands]
    available = available_budget(common, memory.client.handle)
    minimum = base + sum(minimums)
    if minimum > memory_ceiling(settings, host_capacity()):
        raise MemoryError('验证原始最小布局超过主机或项目上限')
    raw = max(sum(minimums),min(sum(desires),int(max(0,available-base)/(1+preferred))))
    buffered = reserve_margin(raw,max(0,available-base),preferred)
    target = base + buffered['memory_bytes']
    quotas = divide_budget(desires,minimums,raw)
    updated = replace(original, fixed_bytes=original.fixed_bytes + target - base)
    applied = memory.client.request_replan(updated, safe_boundary=SafeBoundary('before_case', update),
                                           layout=allocation['resource_layout'])
    return dict(memory_bytes=applied['memory_bytes'], quotas=quotas, base_bytes=base,
                safety_margin=1+buffered['memory_margin_fraction'],available_bytes=available,
                raw_minimum_bytes=minimum,raw_increment_bytes=raw,
                requested_memory_margin_fraction=preferred,
                memory_margin_fraction=buffered['memory_margin_fraction'],margin_bytes=buffered['margin_bytes'])


def validate_parallel(bundle, dataset, names, rank, common, allocation, gate, device, memory, update):
    from ..execution import all_states
    from .validation import validate_records

    root = Path(bundle.config['run_dir']) / bundle.stage
    shapes = {row['case_id']: row['working_shape'] for row in allocation['supply']['case_estimates']}
    groups = assignments(names, shapes, allocation['world'])
    rows = {row['case_id']: row for row in bundle.prepared['manifest']['cases']}
    if rank == 0:
        event(root, 'q_validation_assignment', update=update, assignments=groups,
              world_size=allocation['world'], cases=len(names))
    spacing = bundle.prepared['plan']['configurations']['3d_fullres']['spacing']
    demand_by_case = {name: case_demand(bundle.stage, shapes[name], bundle.spec['patch'], common['resources'],spacing)
                      for name in names}
    demands = [max((demand_by_case[name] for name in group), key=lambda item:item['desired']) if group else None
               for group in groups]
    dataset.clear()
    bundle.clear_frozen()
    gc.collect()
    phase = f'q-validation:{update}'
    gate.ready.wait(rank, phase + ':memory', device)
    plan = request_budget(common, allocation, memory, demands, update) if rank == 0 else None
    gate.ready.wait(rank, phase + ':planned')
    plan = all_states(plan)[0]
    handle = memory.client.handle
    if rank == 0:
        with handle.condition:
            handle.state['q_validation_finished'] = [not group for group in groups]
        event(root, 'q_validation_memory_plan', update=update, assignments=groups, **plan)
    gate.start(rank, phase, list(map(len,groups)), device)
    dataset.mode, dataset.position = 'validation', 0
    bundle.validation_guard = memory
    bundle.validation_policy = dict(quota_bytes=plan['quotas'][rank], demands=demand_by_case, rank=rank,
                                    handle=handle, shared_quotas=plan['quotas'], update=update)
    with torch.no_grad():
        local = validate_records(bundle, dataset, groups[rank]) if groups[rank] else []
    dataset.clear()
    gc.collect()
    with handle.condition:
        finished = list(handle.state['q_validation_finished'])
        finished[rank] = True
        handle.state['q_validation_finished'] = finished
    gate.finish(rank, phase, device)
    records = [row for result in all_states(local) for row in result]
    dataset.clear()
    bundle.clear_frozen()
    gc.collect()
    gate.ready.wait(rank, f'q-validation:{update}:release', device)
    if rank == 0:
        memory.client.request_replan(ResourceDemand(**allocation['resource_demand']),
            safe_boundary=SafeBoundary('before_case', update), layout=allocation['resource_layout'])
    gate.ready.wait(rank, f'q-validation:{update}:training-memory')
    return aggregate(records, names, rows)


def display_progress(root, world, last):
    now = perf_counter()
    if now - last < 15:
        return last
    for rank in range(world):
        path = root / f'validation-rank{rank}.json'
        if path.is_file():
            row = read_json(path)
            if row['state'] != 'running':
                continue
            seconds = now - row.get('started_monotonic', now - row['seconds'])
            print(f"[validation] rank {rank} | {row['case_id']} | {row['phase']} | "
                  f"块 {row['completed']}/{row['total']} | {row['storage']} | "
                  f"块预算 {row['block_bytes']/MIB:.1f}MiB | 当前病例 {seconds:.1f}秒", flush=True)
    return now
