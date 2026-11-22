from dataclasses import replace

from .contracts import ResourceDemand


def minimum_host_bytes(request, demand, planner, facts, limit):
    if planner is None:
        return demand.memory_bytes, []
    from .registry import layout as batch_layout
    r = request.resources
    maximum = demand.gpu_count or len(demand.local_batches or []) or r.get('cpu_processes',1)
    minimum = (r.get('min_gpus') or maximum) if demand.gpu_count else maximum
    records = []
    for world in range(maximum,minimum-1,-1):
        batch = sum(demand.local_batches or [1])
        if r.get('gpu_spec',{}).get('resource_model') == 'medsam2' and batch % world:
            continue
        shares = batch_layout(batch,world,r.get('batch_layout'))
        plan = planner(world=world,shares=shares,memory_budget_bytes=limit,
                       cpu_budget_slots=max(facts['cores'],demand.cpu_slots),facts=facts)
        records.append(dict(world=world,raw_minimum_bytes=plan.get('raw_minimum_bytes',plan['minimum_bytes'])))
    if not records:
        raise ValueError('没有符合批量分担的资源拓扑')
    return min(row['raw_minimum_bytes'] for row in records), records


def select_admission(request, demand, planner, *, facts, memory_budget_bytes, cpu_budget_slots,
                     occupied_bytes, history, peaks, peers, layout, keep_gpus=None, initial=True):
    resources = request.resources

    def candidate(*,world,shares,gpus):
        if planner is None:
            proposed = dict(admitted=True,minimum_bytes=demand.memory_bytes,reason='固定资源需求',
                            demand=demand,layout=layout,supply={})
        else:
            proposed = planner(world=world,shares=shares,memory_budget_bytes=max(0,memory_budget_bytes),
                cpu_budget_slots=cpu_budget_slots,facts=facts)
            if not proposed['admitted']:
                return proposed
        chosen = proposed['demand']
        if not isinstance(chosen,ResourceDemand):
            raise TypeError('初始规划器必须返回ResourceDemand')
        if planner is not None:
            if chosen.gpu_count != len(gpus) or list(chosen.local_batches or []) != shares:
                raise ValueError('规划器的GPU数量或份额与候选拓扑不同')
            if len(proposed['layout']['workers']) != world:
                raise ValueError('规划器的worker布局没有覆盖全部rank')
            if chosen.cpu_slots > cpu_budget_slots or chosen.memory_bytes > memory_budget_bytes:
                raise ValueError('规划器返回的布局超过已提供的资源边界')
        cpu = min(chosen.cpu_slots,cpu_budget_slots) if chosen.allow_partial_cpu else chosen.cpu_slots
        required = max(chosen.memory_bytes,occupied_bytes+(chosen.workspace_bytes if chosen.reader_slots else 0))
        if cpu < 1 or cpu > cpu_budget_slots or required > memory_budget_bytes:
            return dict(admitted=False,minimum_bytes=required,reason='实际CPU或内存余量不足最低布局')
        applied = replace(chosen,cpu_slots=cpu,gpu_count=len(gpus),gpu_ids=[g['uuid'] for g in gpus] if gpus else None,
                          local_batches=shares or None)
        return dict(proposed,demand=applied,cpu_slots=cpu,memory_bytes=required)

    if keep_gpus is not None:
        world = len(keep_gpus) or len(demand.local_batches or []) or resources.get('cpu_processes',1)
        shares = list(demand.local_batches or [])
        plan = candidate(world=world,shares=shares,gpus=keep_gpus)
        return dict(gpus=keep_gpus if plan['admitted'] else None,world=world,shares=shares,
                    plan=plan,reason=plan['reason'],alternatives=[])
    if demand.gpu_count:
        from .registry import fleet_decision
        precision = resources['precision']
        if precision not in ('fp32','fp16','bf16') or not demand.local_batches:
            raise ValueError('GPU准入需要实际执行精度及完整全局batch份额')
        minimum = resources.get('min_gpus') if initial else None
        if planner is None and minimum not in (None,demand.gpu_count):
            raise ValueError('自动改变GPU数量需要初始供数规划器')
        config = dict(resources=dict(resources,gpu_count=demand.gpu_count,gpu_ids=demand.gpu_ids,
            min_gpus=minimum,batch_layout=resources.get('batch_layout') if initial else demand.local_batches),
            training=dict(batch=sum(demand.local_batches),precision=precision))
        return fleet_decision(config,resources.get('gpu_spec',{}),history,peaks,peers,
            resources.get('previous_world'),plan_candidate=candidate)
    world = len(demand.local_batches or []) or resources.get('cpu_processes',1)
    shares = list(demand.local_batches or [])
    plan = candidate(world=world,shares=shares,gpus=[])
    return dict(gpus=[] if plan['admitted'] else None,world=world,shares=shares,plan=plan,
                reason=plan['reason'],alternatives=[])
