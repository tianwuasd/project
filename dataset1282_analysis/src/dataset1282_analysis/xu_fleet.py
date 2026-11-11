"""Choose GPU groups from the job-visible pool; persist actual membership separately per stage."""
from dataclasses import replace
from itertools import product
from functools import lru_cache
import os
import time

from .contracts import atomic_json, digest, read_json
from .xu_resources import ResourceUnavailable, GpuResourcesChanged, stage_resources

TRAIN_3D = {'crop64','crop128','all64','all128','init64','init128','auto-init64','auto-init128'}
INFERENCE = {'localize','evaluate-all64','evaluate-all128','four-way','six-way','auto-six-way','with-blood'}
CPU_STAGES = {'audit-roi','targets','auto-targets','vessels'}


def preflight_sizes(stage):
    if stage in TRAIN_3D:
        return ('3d'+('128' if stage.endswith('128') else '64'),)
    return ('blood-encoder','blood-sequence') if stage == 'blood' else ()


@lru_cache(maxsize=256)
def needs_for_width(stage, width, profile):
    if profile != 'paper':
        return [0] * width
    from .xu_capacity import device_demands, specification
    if stage in INFERENCE:
        # Batch-one inference: resident weights plus a conservative allowance for full-resolution maps.
        sizes = ['3d64'] if stage == 'evaluate-all64' else ['3d128']
        if stage == 'localize':
            sizes.append('3d64')  # Both frozen locators reside on each inference card.
        specs = [specification(s) for s in sizes]
        weights = sum(s['trainable_parameters']*4 for s in specs)
        side = specs[0]['shape'][-1]
        need = weights + 6*side**3*(64 if side==64 else 48)*4
        if stage == 'with-blood':
            blood = specification('blood-encoder')
            need = max(need, (blood['trainable_parameters']+blood['other_resident_parameters'])*4
                       + 512*512*64*4*16)
        return [need]*width
    names = [f'cuda:{i}' for i in range(width)]
    needs = [0] * width
    for size in preflight_sizes(stage):
        active = names[:4] if size != 'blood-sequence' else names[:7]
        demand = device_demands(size, active, specification(size))
        for i, device in enumerate(active):
            needs[i] = max(needs[i], demand[device])
    return needs


def work_limits(name, args, cache, pool_size):
    if name in TRAIN_3D:
        return (4 if args.profile == 'paper' else 1), 0
    if name == 'blood':
        return 1, 0
    if name in INFERENCE:
        role = 'development'
        cases = list(cache.case_ids(role))
        if name == 'localize':
            cases.extend(cache.case_ids('train'))
        per_card = 0
        if pool_size > 1 and cases:
            from .xu_inference_pool import inference_lane_bytes
            memo = getattr(args,'inference_memory',None)
            if memo is None:
                memo = args.inference_memory = {}
            key = 'all' if name == 'localize' else role
            if key not in memo:
                memo[key] = inference_lane_bytes((cache.geometry(case) for case in cases),
                                                  evaluation=name != 'localize')
            per_card = memo[key]
        return max(1,len(cases)), per_card
    raise ValueError(f'Unknown GPU stage: {name}')


def distribute(names, caps, free, needs):
    """Largest feasible total first. Preserve whole-batch counts; satisfy largest slots first."""
    widths = [range(1,min(caps[n],len(free))+1) for n in names]
    layouts = sorted((x for x in product(*widths) if sum(x)<=len(free)),
                     key=lambda x:(sum(x),min(x),x[-1]), reverse=True)
    for layout in layouts:
        groups = {name:[None]*w for name,w in zip(names,layout)}
        slots = [(need,name,i) for name,w in zip(names,layout)
                 for i,need in enumerate(needs(name,w))]
        remaining = dict(free)
        for need,name,index in sorted(slots,key=lambda item:-item[0]):
            eligible = [card for card,bytes_ in remaining.items() if bytes_>=need]
            if not eligible:
                break
            # Use the strongest remaining card for the heaviest slot.
            card = max(eligible, key=remaining.get) if need and name in TRAIN_3D | {'blood'} else eligible[0]
            groups[name][index] = card
            remaining.pop(card)
        else:
            return groups
    return None


def available(pool, args, children):
    from .xu_gpu_probe import gpu_snapshot, visible_devices
    from .xu_resources import _check_device
    inventory = gpu_snapshot()
    visible = {row['device']: row for row in visible_devices(inventory)}
    owned = children.owned_identities()
    free, rejected = {}, {}
    for device in pool:
        try:
            card = visible[device]
            info = {**inventory[card['uuid']], **card}
            checked = _check_device(info, args.resource_policy, owned,
                                    args.resource_policy == 'v2' and os.name == 'posix')
            free[device] = checked['allocator_budget_bytes']
        except ResourceUnavailable as exc:
            rejected[device] = str(exc)
    return free, rejected


def _with_primary(stage, group):
    argv = list(stage.argv)
    if '--device' in argv and group:
        argv[argv.index('--device') + 1] = group[0]
    return replace(stage,argv=tuple(argv))


def assign_wave(wave, args, out, cache, children, contract_id):
    """Read committed assignments without resource probing; reselect between fresh stages."""
    pool = args.device_names
    if pool == ['cpu']:
        return [(tuple((s,['cpu']) for s in wave))]
    fixed, fresh, live_fixed = {}, [], set()
    def file(stage):
        return out/'assignments'/(stage.name+'.json')
    for stage in wave:
        durable = (out/'started'/(stage.name+'.json')).is_file() or (out/'receipts'/(stage.name+'.json')).is_file()
        if durable:
            if not file(stage).is_file():
                raise ValueError(f'Missing GPU assignment for {stage.name}')
            row = read_json(file(stage))
            group = row.get('devices', [])
            if (row.get('signature') != digest([contract_id,stage.name,group]) or len(set(group))!=len(group)
                    or any(d not in [*pool,'cpu'] for d in group)):
                raise ValueError(f'Invalid GPU assignment for {stage.name}')
            if ((getattr(args,'reallocate',False) or (stage.name == 'blood' and len(group) > 1))
                    and not stage.skip_reason
                    and stage.name not in CPU_STAGES
                    and not (out/'receipts'/(stage.name+'.json')).exists()):
                fresh.append(stage.name)
            else:
                fixed[stage.name] = group
                if not stage.skip_reason and not (out/'receipts'/(stage.name+'.json')).exists():
                    live_fixed.update(group)
        elif stage.skip_reason or stage.name in CPU_STAGES:
            fixed[stage.name] = ['cpu'] if stage.name in CPU_STAGES else []
        else:
            fresh.append(stage.name)

    rejected = {}
    free = {}
    groups = dict(fixed)
    serial = False
    caps, memory = {}, {}
    if fresh:
        limits = {name:work_limits(name,args,cache,len(pool)) for name in fresh}
        deadline = time.monotonic()+args.wait_seconds
        memo = {}
        def needs(n,w):
            if (n,w) not in memo:
                memo[n,w] = needs_for_width(n,w,args.profile)
            return memo[n,w]
        while True:
            try:
                free,rejected = available(pool,args,children)
                if not free:
                    raise GpuResourcesChanged(f'No idle admitted GPU: {rejected}',pool)
                host_plan = stage_resources(['cpu'],policy=args.resource_policy)
                host = host_plan['host_budget_bytes']
                memory = {n:m for n,(_,m) in limits.items()}
                minimum_host = 0
                if args.profile == 'paper':
                    from .xu_capacity import specification
                    minimum_host = max((specification(s)['estimated_host_bytes']
                                        for name in fresh for s in preflight_sizes(name)),default=0)
                from .xu_supply_budget import stage_minimum, GIB
                minimum_supply = max(stage_minimum(args,name) for name in fresh)
                if minimum_supply:
                    minimum_host = (minimum_host+minimum_supply+GIB)*5//4
                def capacities(lanes):
                    return {n:min(work,host//lanes//m if m else work) for n,(work,m) in limits.items()}
                caps = capacities(2 if len(fresh)==2 or live_fixed else 1)
                unclaimed = {k:v for k,v in free.items() if k not in live_fixed}
                parallel_host = (host_plan.get('cpu_slots',2) >= 2 and host//2 >= minimum_host)
                assigned = (distribute(fresh,caps,unclaimed,needs)
                            if parallel_host or (len(fresh)==1 and not live_fixed) else None)
                if assigned:
                    break
                # A single usable device or asymmetrical capacities can require serial branches.
                serial = True
                caps = capacities(1)
                assigned = {}
                for name in fresh:
                    choice = distribute([name],caps,free,needs)
                    if not choice:
                        raise ResourceUnavailable(f'No feasible fixed-workload layout for {name}; '
                                                  f'eligible cards={free}, working slots={caps[name]}, excluded={rejected}')
                    assigned.update(choice)
                break
            except ResourceUnavailable as exc:
                if time.monotonic() >= deadline:
                    atomic_json(out/'resources'/f'fleet-denied-{time.time_ns()}.json',dict(reason=str(exc),stages=fresh))
                    raise
                time.sleep(min(2,max(0,deadline-time.monotonic())))
        groups.update(assigned)

    # Only simultaneously unfinished jobs require disjoint devices. A verified receipt needs no card.
    active_sets = [set(groups[s.name]) for s in wave
                   if not s.skip_reason and not (out/'receipts'/(s.name+'.json')).exists()]
    overlap = len(active_sets)==2 and bool(active_sets[0] & active_sets[1])
    output = []
    for stage in wave:
        group = groups[stage.name]
        if (not (out/'receipts'/(stage.name+'.json')).exists()
                and (not (out/'started'/(stage.name+'.json')).exists() or stage.name in fresh)):
            history = []
            if file(stage).exists():
                previous = read_json(file(stage))
                history = [*previous.get('history',[]), {k:v for k,v in previous.items() if k != 'history'}]
            atomic_json(file(stage), dict(stage=stage.name,devices=group,authorized=pool,
                        not_assigned=[d for d in pool if d not in group],unavailable=rejected,
                        idle_budget_bytes=free,working_slots=caps.get(stage.name),
                        host_per_inference_case=memory.get(stage.name), signature=digest([contract_id,stage.name,group]),
                        history=history, assigned_at=time.time()))
        output.append((_with_primary(stage,group),group))
    return [(entry,) for entry in output] if overlap or serial else [tuple(output)]
