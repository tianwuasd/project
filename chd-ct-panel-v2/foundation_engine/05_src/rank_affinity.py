"""Current Linux PCI/NUMA affinity, respecting inherited allocation and CPU quota."""
import os
from pathlib import Path


def parse_cpulist(value):
    result=set()
    for token in value.strip().split(','):
        if not token:continue
        parts=token.split('-')
        first=int(parts[0]);last=int(parts[-1])
        if first<0 or last<first:raise ValueError('无效CPU范围')
        result.update(range(first,last+1))
    return result


def required_slots(row):
    main=int(row.get('main_threads',row.get('threads',1)))
    train=int(row.get('train_workers',row.get('augmentation_workers',0)))*int(row.get('worker_threads',1))
    val=int(row.get('val_workers',row.get('validation_workers',0)))*int(row.get('val_worker_threads',row.get('worker_threads',1)))
    return main+max(train,val)


def physical_first(cpus, facts):
    by_cpu={r['cpu']:(r['socket'],r['core']) for r in facts}
    first=[];siblings=[];seen=set()
    for cpu in sorted(cpus):
        key=by_cpu.get(cpu,('unknown',cpu))
        (siblings if key in seen else first).append(cpu);seen.add(key)
    return first+siblings


def placement(gpus,resource,inherited,pools):
    rows=resource.get('rank_settings',[resource for _ in gpus])
    needs=[required_slots(r) for r in rows]
    allowed=set(resource.get('original_cpu_affinity') or inherited)
    # Saved original process allocation permits later expansion after a prior
    # narrower bind. The stage snapshot already intersected ancestor cpusets.
    facts=resource.get('allocation_snapshot',{}).get('physical_cores',[])
    if sum(needs)>min(len(allowed),resource.get('cpu_budget',len(allowed))):
        raise ValueError('aggregate active rank CPU demand exceeds allocation')
    assignments=[];used=set()
    for gpu,need in zip(gpus,needs):
        preferred=pools.get(gpu,(None,set()))[1]&allowed-used
        choices=physical_first(preferred,facts)+physical_first(allowed-used-preferred,facts)
        chosen=choices[:need];assignments.append(chosen);used.update(chosen)
    return assignments


def bind(rank,gpus,resource):
    if not hasattr(os,'sched_getaffinity'):
        return dict(status='unavailable',reason='Linux affinity API unavailable')
    inherited=set(os.sched_getaffinity(0))
    from evaluation_gpu import query
    pools={};missing=[]
    try:
        buses={uuid:bus.lower() for uuid,bus in query('uuid,pci.bus_id')}
        for uuid in gpus:
            bus=buses[uuid];domain,rest=bus.split(':',1)
            node=int((Path('/sys/bus/pci/devices')/f'{int(domain,16):04x}:{rest}'/'numa_node').read_text())
            pool=parse_cpulist((Path('/sys/devices/system/node')/f'node{node}'/'cpulist').read_text())
            if node<0 or not pool:raise ValueError('GPU local NUMA unavailable')
            pools[uuid]=(node,pool)
    except (OSError,ValueError,KeyError) as error:
        missing.append(str(error))
        if not resource.get('original_cpu_affinity'):
            return dict(status='unavailable',reason=str(error),inherited_cpu_affinity=sorted(inherited))
    try:
        chosen=placement(gpus,resource,inherited,pools)[rank]
        # Linux affinity is per native TID. Cover existing BLAS/Torch threads,
        # and children inherit the bounded parent thread mask at spawn.
        tids=[int(p.name) for p in Path('/proc/self/task').iterdir() if p.name.isdigit()]
        failures=[];bound=[]
        for tid in tids:
            try:os.sched_setaffinity(tid,chosen);bound.append(tid)
            except OSError as error:failures.append(dict(tid=tid,error=str(error)))
        local=pools.get(gpus[rank],(None,set()))
        return dict(status='partial' if failures else 'bound',cpu_affinity=chosen,
            native_tids_bound=bound,native_tid_failures=failures,missing_evidence=missing,
            numa_node=local[0],numa_local=set(chosen)<=local[1],
            original_cpu_affinity=resource.get('original_cpu_affinity',sorted(inherited)),
            requested_cpu_slots=required_slots(resource.get('rank_settings',[resource])[rank if 'rank_settings' in resource else 0]),
            memory_policy='first-touch only; no privileged memory binding')
    except (OSError,ValueError,KeyError) as error:
        return dict(status='unavailable',reason=str(error),inherited_cpu_affinity=sorted(inherited))
