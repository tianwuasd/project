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


def bind(rank,gpus,resource):
    if not hasattr(os,'sched_getaffinity'):
        return dict(status='unavailable',reason='Linux affinity API unavailable')
    inherited=set(os.sched_getaffinity(0))
    from evaluation_gpu import query
    try:
        buses={uuid:bus.lower() for uuid,bus in query('uuid,pci.bus_id')}
        pools={}
        for uuid in gpus:
            # nvidia-smi uses an eight-digit PCI domain; Linux sysfs uses four.
            bus=buses[uuid];domain,rest=bus.split(':',1)
            node=int((Path('/sys/bus/pci/devices')/f'{int(domain,16):04x}:{rest}'/'numa_node').read_text())
            pool=parse_cpulist((Path('/sys/devices/system/node')/f'node{node}'/'cpulist').read_text())&inherited
            if node<0 or not pool:raise ValueError('GPU本地NUMA可用CPU未知')
            pools[uuid]=(node,pool)
        node,pool=pools[gpus[rank]]
        peers=[g for g in gpus if pools[g][0]==node]
        local_rank=peers.index(gpus[rank]);partition=sorted(pool)[local_rank::len(peers)]
        needed=resource['threads']+resource.get('augmentation_workers',0)+resource.get('validation_workers',0)
        if len(partition)<needed:raise ValueError('本地NUMA分区不足以容纳已冻结主线程和增强worker')
        chosen=partition[:needed]
        os.sched_setaffinity(0,chosen)
        return dict(status='bound',numa_node=node,cpu_affinity=chosen,
            inherited_count=len(inherited),memory_policy='first-touch only; no privileged memory binding')
    except (OSError,ValueError,KeyError) as error:
        return dict(status='unavailable',reason=str(error),inherited_cpu_affinity=sorted(inherited),
            policy='retain inherited allocation; do not fabricate a NUMA map')
