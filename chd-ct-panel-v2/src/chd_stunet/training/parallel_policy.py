"""Whole-group V2 budgets and natural-epoch expansion decisions, without CUDA."""
import math
import statistics
from ..acceleration.preprocess_cpu import GIB,policy
from ..acceleration.training_cpu import serial_estimate,worker_estimate


def layout(batch,world):
    if type(batch) is not int or type(world) is not int or not 1<=world<=batch:
        raise ValueError('Ranks must be nonempty and no more than global batch')
    return [batch//world+int(rank<batch%world) for rank in range(world)]


def slots_for(batch,world,rank):
    shares=layout(batch,world)
    if not 0<=rank<world:raise ValueError('Invalid rank')
    start=sum(shares[:rank])
    return list(range(start,start+shares[rank]))


def supply_plan(facts,settings,loaders,world,maximum=None,reserve_at_entry=None):
    """Allocate 80% ONCE, before ranks spawn; --workers is a GROUP upper cap.

    Reserve one supervisor/rank CPU slot and 2 GiB per rank for imports, model,
    optimizer and checkpoint staging, plus serial/pinned input working sets.
    Worker priors already include queued outputs and augmented initial crops.
    This is admission, not an OS hard limit; the live guard remains required.
    """
    shares=layout(settings['batch_size'],world)
    stage=policy(facts,[dict(estimate=1)]*max(1,int(facts.get('cpu_budget',0))))
    if reserve_at_entry is not None:
        stage=dict(stage,reserve_bytes=max(stage['reserve_bytes'],reserve_at_entry))
        stage['memory_budget_bytes']=max(0,min(stage['memory_budget_bytes'],
            stage['available_memory_bytes']-stage['reserve_bytes']))
    local=[dict(settings,batch_size=n) for n in shares]
    resident=[2*GIB+serial_estimate(s) for s in local]
    estimates=[worker_estimate(s,loaders) for s in local]
    spare_cpu=stage['cpu_budget']-world-1
    spare_ram=stage['memory_budget_bytes']-sum(resident)-GIB
    if spare_cpu<0 or spare_ram<0:
        raise MemoryError('Whole-group rank/serial input admission exceeds V2 CPU/RAM budget')
    # Start enough persistent workers to overlap supply with GPU work. Trial
    # more only after observing queue wait instead of filling spare RAM at boot.
    cap=min(2*world, spare_cpu if maximum is None else min(maximum,spare_cpu))
    workers=[0]*world
    # Fair rounds give every resident worker a dispatch slot; no pool is
    # created merely to leave its workers permanently waiting behind a queue.
    while sum(workers)<cap:
        candidates=[i for i in range(world) if estimates[i]<=spare_ram and workers[i]<settings['train_steps_per_epoch']]
        if not candidates:break
        rank=min(candidates,key=lambda i:(workers[i]/shares[i],i))
        workers[rank]+=1;spare_ram-=estimates[rank]
    ranks=[dict(stage,workers=w,cpu_budget=w+1,
                memory_budget_bytes=resident[i]+w*estimates[i],
                requested_workers=w,source='one V2 whole-group budget, partitioned per rank')
           for i,w in enumerate(workers)]
    return dict(stage=stage,layout=shares,workers=workers,ranks=ranks,
        reserved_rank_bytes=resident,worker_estimate_bytes=estimates,
        unused_memory_budget_bytes=spare_ram,requested_total_workers=maximum,
        applied_total_workers=sum(workers),supervisor_and_rank_cpu_slots=world+1)


def expansion_baseline(rows,batch,members):
    rows=[r for r in rows if r.get('members')==members and r.get('global_batch')==batch]
    if len(rows)<4:return None
    recent=rows[-3:]
    if any(not _valid(r.get('seconds')) for r in recent):return None
    return dict(reference_seconds=statistics.median(r['seconds'] for r in recent),
        global_batch=batch,previous_members=members,baseline_epochs=[r['epoch'] for r in recent])


def expansion_regressed(baseline,rows,batch):
    if not baseline or baseline['global_batch']!=batch or len(rows)<4:return False
    # First completed epoch is warmup. Require the THREE latest observations
    # after it, not a cumulative mean including an unrelated group.
    return all(_valid(r.get('seconds')) and r['global_batch']==batch
               and r['seconds']>1.25*baseline['reference_seconds'] for r in rows[-3:])


def _valid(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0
