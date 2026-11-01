"""Stdlib live facts and pure stage resource policies (schema version 2).

CPU slots are time shares, not physical cores. Unknown facts remain explicit.
Policies never benchmark or rank runtime candidates. Pool transport is pipe based.
"""
import copy
import math
import os
from pathlib import Path, PurePosixPath
import time
from stunet_recipe import EPOCHS,TRAIN_ITERATIONS,VALIDATION_ITERATIONS


def _cpulist(text):
    result=set()
    for item in text.strip().split(','):
        if not item:continue
        ends=item.split('-');start=int(ends[0]);end=int(ends[-1])
        result.update(range(start,end+1))
    return result


def snapshot(*, read_text=None, affinity=None, cpu_count=None, sleep=None, sample_seconds=.2):
    """Fresh facts; injectable OS readers permit deterministic Linux fixture tests."""
    read_text=read_text or (lambda path:Path(path).read_text())
    cpu_count=cpu_count or os.cpu_count
    sleep=sleep or time.sleep
    missing=[]; evidence=[]; usage_samples=[]
    def read(path):
        try:return read_text(str(path))
        except (OSError,ValueError,KeyError):return None
    try:
        ids=sorted(affinity() if affinity else os.sched_getaffinity(0))
        affinity_source='sched_getaffinity'
    except (AttributeError,OSError):
        count=cpu_count();ids=list(range(count)) if count is not None else []
        affinity_source='cpu_count_fallback';missing.append('CPU affinity unavailable; CPU count is an upper bound')
    permitted=float(len(ids)) if ids or affinity_source=='sched_getaffinity' else None
    def usage(path, version):
        raw=read(path)
        if raw is None:return None
        try:
            if version==2:return int(dict(line.split() for line in raw.splitlines())['usage_usec'])/1e6
            return int(raw)/1e9
        except (ValueError,KeyError):return None
    limits=[]; memory_remaining=[]; memory_usage_missing=False; cpuset=set(ids)
    membership=read('/proc/self/cgroup'); mountinfo=read('/proc/self/mountinfo')
    if membership is None or mountinfo is None:missing.append('cgroup membership/mount evidence unavailable')
    else:
        members=[]
        for line in membership.splitlines():
            parts=line.split(':',2)
            if len(parts)==3:members.append((set(parts[1].split(',')),parts[2]))
        for line in mountinfo.splitlines():
            try:
                before,after=line.split(' - ',1);left=before.split();right=after.split()
                kind=right[0]
                if kind not in ('cgroup','cgroup2'):continue
                controllers=set(','.join(right[1:]).split(','))
                root=PurePosixPath(left[3]);mount=PurePosixPath(left[4])
                for groups,member in members:
                    if kind=='cgroup2' and groups!={''}:continue
                    if kind=='cgroup' and not (groups & controllers):continue
                    try:relative=PurePosixPath(member).relative_to(root)
                    except ValueError:
                        # A namespaced membership may already be relative to the mount root.
                        if member!='/':continue
                        relative=PurePosixPath('.')
                    current=mount/relative
                    while True:
                        row={'version':2 if kind=='cgroup2' else 1,'path':str(current)}
                        if kind=='cgroup2' or 'cpu' in groups:
                            raw=read(current/('cpu.max' if kind=='cgroup2' else 'cpu.cfs_quota_us'))
                            period=None if kind=='cgroup2' else read(current/'cpu.cfs_period_us')
                            if raw is not None:
                                try:
                                    q,p=raw.split() if kind=='cgroup2' else (raw,period)
                                    if q.strip()!='max' and float(q)>=0 and float(p)>0:
                                        row['cpu_quota']=float(q)/float(p);limits.append(row['cpu_quota'])
                                        usage_path=current/('cpu.stat' if kind=='cgroup2' else 'cpuacct.usage')
                                        used_cpu=usage(usage_path,row['version'])
                                        if used_cpu is None:missing.append(f'quota CPU usage sample unavailable at {current}')
                                        else:usage_samples.append((row,usage_path,used_cpu))
                                except (ValueError,TypeError):missing.append(f'invalid CPU quota at {current}')
                        if kind=='cgroup2' or 'cpuset' in groups:
                            raw=read(current/('cpuset.cpus.effective' if kind=='cgroup2' else 'cpuset.cpus'))
                            effective=raw is not None and kind=='cgroup2'
                            if effective and not raw.strip():cpuset.clear();row['cpuset_cpus']=[]
                            if raw is None and kind=='cgroup2':raw=read(current/'cpuset.cpus')
                            if raw is not None and raw.strip():
                                try:cpuset &= _cpulist(raw);row['cpuset_cpus']=sorted(cpuset)
                                except ValueError:missing.append(f'invalid cpuset at {current}')
                        if kind=='cgroup2' or 'memory' in groups:
                            maximum=read(current/('memory.max' if kind=='cgroup2' else 'memory.limit_in_bytes'))
                            used=read(current/('memory.current' if kind=='cgroup2' else 'memory.usage_in_bytes'))
                            if maximum is not None and maximum.strip()!='max':
                                try:
                                    maximum=int(maximum)
                                    if maximum < 2**60:
                                        if used is None:
                                            missing.append(f'cgroup memory usage unavailable at {current}');memory_usage_missing=True
                                        else:
                                            row['memory_remaining_bytes']=max(0,maximum-int(used));memory_remaining.append(row['memory_remaining_bytes'])
                                except ValueError:missing.append(f'invalid memory limit at {current}')
                        evidence.append(row)
                        if current==mount:break
                        current=current.parent
            except (ValueError,IndexError):missing.append('invalid cgroup mount evidence')
    ids=sorted(cpuset)
    if permitted is not None:permitted=min(float(len(ids)),*limits) if limits else float(len(ids))
    def stat_sample():
        raw=read('/proc/stat');values={}
        if raw is None:return values
        for line in raw.splitlines():
            fields=line.split()
            if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
                try:values[int(fields[0][3:])]=list(map(int,fields[1:9]))
                except ValueError:pass
        return values
    first=stat_sample();idle=None;started=time.monotonic()
    can_sample=bool(ids) and all(i in first for i in ids)
    if can_sample or usage_samples:sleep(sample_seconds)
    elapsed=max(sample_seconds,time.monotonic()-started,1e-9)
    if can_sample:
        second=stat_sample();idle=0.
        for i in ids:
            delta=[b-a for a,b in zip(first[i],second.get(i,[]))]
            if len(delta)<4 or sum(delta)<=0 or min(delta)<0:idle=None;break
            idle+=(delta[3]+(delta[4] if len(delta)>4 else 0))/sum(delta)
    if idle is None:missing.append('permitted-CPU idle sample unavailable; using permitted ceiling, not measured availability')
    available=min(permitted,idle) if permitted is not None and idle is not None else permitted
    quota_remaining=[]
    for row,path,first_usage in usage_samples:
        second_usage=usage(path,row['version'])
        if second_usage is None or second_usage<first_usage:
            missing.append(f'quota CPU usage delta unavailable at {row["path"]}')
        else:
            row['sampled_used_cpu_shares']=(second_usage-first_usage)/elapsed
            row['sampled_remaining_cpu_shares']=max(0,row['cpu_quota']-row['sampled_used_cpu_shares'])
            quota_remaining.append(row['sampled_remaining_cpu_shares'])
    if quota_remaining and available is not None:available=min(available,*quota_remaining)
    cpu_budget=math.floor(.8*available) if available is not None else None
    memory=None;raw=read('/proc/meminfo')
    if raw:
        try:memory=int(dict(line.split(':',1) for line in raw.splitlines())['MemAvailable'].split()[0])*1024
        except (ValueError,KeyError):pass
    if memory is None:missing.append('MemAvailable unavailable')
    memory_values=memory_remaining+([memory] if memory is not None else [])
    memory_available=min(memory_values) if memory_values else None
    if memory_usage_missing:
        memory_available=None;missing.append('finite cgroup RAM limit has unknown remaining capacity')
    if memory_available is None:missing.append('available RAM unknown')
    topology=[];cores=set()
    for i in ids:
        core=read(f'/sys/devices/system/cpu/cpu{i}/topology/core_id')
        socket=read(f'/sys/devices/system/cpu/cpu{i}/topology/physical_package_id')
        if core is not None and socket is not None:
            try:
                pair=(int(socket),int(core));cores.add(pair)
                topology.append(dict(cpu=i,socket=pair[0],core=pair[1]))
            except ValueError:pass
    if len(topology)!=len(ids):missing.append('physical core mapping incomplete')
    nodes={};online=read('/sys/devices/system/node/online')
    if online:
        try:
            for node in sorted(_cpulist(online)):
                raw=read(f'/sys/devices/system/node/node{node}/cpulist')
                if raw is not None:nodes[str(node)]=sorted(set(ids)&_cpulist(raw))
        except ValueError:pass
    if not any(nodes.values()):missing.append('NUMA mapping unavailable')
    pressure=None;raw=read('/proc/pressure/io')
    if raw:
        try:pressure={k:float(v) for k,v in (x.split('=') for x in raw.splitlines()[0].split()[1:])}
        except (ValueError,IndexError):missing.append('invalid I/O pressure sample')
    else:missing.append('I/O pressure unavailable')
    return dict(schema_version=2,collected_at=time.time(),cpu_permitted=permitted,cpu_available=available,
        cpu_budget=cpu_budget,cpu_slots=cpu_budget,cpu_ids=ids,affinity_cpus=ids,affinity_source=affinity_source,
        logical_cpu_count=cpu_count(),physical_core_count=len(cores) if len(topology)==len(ids) and ids else None,
        physical_cores=topology,numa_nodes=nodes,cgroup_cpu_quota=min(limits) if limits else None,
        cpu_idle_fraction=idle/len(ids) if idle is not None and ids else None,
        cpu_availability_source=('permitted_cpu_idle_sample' if idle is not None else 'permitted_ceiling_unmeasured')+('+ancestor_quota_usage' if quota_remaining else ''),
        available_memory_bytes=memory_available,memory_budget_bytes=math.floor(.8*memory_available) if memory_available is not None else None,
        cgroup_evidence=evidence,missing_evidence=missing,io_pressure=pressure,
        shared_memory_available_bytes=None,load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None)


def _budgets(facts):
    cpu=facts.get('cpu_budget')
    if 'cpu_budget' not in facts and facts.get('cpu_slots') is not None:
        # Old snapshots reported availability, not the v2 80% budget.
        cpu=int(facts['cpu_slots']) if facts.get('cpu_slots_are_budgeted') is True else math.floor(.8*facts['cpu_slots'])
    memory=facts.get('memory_budget_bytes')
    if 'memory_budget_bytes' not in facts and facts.get('available_memory_bytes') is not None:
        memory=math.floor(.8*facts['available_memory_bytes'])
    reasons=[]
    if cpu is None:cpu=1;reasons.append('unknown CPU capacity; bounded one-slot fallback')
    if memory is None:reasons.append('unknown RAM; bounded serial fallback')
    return max(0,int(cpu)),memory,reasons


def case_policy(snapshot, case_count, bytes_per_case, supported_threads=4):
    """Direct homogeneous case capacity; read admission is separate from compute."""
    if case_count<0 or bytes_per_case<0 or supported_threads<1:raise ValueError('invalid case capacity input')
    cpu,memory,reasons=_budgets(snapshot)
    workers=min(int(case_count),cpu)
    if memory is None:workers=min(workers,1)
    elif bytes_per_case:workers=min(workers,int(memory)//math.ceil(bytes_per_case))
    status='bounded'
    if not cpu:status='known_insufficient_cpu'
    elif case_count and not workers:status='known_insufficient_memory'
    elif reasons:status='unknown_resources_bounded_fallback'
    threads=min(int(supported_threads),cpu//workers) if workers else 0
    pressure=(snapshot.get('io_pressure') or {}).get('avg10',0)>=10
    if pressure:reasons.append('I/O pressure limits new reads; existing compute allocation retained')
    return dict(workers=workers,threads_per_worker=threads,total_cpu_slots=workers*threads,
        estimated_ram_bytes=workers*math.ceil(bytes_per_case),cpu_budget=cpu,memory_budget_bytes=memory,
        read_concurrency=min(workers,2) if pressure else workers,prefetch=0 if pressure else workers,
        capacity_status=status,reasons=reasons,missing_evidence=list(snapshot.get('missing_evidence',[])),estimated_bytes_per_case=bytes_per_case)


def _rank_setting(cpu, ram, batch, need, rank, shared, private_bytes=None):
    private_bytes=max(2*batch,int(private_bytes or 0))
    main=min(4,cpu)
    thread=2 if cpu>=main+need*2 else 1
    if cpu<main+need:main=max(1,cpu-need)
    maximum=max(0,(cpu-main)//thread)
    val=min(4,max(0,(cpu-main)//2));vt=2
    # Both pools remain resident; private/temporary=2 batches per worker,
    # plus separate bounded prefetch queues and two pinned batches per rank.
    def build(workers, validation):
        resident=(workers+validation)*private_bytes
        base=shared+resident+3*batch
        free=max(0,ram-base) if ram is not None else 0
        q=min(2*workers,max(0,free-validation*batch)//batch) if workers else 0
        free-=q*batch
        vq=min(2*validation,free//batch) if validation else 0
        return resident,base,q,vq,base+(q+vq)*batch
    if ram is None:maximum=0;val=0;main=1
    while val and build(0,val)[1]+val*batch>ram:val-=1
    def supported(validation):
        limit=maximum
        # Reserve both resident pools AND one slot per worker before choosing W.
        while limit and build(limit,validation)[1]+batch*(limit+validation)>ram:limit-=1
        return limit
    # Validation runs in a separate phase. Do not retain four idle validation
    # processes if that prevents the training pool from getting runnable slots.
    while val>1 and supported(val)<min(need,maximum):val-=1
    limit=supported(val)
    workers=min(need,limit)
    # Reconsider the supported two-thread leaf after RAM fixes the actual W.
    # Never trade away workers; keep feedback's limit valid for this thread size.
    if workers and main+workers*2<=cpu:
        thread=2
        limit=min(limit,(cpu-main)//thread)
    resident,base,q,vq,total=build(workers,val)
    return dict(rank=rank,cpu_share=cpu,main_threads=main,worker_threads=thread,
        train_workers=workers,val_workers=val,val_worker_threads=vt,prefetch=q,val_prefetch=vq,
        train_cpu_slots=main+workers*thread,val_cpu_slots=main+val*vt,
        total_cpu_slots=main+max(workers*thread,val*vt),resident_pool_bytes=resident,
        estimated_ram_bytes=total,memory_budget_bytes=ram,source_shared_bytes=shared,
        train_worker_limit=limit,requested_train_workers=need,base_memory_bytes=base,worker_private_bytes=private_bytes)


def training_policy(snapshot, rank_count, batch_bytes, history=None):
    """One direct per-rank configuration; train/validation compute phases alternate."""
    if isinstance(rank_count,bool) or not isinstance(rank_count,int) or rank_count<1:
        raise ValueError('positive rank count required')
    batches=list(batch_bytes) if isinstance(batch_bytes,(tuple,list)) else [batch_bytes]*rank_count
    if (len(batches)!=rank_count or any(isinstance(b,bool) or not isinstance(b,(int,float)) or not math.isfinite(b) or b<=0 for b in batches)):
        raise ValueError('positive rank count and per-rank batch byte estimate required')
    batches=[math.ceil(b) for b in batches]
    cpu,memory,reasons=_budgets(snapshot);batch=max(batches)
    shared=max(0,int(snapshot.get('source_shared_bytes',0)))
    need=16
    if history and history.get('batch_wait_seconds') is not None and history.get('compute_seconds',0)>0:
        need=max(4,4*math.ceil(math.ceil(2*history['batch_wait_seconds']/history['compute_seconds'])/4))
    status='ready'
    if cpu<rank_count:status='blocked';reasons.append('known insufficient CPU budget for one main thread per rank')
    if memory is not None and memory<3*sum(batches)+shared:
        status='blocked';reasons.append('known insufficient RAM for main and pinned batch buffers')
    ranks=[]
    if status!='blocked':
        from batch_distribution import weighted_shares
        shares=weighted_shares(cpu,batches,minimum=1)
        memories=weighted_shares(memory-shared,batches) if memory is not None else [None]*rank_count
        private=snapshot.get('worker_private_bytes')
        privates=list(private) if isinstance(private,(tuple,list)) else [private]*rank_count
        if len(privates)!=rank_count:raise ValueError('one private-memory estimate per rank required')
        for rank in range(rank_count):
            share=shares[rank];ram=memories[rank]
            settings=_rank_setting(share,ram,batches[rank],need,rank,0,privates[rank])
            settings['batch_bytes']=batches[rank]
            ranks.append(settings)
            if settings['train_workers']<need:reasons.append(f'rank {rank}: worker demand bounded by CPU/RAM capacity')
        if any(r.startswith('unknown') for r in reasons):status='bounded_fallback'
    return dict(schema_version=2,status=status,rank_settings=ranks,rank_count=rank_count,
        cpu_budget=cpu,memory_budget_bytes=memory,total_cpu_slots=sum(r['total_cpu_slots'] for r in ranks),
        estimated_ram_bytes=shared+sum(r['estimated_ram_bytes'] for r in ranks),batch_bytes=batch,
        source_shared_bytes=shared,transport='pipe',reasons=reasons,missing_evidence=list(snapshot.get('missing_evidence',[])),
        limits=dict(cpu_budget=cpu,memory_budget_bytes=memory,train_workers=[r['train_worker_limit'] for r in ranks]),
        formal_epochs=EPOCHS,train_iterations_per_epoch=TRAIN_ITERATIONS,validation_iterations_per_epoch=VALIDATION_ITERATIONS)


def feedback_policy(settings, windows, capacity):
    """Observe natural windows; expand only after two sustained waiting windows."""
    if settings['rank_count']!=capacity['rank_count']:
        return dict(action='reconfigure',settings=copy.deepcopy(capacity),reasons=['rank membership changed; use full-epoch checkpoint boundary'],boundary='full_epoch')
    if capacity.get('status')=='blocked':
        return dict(action='block',settings=copy.deepcopy(capacity),reasons=['fresh capacity cannot run the rank group'],boundary='replay_safe_batch_or_full_epoch')
    if (settings['total_cpu_slots']>capacity['cpu_budget'] or
            (capacity.get('memory_budget_bytes') is not None and settings['estimated_ram_bytes']>capacity['memory_budget_bytes'])):
        return dict(action='recover',settings=copy.deepcopy(capacity),reasons=['real fresh CPU/RAM capacity reduction requires bounded recovery'],boundary='replay_safe_batch_or_full_epoch')
    result=copy.deepcopy(settings);reasons=[];action='keep'
    result['cpu_budget']=capacity['cpu_budget']
    result['memory_budget_bytes']=capacity.get('memory_budget_bytes')
    result['missing_evidence']=list(capacity.get('missing_evidence',[]))
    result['limits']=copy.deepcopy(capacity['limits'])
    for rank,row in enumerate(result['rank_settings']):
        row['cpu_share']=capacity['rank_settings'][rank]['cpu_share']
        row['memory_budget_bytes']=capacity['rank_settings'][rank]['memory_budget_bytes']
    recent=windows[-2:]
    eligible=len(recent)==2 and all(w.get('steps',0)>=20 and w.get('seconds',0)>=30 for w in recent)
    if eligible and any((w.get('io_pressure') or {}).get('avg10',0)>=10 for w in recent):
        action='limit_reads';reasons.append('observed host I/O pressure; causal storage/IPC classification unavailable; retain compute pools and reduce speculative submissions')
        for row in result['rank_settings']:
            row['read_concurrency']=min(2,max(1,row['train_workers'],row['val_workers']))
            row['read_admission_scope']='per-rank actual case open/decompression and lazy patch reads; transforms outside slots'
            row['prefetch']=min(row['prefetch'],max(1,row['train_workers']))
            row['val_prefetch']=min(row['val_prefetch'],max(1,row['val_workers']))
            row['estimated_ram_bytes']=row['base_memory_bytes']+(row['prefetch']+row['val_prefetch'])*row.get('batch_bytes',result['batch_bytes'])
        result['estimated_ram_bytes']=result.get('source_shared_bytes',0)+sum(r['estimated_ram_bytes'] for r in result['rank_settings'])
    elif eligible and all(w.get('wait_fraction',0)>.2 for w in recent):
        cpu=capacity['cpu_budget'];ram=capacity.get('memory_budget_bytes');ranks=settings['rank_count']
        targets=[]
        for rank,old in enumerate(settings['rank_settings']):
            limit=capacity['rank_settings'][rank]['train_worker_limit'] if capacity.get('rank_settings') else old['train_workers']
            targets.append(min(limit,math.ceil(1.5*old['train_workers']) if old['train_workers'] else 16))
        # Rebuild using unchanged per-worker/main settings. Capacity from a newer
        # snapshot cannot authorize more threads than its newly reported budget.
        for rank,(old,target) in enumerate(zip(result['rank_settings'],targets)):
            share=capacity['rank_settings'][rank]['cpu_share']
            target=min(target,max(0,(share-old['main_threads'])//old['worker_threads']))
            batch=old.get('batch_bytes',settings['batch_bytes'])
            rank_ram=capacity['rank_settings'][rank]['memory_budget_bytes']
            if rank_ram is not None:
                # Capacity includes both resident pools and at least one queued
                # batch per live pool; prefetch can grow only within the remainder.
                private=old.get('worker_private_bytes',2*batch)
                bound=max(0,(rank_ram-old['val_workers']*private-(3+old['val_workers'])*batch)//(private+batch))
                target=min(target,bound)
            candidate=copy.deepcopy(old);candidate['train_workers']=max(old['train_workers'],target)
            candidate['read_concurrency']=max(1,candidate['train_workers'],candidate['val_workers'])
            delta=candidate['train_workers']-old['train_workers']
            candidate['resident_pool_bytes']+=delta*old.get('worker_private_bytes',2*batch)
            candidate['base_memory_bytes']=candidate['resident_pool_bytes']+3*batch
            free=max(0,rank_ram-candidate['base_memory_bytes']) if rank_ram is not None else 0
            candidate['prefetch']=min(2*candidate['train_workers'],max(0,free-old['val_workers']*batch)//batch)
            free-=candidate['prefetch']*batch
            candidate['val_prefetch']=min(2*old['val_workers'],free//batch)
            candidate['estimated_ram_bytes']=candidate['base_memory_bytes']+(candidate['prefetch']+candidate['val_prefetch'])*batch
            candidate['train_cpu_slots']=old['main_threads']+candidate['train_workers']*old['worker_threads']
            candidate['total_cpu_slots']=max(candidate['train_cpu_slots'],candidate['val_cpu_slots'])
            proposed=result['estimated_ram_bytes']-old['estimated_ram_bytes']+candidate['estimated_ram_bytes']
            live_ok=True
            if settings.get('host_memory_guard'):
                from memory_guard import growth_admitted
                live_ok=growth_admitted(proposed-settings['estimated_ram_bytes'],settings['host_memory_guard'],capacity.get('live_memory',{}))
                if not live_ok and 'live RAM headroom blocks worker expansion' not in reasons:
                    reasons.append('live RAM headroom blocks worker expansion')
            if (live_ok and ram is not None and proposed<=ram and candidate['train_workers']>old['train_workers']
                    and candidate['prefetch']>=candidate['train_workers'] and candidate['val_prefetch']>=candidate['val_workers']):
                result['rank_settings'][rank]=candidate;result['estimated_ram_bytes']=proposed;action='expand'
        result['total_cpu_slots']=sum(r['total_cpu_slots'] for r in result['rank_settings'])
        if action=='expand':reasons.append('two valid high-wait windows; bounded worker expansion at replay-safe boundary')
    if action=='limit_reads' and result['rank_settings']==settings['rank_settings']:
        action='keep';reasons.append('pressure admission already applied; preserve resident pools')
    if (action=='keep' and eligible and all(w.get('io_pressure') is not None and w['io_pressure'].get('avg10',0)<10 for w in recent)
            and any(r.get('read_concurrency',max(1,r['train_workers'],r['val_workers']))<max(1,r['train_workers'],r['val_workers']) for r in result['rank_settings'])):
        action='restore_reads';reasons.append('two pressure-clear windows; restore reads within retained compute allocation')
        for row in result['rank_settings']:row['read_concurrency']=max(1,row['train_workers'],row['val_workers'])
    if not reasons:reasons.append('retain allocation absent sustained wait, real pressure, fault or degradation')
    return dict(action=action,settings=result,reasons=reasons,boundary='replay_safe_batch_or_full_epoch')
