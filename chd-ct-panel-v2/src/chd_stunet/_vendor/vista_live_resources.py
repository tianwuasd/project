"""Stdlib live facts and pure stage resource policies (schema version 2).

CPU slots are time shares, not physical cores. Unknown facts remain explicit.
Policies never benchmark or rank runtime candidates. Pool transport is pipe based.
"""
import copy
import math
import os
from pathlib import Path, PurePosixPath
import time


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


