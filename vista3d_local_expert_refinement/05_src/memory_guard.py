"""Live host/cgroup headroom. Estimates and samples are not hard RAM limits."""
import math
import os
from pathlib import Path, PurePosixPath
import time

GIB=2**30


class HostMemoryPressureError(RuntimeError):
    classification='cpu_memory_pressure'
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('主机内存保护触发；停止本组，保留最近完整检查点；'+str(evidence.get('reason')))


def _read(path):
    try:return Path(path).read_text()
    except OSError:return None


def _groups(read):
    """Resolve the current cgroup namespace against its actual mount root."""
    memberships=(read('/proc/self/cgroup') or '').splitlines()
    mounts=(read('/proc/self/mountinfo') or '').splitlines()
    result=[]
    for line in memberships:
        try:_,controllers,member=line.split(':',2)
        except ValueError:continue
        for mount in mounts:
            try:
                before,after=mount.split(' - ',1);fields=before.split();kind=after.split()[0]
                if not (kind=='cgroup2' and not controllers or kind=='cgroup' and 'memory' in controllers.split(',') and 'memory' in after.split()[2].split(',')):continue
                root=PurePosixPath(fields[3]);point=Path(fields[4])
                relative=PurePosixPath(member).relative_to(root)
                if '..' in relative.parts:continue
                current=point.joinpath(*relative.parts)
                while True:
                    row=(str(current),2 if kind=='cgroup2' else 1)
                    if row not in result:result.append(row)
                    if current==point:break
                    current=current.parent
            except (ValueError,IndexError):continue
    return result


def memory_snapshot(*,read_text=None,groups=None):
    read=read_text or _read; missing=[]
    raw=read('/proc/meminfo'); host=None;total=None
    try:
        values={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in (raw or '').splitlines())}
        host=values.get('MemAvailable');total=values.get('MemTotal')
    except (ValueError,IndexError):missing.append('invalid /proc/meminfo')
    if host is None and os.name=='nt' and read_text is None:
        try:
            import psutil
            mem=psutil.virtual_memory();host=mem.available;total=mem.total
        except (ImportError,OSError):pass
    available=[host] if host is not None else []; rows=[];unknown=False
    for path,version in (_groups(read) if groups is None else groups):
        root=PurePosixPath(path);limit=read(root/('memory.max' if version==2 else 'memory.limit_in_bytes'))
        used=read(root/('memory.current' if version==2 else 'memory.usage_in_bytes'))
        row=dict(path=str(root),version=version,limit_bytes=None,current_bytes=None,events={})
        try:
            row['current_bytes']=int(used) if used is not None else None
            if limit is not None and limit.strip()!='max' and int(limit)<2**60:
                row['limit_bytes']=int(limit)
                if row['current_bytes'] is None:unknown=True;missing.append('finite cgroup usage unavailable: '+str(root))
                else:available.append(max(0,row['limit_bytes']-row['current_bytes']))
            for name in ('memory.events','memory.events.local') if version==2 else ('memory.failcnt',):
                text=read(root/name)
                if text:
                    row[name]={k:int(v) for k,v in (line.split() for line in text.splitlines())} if version==2 else {'failcnt':int(text)}
            row['events']=row.get('memory.events',row.get('memory.failcnt',{}))
        except (ValueError,TypeError):unknown=True;missing.append('invalid cgroup memory evidence: '+str(root))
        rows.append(row)
    if host is None:missing.append('host available RAM unknown');unknown=True
    return dict(time=time.time(),available_bytes=min(available) if available and not unknown else None,
                host_available_bytes=host,host_total_bytes=total,cgroups=rows,missing_evidence=missing,
                scope='host/cgroup remaining capacity; not attribution to this task')


def guard_policy(allocation):
    initial=allocation.get('allocation_snapshot',allocation)
    available=initial.get('available_memory_bytes')
    return dict(reserve_bytes=max(4*GIB,math.ceil(.2*available)) if available is not None else 4*GIB,
                poll_seconds=1,source='20% of stage-admitted available RAM, minimum4GiB',
                groups=[(r['path'],r['version']) for r in initial.get('cgroup_evidence',[])])


def growth_admitted(extra_bytes,policy,snapshot):
    available=snapshot.get('available_bytes')
    limits={r['path']:r.get('limit_bytes') for r in snapshot.get('cgroups',[])}
    if any(limits.get(path)!=limit for path,limit in policy.get('known_limits',{}).items()):return False
    return available is not None and max(0,extra_bytes)<=available-policy['reserve_bytes']


def process_tree_evidence(pid=None):
    """PSS/USS where permitted, RSS separately; no double-counted RSS budget."""
    try:
        import psutil
        root=psutil.Process(pid or os.getpid()); rows=[];missing=[]
        for process in [root,*root.children(recursive=True)]:
            try:
                info=process.memory_full_info()
                rows.append(dict(pid=process.pid,created=process.create_time(),rss_bytes=info.rss,
                                 pss_bytes=getattr(info,'pss',None),uss_bytes=getattr(info,'uss',None)))
            except (psutil.Error,OSError):missing.append(process.pid)
        return dict(processes=rows,missing_pids=missing,
                    summed_pss_bytes=sum(r['pss_bytes'] for r in rows) if rows and all(r['pss_bytes'] is not None for r in rows) else None,
                    summed_uss_bytes=sum(r['uss_bytes'] for r in rows) if rows and all(r['uss_bytes'] is not None for r in rows) else None)
    except Exception as error:return dict(processes=[],missing_evidence=['process-tree PSS/USS unavailable: '+type(error).__name__])


class LiveMemoryGuard:
    def __init__(self,work,allocation,*,reader=None,clock=time.monotonic):
        self.work=Path(work);self.policy=allocation.get('host_memory_guard') or guard_policy(allocation)
        groups=self.policy.get('groups')
        self.reader=reader or (lambda:memory_snapshot(groups=groups or None))
        self.clock=clock;self.last_check=-float('inf');self.last_record=-float('inf')
        self.baseline=self.reader();self.latest=self.baseline
        self.policy['known_limits']={r['path']:r['limit_bytes'] for r in self.baseline.get('cgroups',[]) if r.get('limit_bytes') is not None}

    def check(self,*,force=False):
        tick=self.clock()
        if not force and tick-self.last_check<self.policy.get('poll_seconds',1):return
        self.last_check=tick;self.latest=self.reader()
        available=self.latest.get('available_bytes');reserve=self.policy['reserve_bytes']
        reason='available RAM unknown' if available is None else 'live remaining RAM below reserve' if available<reserve else None
        if not growth_admitted(0,self.policy,self.latest) and reason is None:reason='previously known cgroup memory limit changed or disappeared'
        evidence=dict(snapshot=self.latest,reserve_bytes=reserve,reason=reason,
                      initial_cgroups=self.baseline.get('cgroups',[]),guarantee='sampled protection; abrupt external allocation can outrun it')
        if reason or tick-self.last_record>=30:
            from server_data import write_json
            from runtime_observations import event
            evidence['task_process_tree']=process_tree_evidence()
            write_json(self.work/'diagnostics/memory-guard-latest.json',evidence)
            logged=evidence if reason else dict(evidence,task_process_tree={k:v for k,v in evidence['task_process_tree'].items() if k!='processes'},
                task_process_count=len(evidence['task_process_tree'].get('processes',[])))
            event(self.work,'memory.pressure' if reason else 'memory.guard_sample',**logged)
            self.last_record=tick
        if reason:
            available_text='未知' if available is None else f'{available/GIB:.1f}GiB'
            print(f'主机内存保护：当前可用{available_text}，预留{reserve/GIB:.1f}GiB；停止本组并保留最近完整检查点。',flush=True)
            raise HostMemoryPressureError(evidence)


def smaller_host_budget(resource):
    """Reduce the retained host allocation, so feedback cannot regrow old pools."""
    values=[resource.get('memory_budget_bytes'),resource.get('estimated_ram_bytes')]
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        raise HostMemoryPressureError(dict(reason='RAM budget/estimate unavailable; automatic adjustment refused'))
    return max(1,int(.6*min(values)))


def cleanup_owned_segments(invocation,owners):
    """Run after owned ranks/workers exit; never enumerate global shared memory."""
    import json
    import re
    from multiprocessing.shared_memory import SharedMemory
    root=Path(invocation);removed=[]
    pattern=re.compile(r'vista_'+re.escape(root.name)+r'_[0-9a-f]{32}')
    for path in (root/'shared-memory').glob('*.json'):
        if path.is_symlink():raise ValueError('shared-memory ownership record is a symlink')
        row=json.loads(path.read_text(encoding='utf-8'));owner=row.get('rank_owner') or {}
        if not (pattern.fullmatch(row.get('name','')) and owner.get('identity')
                and owners.get(owner.get('pid'))==owner['identity']):continue
        try:segment=SharedMemory(name=row['name'])
        except FileNotFoundError:continue
        try:segment.unlink();removed.append(row['name'])
        finally:segment.close()
    return removed
