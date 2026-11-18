from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from copy import deepcopy
import multiprocessing as mp
import os
from pathlib import Path
from threading import Event, RLock
from time import monotonic, sleep

from filelock import FileLock
import psutil

from ..artifacts.store import read_json, write_json
from ..observability.events import event
from .client import ResourceClient
from .contracts import Allocation, ResourceHandle, ResourceDemand, SafeBoundary
from .host import host_capacity
from .memory import identity, alive, system_headroom
from .timing import TimingBook
from .accounting import reconcile
from .admission import select_admission


def attach(handle):
    member = identity()
    member['baseline_rss'] = psutil.Process().memory_info().rss
    with handle.condition:
        previous = handle.members.get(member['pid'])
        if previous is None or previous['birth'] != member['birth']:
            handle.members[member['pid']] = member
    return ResourceClient(handle)


class ResourceSession:
    def __init__(self, request, *, context=None, cancel=None, layout=None, planner=None):
        self.request = request
        self.context = context or mp.get_context('spawn')
        self.cancel = cancel or self.context.Event()
        self.stopping, self.timing = Event(), TimingBook()
        self.failure = None
        self.control_lock = RLock()
        self.requested_layout = self._layout(layout)
        self.pending_token = None
        self.planner = planner

    @staticmethod
    def _layout(layout):
        value = deepcopy(layout or {})
        workers = value.setdefault('workers',[0])
        if not workers or any(type(n) is not int or n < 0 for n in workers):
            raise ValueError('供数worker布局必须为非空非负整数列表')
        if value.setdefault('prefetch',2) not in (1,2):
            raise ValueError('供数预取容量必须为1或2')
        return value

    def _admit(self, demand, *, layout=None, keep_gpus=None, initial=False):
        root = Path(self.request.registry_root)
        root.mkdir(parents=True, exist_ok=True)
        began = monotonic()
        last_log = -float('inf')
        history = []
        while True:
            if self.cancel.is_set():
                raise InterruptedError('资源准入已取消')
            if self.pending_token is not None:
                with self.condition:
                    pending_request = self.state.get('replan_request')
                    if (pending_request is None or pending_request['token'] != self.pending_token or
                            pending_request['abandoned']):
                        raise InterruptedError('资源重规划请求已经结束等待')
            facts = host_capacity()
            r = self.request.resources
            observed = None
            if demand.gpu_count and keep_gpus is None:
                from .registry import inventory
                owned = [self.owner['pid']]
                if hasattr(self, 'members'):
                    owned.extend(m['pid'] for m in self.members.values() if alive(m))
                observed = inventory(owned_pids=owned)
                history.append((monotonic(), observed))
                while len(history)>1 and history[-1][0]-history[1][0] >= r.get('share_window_seconds',30):
                    history.pop(0)
            headroom, _ = system_headroom(facts, min(.8, r.get('ram_fraction', .8)), r.get('reserve_ram_bytes', 4*2**30))
            with self.control_lock, FileLock(str(root / '.resource.lock')):
                path = root / 'active.json'
                rows = [row for row in (read_json(path) if path.exists() else []) if alive(row)]
                peers = [row for row in rows if row['pid'] != self.owner['pid'] or row['birth'] != self.owner['birth']]
                limit = min(100_000_000_000, r.get('memory_limit_bytes', 100_000_000_000))
                project = sum(max(row.get('ram_bytes', 0), row.get('rss_bytes', 0)) for row in peers)
                pending = sum(max(0, row.get('ram_bytes', 0)-row.get('rss_bytes', 0)) for row in peers)
                current = psutil.Process().memory_info().rss
                if hasattr(self, 'members'):
                    current = sum(psutil.Process(m['pid']).memory_info().rss
                                  for m in self.members.values() if alive(m))
                occupied = current
                if hasattr(self,'condition'):
                    with self.condition:
                        occupied += self.state.get('unrealized_bytes',0)
                cpu_available = max(0, int(facts['cores']*r.get('cpu_fraction', .8))-
                                    sum(row.get('cpu_slots', 0) for row in peers))
                budget = min(limit-project,current+headroom-pending)
                peak_path = root/'gpu-peaks.json'
                peaks = read_json(peak_path) if observed is not None and peak_path.is_file() else {}
                decision = select_admission(self.request,demand,self.planner if initial else None,
                    facts=facts,memory_budget_bytes=budget,cpu_budget_slots=cpu_available,
                    occupied_bytes=occupied,history=history,peaks=peaks,peers=peers,
                    layout=layout if layout is not None else self.requested_layout,
                    keep_gpus=keep_gpus,initial=initial)
                if decision['gpus'] is not None:
                    plan = decision['plan']
                    selected,amount,cpu_slots = plan['demand'],plan['memory_bytes'],plan['cpu_slots']
                    gpus,world,shares = decision['gpus'],decision['world'],decision['shares']
                    applied_layout = self._layout(plan['layout'])
                    generation = self.state['plan_generation']+1 if hasattr(self,'state') else 0
                    rows = peers + [{**self.owner, 'members': list(self.members.values()) if hasattr(self,'members') else [],
                        'ram_bytes': amount, 'rss_bytes': current, 'cpu_slots': cpu_slots,
                        'uuids': [g['uuid'] for g in gpus], 'memory_limit_bytes': limit,
                        'layout': applied_layout, 'reader_slots': selected.reader_slots,
                        'workspace_capacity': selected.workspace_bytes if selected.reader_slots else 0,
                        'plan_generation': generation,'world':world,'shares':shares,'precision':r.get('precision')}]
                    if hasattr(self,'condition'):
                        with self.condition:
                            write_json(path,rows,durable=False)
                            self.capacity,self.cpu_slots,self.gpus = amount,cpu_slots,gpus
                            self.rows = rows
                            self.state.update(capacity=amount,reader_slots=selected.reader_slots,
                                workspace_capacity=selected.workspace_bytes if selected.reader_slots else 0,
                                layout=applied_layout,plan_generation=generation,cpu_slots=cpu_slots,gpus=gpus,
                                world=world,shares=shares,precision=r.get('precision'))
                            self.condition.notify_all()
                    else:
                        write_json(path,rows,durable=False)
                        self.capacity,self.cpu_slots,self.gpus = amount,cpu_slots,gpus
                    self.applied_demand = selected
                    self.applied_plan = dict(layout=applied_layout,supply=plan['supply'])
                    self.world,self.shares = world,shares
                    self.admission_decision = decision
                    return
            limit = r.get('max_wait')
            if monotonic()-last_log >= 15.:
                event(self.request.run_dir, 'waiting_resources', stage=self.request.stage,
                      seconds=monotonic()-began, wait_forever=limit is None,
                      reason=decision['reason'], requested_bytes=demand.memory_bytes,
                      requested_cpu_slots=demand.cpu_slots, available_cpu_slots=cpu_available,
                      requested_gpu_count=demand.gpu_count,precision=r.get('precision'),
                      memory_budget_bytes=budget,alternatives=decision['alternatives'],
                      allow_partial_cpu=demand.allow_partial_cpu,
                      project_accounted_bytes=project, system_headroom_bytes=headroom)
                last_log = monotonic()
            if limit is not None and monotonic()-began >= limit:
                raise TimeoutError('已达到显式资源准入等待时限')
            delay = r.get('poll_seconds',2.)
            if limit is not None:
                delay = min(delay,max(0,limit-(monotonic()-began)))
            self.cancel.wait(delay)

    def _sample(self):
        with self.control_lock:
            self._sample_locked()

    def _sample_locked(self):
        with self.timing.measure('resource_sampling'):
            began, facts = monotonic(), host_capacity()
            members = list(self.members.values())
            rss = sum(psutil.Process(m['pid']).memory_info().rss for m in members if alive(m))
            r = self.request.resources
            spare, _ = system_headroom(facts, min(.8, r.get('ram_fraction', .8)), r.get('reserve_ram_bytes', 4*2**30))
            root = Path(self.request.registry_root)
            if not hasattr(self, 'last_registry') or monotonic()-self.last_registry >= 5.:
                with FileLock(str(root / '.resource.lock')):
                    self.rows = [row for row in read_json(root/'active.json') if alive(row)]
                    own = next(row for row in self.rows if row['pid']==self.owner['pid'] and row['birth']==self.owner['birth'])
                    own.update(rss_bytes=rss, members=[m for m in members if m != self.owner])
                    write_json(root/'active.json', self.rows, durable=False)
                self.last_registry = monotonic()
            own = next(row for row in self.rows if row['pid']==self.owner['pid'] and row['birth']==self.owner['birth'])
            own['rss_bytes'] = rss
            project = sum(max(row.get('ram_bytes', 0), row.get('rss_bytes', 0)) for row in self.rows)
            pending = sum(max(0, row.get('ram_bytes', 0)-row.get('rss_bytes', 0)) for row in self.rows)
            limit = min(100_000_000_000, r.get('memory_limit_bytes', 100_000_000_000))
            with self.condition:
                fixed_rss = sum(m.get('baseline_rss',0) for m in members if alive(m))
                reconcile(self.state,rss,fixed_rss)
                self.state.update(sampled_at=datetime.now(timezone.utc).isoformat(), sampled_monotonic=monotonic(),
                    host=facts, project_bytes=project,
                    headroom=max(0, min(self.capacity-rss, spare-pending+max(0,self.capacity-rss))),
                    task_rss=rss, peak_task_rss=max(rss,self.state.get('peak_task_rss',0)),
                    fixed_rss=fixed_rss,
                    paused=spare-pending<0 or project>limit or rss>self.capacity,
                    sampling_seconds=monotonic()-began)
                self.condition.notify_all()

    def _monitor(self):
        try:
            while not self.stopping.wait(1.):
                self._sample()
        finally:
            self.state['monitor_failed'] = not self.stopping.is_set()

    def __enter__(self):
        self.owner = identity()
        self.owner['baseline_rss'] = psutil.Process().memory_info().rss
        self._admit(self.request.demand,initial=True)
        self.manager = self.context.Manager()
        self.condition = self.context.Condition()
        demand = self.applied_demand
        self.requested_layout = self.applied_plan['layout']
        self.state = self.manager.dict(capacity=self.capacity, reserved=0, monitor_failed=False,
            reader_slots=demand.reader_slots,active_readers=0,workspace_reserved=0,
            workspace_capacity=demand.workspace_bytes if demand.reader_slots else 0,
            layout=self.requested_layout,plan_generation=0,replanning=False,replan_request=None,replan_result=None,
            cpu_slots=self.cpu_slots,gpus=self.gpus,world=self.world,shares=self.shares,
            precision=self.request.resources.get('precision'))
        self.state.update(claims={},unrealized_bytes=0)
        manager_member = identity(psutil.Process(self.manager._process.pid))
        manager_member['baseline_rss'] = psutil.Process(manager_member['pid']).memory_info().rss
        self.members = self.manager.dict({self.owner['pid']: self.owner, manager_member['pid']: manager_member})
        self.handle = ResourceHandle(self.owner, self.condition, self.state, self.cancel,
                                     self.request.resources, self.members)
        self._sample()
        self.allocation = Allocation(self.owner, self.gpus, self.requested_layout['workers'],
                                    demand.reader_slots, self.requested_layout['prefetch'], self.cpu_slots, self.capacity,
                                    requested=asdict(self.request.demand),
                                    applied=self._applied(),world=self.world,shares=self.shares,
                                    supply=self.applied_plan['supply'])
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='resource-monitor')
        self.future = self.executor.submit(self._monitor)
        return self

    def _applied(self):
        return dict(memory_bytes=self.capacity,cpu_slots=self.cpu_slots,gpus=self.gpus,
            world=self.world,shares=self.shares,precision=self.request.resources.get('precision'),
            layout=self.requested_layout,plan_generation=self.state['plan_generation'])

    def replan(self, demand, *, safe_boundary, layout=None):
        if os.getpid() != self.owner['pid']:
            raise RuntimeError('只有会话所有者可以应用重规划，请使用request_replan')
        if safe_boundary.kind not in ('before_case','block_group','batch_consumed','checkpoint_saved','prediction_case_completed'):
            raise ValueError('资源调整需要明确安全边界')
        previous = self.applied_demand
        if demand.gpu_count == previous.gpu_count and demand.gpu_ids in (None,self.request.demand.gpu_ids):
            demand = replace(demand,gpu_ids=previous.gpu_ids)
        changed_gpu = ((demand.gpu_count,demand.gpu_ids) != (previous.gpu_count,previous.gpu_ids) or
                       demand.gpu_count and demand.local_batches != previous.local_batches)
        if changed_gpu and not safe_boundary.checkpoint_path:
            raise ValueError('改变GPU组需要完整恢复点')
        with self.condition:
            if self.state['active_readers'] or self.state['workspace_reserved']:
                raise RuntimeError('活动病例额度释放后才能应用重规划')
            if self.state.get('replanning') and self.pending_token is None:
                raise RuntimeError('已有资源重规划正在执行')
            self.state['replanning'] = True
        try:
            self._admit(demand,layout=layout if layout is not None else self.state['layout'],
                        keep_gpus=None if changed_gpu else self.gpus)
            with self.control_lock, self.condition:
                self.requested_layout = self.state['layout']
                self.last_registry = -float('inf')
                self.allocation.memory_bytes = self.capacity
                self.allocation.gpus = self.gpus
                self.allocation.cpu_threads = self.cpu_slots
                self.allocation.reader_slots = demand.reader_slots
                self.allocation.workers = self.requested_layout['workers']
                self.allocation.prefetch = self.requested_layout['prefetch']
                self.allocation.applied = self._applied()
                self.allocation.applied['replan_request'] = asdict(demand)
                self.allocation.world,self.allocation.shares = self.world,self.shares
                self.allocation.supply = self.applied_plan['supply']
                return self.allocation
        finally:
            with self.condition:
                self.state['replanning'] = False
                self.condition.notify_all()

    def service_replans(self):
        if os.getpid() != self.owner['pid']:
            raise RuntimeError('重规划服务只能在会话所有者中执行')
        with self.condition:
            request = self.state.get('replan_request')
            if request is None:
                return None
            if request['abandoned']:
                self.state.update(replan_request=None,replanning=False)
                self.condition.notify_all()
                return None
            self.pending_token = request['token']
        succeeded = False
        try:
            result = self.replan(ResourceDemand(**request['demand']),safe_boundary=SafeBoundary(**request['boundary']),
                                 layout=request['layout'])
            succeeded = True
            return result
        finally:
            with self.condition:
                self.state['replan_result'] = dict(token=request['token'],failed=not succeeded,
                    allocation=asdict(self.allocation) if succeeded else None)
                self.state['replan_request'] = None
                self.state['replanning'] = False
                self.pending_token = None
                self.condition.notify_all()

    def __exit__(self, *_):
        self.stopping.set()
        self.executor.shutdown(wait=True)
        root = Path(self.request.registry_root)
        with FileLock(str(root/'.resource.lock')):
            rows = read_json(root/'active.json')
            write_json(root/'active.json', [r for r in rows if r['pid']!=self.owner['pid'] or r['birth']!=self.owner['birth']], durable=False)
        try:
            self.future.result()
        finally:
            self.manager.shutdown()
