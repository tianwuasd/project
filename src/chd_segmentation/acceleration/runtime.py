from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import multiprocessing as mp
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

from filelock import FileLock
import psutil

from ..artifacts.store import read_json, write_json
from ..observability.events import event
from .client import ResourceClient
from .contracts import Allocation, ResourceHandle
from .host import host_capacity
from .memory import identity, alive, system_headroom
from .timing import TimingBook


def attach(handle):
    member = identity()
    member['baseline_rss'] = psutil.Process().memory_info().rss
    if member['pid'] not in handle.members:
        handle.members[member['pid']] = member
    return ResourceClient(handle)


class ResourceSession:
    def __init__(self, request, *, context=None, cancel=None):
        self.request = request
        self.context = context or mp.get_context('spawn')
        self.cancel = cancel or self.context.Event()
        self.stopping, self.timing = Event(), TimingBook()
        self.failure = None

    def _admit(self, demand):
        root = Path(self.request.registry_root)
        root.mkdir(parents=True, exist_ok=True)
        began = monotonic()
        last_log = -float('inf')
        history = []
        while True:
            if self.cancel.is_set():
                raise InterruptedError('资源准入已取消')
            facts = host_capacity()
            r = self.request.resources
            observed = None
            if demand.gpu_count:
                from .registry import inventory
                owned = [self.owner['pid']]
                if hasattr(self, 'members'):
                    owned.extend(m['pid'] for m in self.members.values() if alive(m))
                observed = inventory(owned_pids=owned)
                history.append((monotonic(), observed))
                while len(history)>1 and history[-1][0]-history[1][0] >= r.get('share_window_seconds',30):
                    history.pop(0)
            headroom, _ = system_headroom(facts, min(.8, r.get('ram_fraction', .8)), r.get('reserve_ram_bytes', 4*2**30))
            with FileLock(str(root / '.resource.lock')):
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
                amount = max(demand.memory_bytes, current)
                cpu_available = max(0, int(facts['cores']*r.get('cpu_fraction', .8))-
                                    sum(row.get('cpu_slots', 0) for row in peers))
                # 只有能按获批数量调整布局的调用者才允许部分授予。
                self.cpu_slots = min(demand.cpu_slots, cpu_available) if demand.allow_partial_cpu else (
                    demand.cpu_slots if cpu_available >= demand.cpu_slots else 0)
                self.gpus = []
                if observed is not None:
                    from .registry import gpu_decision
                    gpu_config = {'resources': {**r, 'gpu_count': demand.gpu_count, 'gpu_ids': demand.gpu_ids},
                        'training': {'batch': sum(demand.local_batches) if demand.local_batches else demand.gpu_count,
                                     'precision': r.get('precision', 'fp32')}}
                    peak_path = root/'gpu-peaks.json'
                    peaks = read_json(peak_path) if peak_path.is_file() else {}
                    self.gpus = gpu_decision(gpu_config, r.get('gpu_spec', {}), history, peaks, peers)['gpus']
                if self.cpu_slots > 0 and self.gpus is not None and amount <= limit-project and max(0, amount-current) <= headroom-pending:
                    rows = peers + [{**self.owner, 'members': [], 'ram_bytes': amount, 'rss_bytes': current,
                        'cpu_slots': self.cpu_slots, 'uuids': [g['uuid'] for g in self.gpus], 'memory_limit_bytes': limit}]
                    write_json(path, rows, durable=False)
                    self.capacity = amount
                    return
            limit = r.get('max_wait')
            if monotonic()-last_log >= 15.:
                event(self.request.run_dir, 'waiting_resources', stage=self.request.stage,
                      seconds=monotonic()-began, wait_forever=limit is None,
                      reason='CPU、GPU或内存额度不足', requested_bytes=demand.memory_bytes,
                      requested_cpu_slots=demand.cpu_slots, available_cpu_slots=cpu_available,
                      allow_partial_cpu=demand.allow_partial_cpu,
                      project_accounted_bytes=project, system_headroom_bytes=headroom)
                last_log = monotonic()
            if limit is not None and monotonic()-began >= limit:
                raise TimeoutError('已达到显式资源准入等待时限')
            sleep(r.get('poll_seconds', 2.))

    def _sample(self):
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
                self.state.update(sampled_at=datetime.now(timezone.utc).isoformat(), sampled_monotonic=monotonic(),
                    host=facts, project_bytes=project,
                    headroom=max(0, min(self.capacity-rss, spare-pending+max(0,self.capacity-rss))),
                    task_rss=rss, peak_task_rss=max(rss,self.state.get('peak_task_rss',0)),
                    fixed_rss=sum(m.get('baseline_rss', 0) for m in members if alive(m)),
                    paused=spare-pending<0 or project>limit, sampling_seconds=monotonic()-began)
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
        self._admit(self.request.demand)
        self.manager = self.context.Manager()
        self.condition = self.context.Condition()
        self.state = self.manager.dict(capacity=self.capacity, reserved=0, monitor_failed=False)
        manager_member = identity(psutil.Process(self.manager._process.pid))
        manager_member['baseline_rss'] = psutil.Process(manager_member['pid']).memory_info().rss
        self.members = self.manager.dict({self.owner['pid']: self.owner, manager_member['pid']: manager_member})
        self.handle = ResourceHandle(self.owner, self.condition, self.state, self.cancel,
                                     self.request.resources, self.members)
        self._sample()
        self.allocation = Allocation(self.owner, self.gpus, [0], 1, 2, self.cpu_slots, self.capacity,
                                    requested=asdict(self.request.demand),
                                    applied=dict(memory_bytes=self.capacity, cpu_slots=self.cpu_slots, gpus=self.gpus))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='resource-monitor')
        self.future = self.executor.submit(self._monitor)
        return self

    def replan(self, demand, *, safe_boundary):
        if safe_boundary.kind not in ('before_case','block_group','batch_consumed','checkpoint_saved','prediction_case_completed'):
            raise ValueError('资源调整需要明确安全边界')
        if demand.gpu_count and not safe_boundary.checkpoint_path:
            raise ValueError('改变GPU组需要完整恢复点')
        self._admit(demand)
        self.last_registry = -float('inf')
        with self.condition:
            self.state['capacity'] = self.capacity
        self.allocation.memory_bytes = self.capacity
        self.allocation.gpus = self.gpus
        self.allocation.cpu_threads = self.cpu_slots
        self.allocation.requested = asdict(demand)
        self.allocation.applied = dict(memory_bytes=self.capacity, cpu_slots=self.cpu_slots, gpus=self.gpus)
        return self.allocation

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
