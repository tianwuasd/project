"""Run independent Xu branches on explicitly selected devices with dependency barriers."""
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import replace
import os

import psutil
from subprocess import Popen
from pathlib import Path
import threading
import time

from .contracts import atomic_json
from .observability.status import register_workers, write_status

PAIRS = {'crop64': 'crop128', 'all64': 'all128', 'init64': 'init128',
         'auto-init64': 'auto-init128',
         'evaluate-all64': 'evaluate-all128'}


def device_identity(name, torch):
    device = torch.device(name)
    if device.type == 'cpu' and device.index is None:
        return {'device': 'cpu'}
    if device.type != 'cuda':
        raise ValueError(f'Requested device unavailable: {name}')
    from .xu_gpu_probe import visible_devices
    index = 0 if device.index is None else device.index
    cards = visible_devices()
    if index >= len(cards):
        raise ValueError(f'Requested GPU is outside the visible allocation: {name}')
    return dict(cards[index])


def select_devices(single, multiple, torch):
    automatic = multiple in ('auto','all-visible') or (multiple is None and single is None)
    if automatic:
        from .xu_gpu_probe import visible_devices
        count = len(visible_devices())
        if count < 1:
            raise ValueError('No visible CUDA devices; select --device cpu for a CPU run')
        mask = os.environ.get('XU_COORDINATOR_AUTH_MASK',os.environ.get('CUDA_VISIBLE_DEVICES'))
        container_mask = os.environ.get('NVIDIA_VISIBLE_DEVICES','').strip().lower()
        restricted = bool(mask or container_mask not in ('','all','void','none'))
        if count > 1 and os.name == 'posix' and not restricted and multiple != 'all-visible':
            raise ValueError('No explicit GPU allocation mask: specify assigned devices, or --devices all-visible '
                             'if all visible physical GPUs are allocated to this job')
        names = [f'cuda:{i}' for i in range(count)]
    else:
        names = [part.strip() for part in multiple.split(',')] if multiple is not None else [single]
    if not all(names):
        raise ValueError('Expected allocated device names or auto')
    devices = [device_identity(name, torch) for name in names]
    if len(devices) > 1:
        if any(row['device'] == 'cpu' or row['uuid'] in ('unavailable', 'None') for row in devices):
            raise ValueError('GPU pooling needs visible CUDA devices with verifiable UUIDs')
        if len({row['uuid'] for row in devices}) != len(devices):
            raise ValueError('Select physically distinct authorized GPU devices')
    return devices


def waves(stages, devices, stop_after):
    stages = iter(stages)
    for first in stages:
        # Never advance over build-targets until its artifacts have been committed.
        if len(devices) > 1 and first.name in PAIRS and first.name != stop_after:
            second = next(stages, None)
            if second is None or second.name != PAIRS[first.name]:
                raise ValueError(f'Unexpected dependency order after {first.name}')
            argv = list(second.argv)
            argv[argv.index('--device') + 1] = devices[1]
            yield (first, replace(second, argv=tuple(argv)))
        else:
            yield (first,)


class Children:
    """Serialize ownership and status updates across branch launcher threads."""
    def __init__(self, out, owner):
        self.out = Path(out)
        self.owner = owner
        self.lock = threading.RLock()
        self.stop_lock = threading.Lock()
        self.active = {}
        self.stopping = False
        self.births = {os.getpid(): psutil.Process().create_time()}
        self.configure_launch(1)

    def configure_launch(self, count):
        if self.active or self.stopping or count < 1:
            raise RuntimeError('启动同步必须在本组进程启动前配置')
        self.launch_members = {}
        self.launch_gate = threading.Barrier(count, action=self._freeze_launch_members)

    def _freeze_launch_members(self):
        with self.lock:
            self.launch_members = dict(self.births)

    def await_launch(self, guard, grouped=True):
        if not grouped:
            self.ensure_running()
            guard.check()
            return self.pids()
        self.launch_gate.wait()
        self.ensure_running()
        guard.check()
        self.launch_gate.wait()
        self.ensure_running()
        return dict(self.launch_members)

    def _snapshot(self):
        rows = [row for _, row in self.active.values()]
        register_workers(self.out, self.owner, [row['pid'] for row in rows])
        if rows:
            write_status(self.out, state='running', stage=rows[-1]['stage'], log=rows[-1]['log'],
                         pid=rows[-1]['pid'], active=rows)
        elif (self.out/'status.json').is_file():
            from .contracts import read_json
            last = read_json(self.out/'status.json')
            write_status(self.out, state='running', stage=last.get('stage'), log=last.get('log'), active=[])

    def attach(self, child, name, device, log, devices=None):
        with self.lock:
            if self.stopping:
                raise RuntimeError('Another branch has stopped this stage group')
            self.active[child.pid] = (child, dict(pid=child.pid, stage=name, device=device,
                                                  devices=list(devices or [device]), log=str(log)))
            self.births[child.pid] = psutil.Process(child.pid).create_time()
            self._snapshot()

    def spawn(self, command, name, device, log, devices=None, **options):
        # 创建和登记期间阻止并发取消；子进程仍等待父进程的启动许可。
        with self.lock:
            self.ensure_running()
            child = Popen(command, **options)
            try:
                self.attach(child, name, device, log, devices)
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=8)
                if child.stdin is not None:
                    child.stdin.close()
                self.active.pop(child.pid, None)
                self.births.pop(child.pid, None)
                raise
            return child

    def detach(self, child):
        with self.lock:
            self.active.pop(child.pid, None)
            self.births.pop(child.pid, None)
            self._snapshot()

    def pids(self):
        with self.lock:
            return dict(self.births)

    def owned_identities(self, stage=None):
        return {pid:row['create_time'] for pid,row in self.owned_processes(stage).items()}

    def owned_processes(self, stage=None):
        with self.lock:
            entries = [(pid,self.births[pid],row['stage']) for pid,(child,row) in self.active.items()
                       if child.poll() is None and pid in self.births and
                       (stage is None or row['stage'] == stage)]
        table = {p.pid:p.info for p in psutil.process_iter(
            attrs=['pid','ppid','create_time','memory_full_info'],ad_value=None)}
        owned = {str(os.getpid()):dict(table[os.getpid()],stage=None)} if stage is None else {}
        for pid,birth,name in entries:
            if pid not in table:
                continue
            row = table[pid]
            if row['create_time'] != birth or row['ppid'] != os.getpid():
                raise RuntimeError('本次子进程身份发生变化')
            owned[str(pid)] = dict(row,stage=name)
            pending = [pid]
            while pending:
                parent = pending.pop()
                for key,child in table.items():
                    if child['ppid'] == parent and str(key) not in owned:
                        if child['create_time'] is None:
                            raise RuntimeError('无法核实本次子进程的创建时间')
                        owned[str(key)] = dict(child,stage=name)
                        pending.append(key)
        return owned

    def peer_pids(self, device):
        with self.lock:
            return {pid: self.births[pid] for pid, (child, row) in self.active.items()
                    if row['device'] != device and pid in self.births and child.poll() is None}

    def stop(self):
        with self.lock:
            self.stopping = True
            running = [child for child, _ in self.active.values()]
        self.launch_gate.abort()
        for child in running:
            self.stop_child(child)

    def stop_child(self, child):
        with self.stop_lock:
            self._stop_child(child)

    def _stop_child(self, child):
        with self.lock:
            entry = self.active.get(child.pid)
        if child.poll() is not None:
            return
        if entry is None:
            raise RuntimeError('拒绝终止未登记的进程')
        name = entry[1]['stage']
        rows = self.owned_processes(name)
        verified = []
        for pid, row in rows.items():
            try:
                process = psutil.Process(int(pid))
                if process.create_time() != row['create_time']:
                    raise RuntimeError('终止前进程身份发生变化')
                verified.append(process)
            except psutil.NoSuchProcess:
                continue
        for process in reversed(verified):
            try:
                if process.is_running():
                    process.terminate()
            except psutil.NoSuchProcess:
                continue
        _, alive = psutil.wait_procs(verified, timeout=8)
        for process in alive:
            try:
                if process.is_running():
                    process.kill()
            except psutil.NoSuchProcess:
                continue
        _, alive = psutil.wait_procs(alive, timeout=8)
        if alive:
            raise RuntimeError('本任务进程尚未退出，禁止重新分配设备')
        child.wait(timeout=8)

    def ensure_running(self):
        with self.lock:
            if self.stopping:
                raise RuntimeError('Branch group cancelled')


def run_wave(wave, worker, children):
    if len(wave) == 1:
        return [worker(wave[0])]
    pool = ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix='xu-branch')
    try:
        futures = [pool.submit(worker, stage) for stage in wave]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            if future.exception() is not None:
                future.result()
        return [future.result() for future in futures]
    except BaseException:
        children.stop()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def reserve_pair(args, out):
    from .xu_resources import ResourceUnavailable, reserve_wave
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            allocation = reserve_wave(args.threads, args.resource_policy, lanes=2)
            atomic_json(out/'resources'/f'pair-share-{time.time_ns()}.json', allocation)
            return allocation
        except ResourceUnavailable as exc:
            if time.monotonic() >= deadline:
                atomic_json(out/'resources'/f'pair-denied-{time.time_ns()}.json', dict(reason=str(exc)))
                raise
            time.sleep(min(2, max(0, deadline-time.monotonic())))


def prefetch_cpu_allocation(slots, threads, workers):
    from .xu_resources import ResourceUnavailable
    if any(type(n) is not int or n < 1 for n in (slots,threads,workers)):
        raise ValueError('CPU 份额、数值线程和供数上限必须为正整数')
    if slots < 2:
        return 1,0
    actual_threads = min(threads,slots//2)
    actual_workers = min(workers,slots//actual_threads-1)
    return actual_threads,actual_workers


def plan_stage(stage, args, out, children, concurrent, purpose, checked):
    from .xu_resources import ResourceUnavailable, GpuResourcesChanged, combined_environment, stage_resources
    from .xu_fleet import INFERENCE, TRAIN_3D, needs_for_width, preflight_sizes
    from .xu_supply_budget import stage_minimum, prefetch_budget
    device = (stage.argv[stage.argv.index('--device') + 1]
              if '--device' in stage.argv and stage.name != 'audit-roi' else 'cpu')
    group = list(getattr(args,'assigned_devices',None) or [device])
    if device != group[0]:
        raise ValueError('Stage primary device differs from its frozen group')
    policy = args.resource_policy
    enabled = args.prefetch and stage.name in TRAIN_3D | {'blood'}
    requested_workers = getattr(args,'prefetch_workers',4)
    if requested_workers < 1:
        raise ValueError('CPU worker 上限必须为正整数')
    demand = cuda_demand = 0
    if device != 'cpu' and stage.name in INFERENCE:
        demand = getattr(args,'inference_worker_bytes',0) * len(group)
    if args.profile == 'paper' and device.startswith('cuda'):
        from .xu_capacity import specification
        sizes = preflight_sizes(stage.name)
        for size in sizes:
            demand = max(demand, specification(size)['estimated_host_bytes'])
        if purpose == 'launch':
            cuda_demand = dict(zip(group,needs_for_width(stage.name,len(group),args.profile)))
            for size in sizes:
                measured = checked.get((tuple(group),size),{})
                for card in group:
                    cuda_demand[card] = max(cuda_demand[card],measured.get(card,0))
    until = time.monotonic() + args.wait_seconds
    minimum = stage_minimum(args,stage.name)
    while True:
        children.ensure_running()
        try:
            plan = stage_resources(group, concurrent=concurrent, threads=args.threads, policy=policy,
                                   owned_pids=children.pids(), host_demand_bytes=demand+minimum,
                                   cuda_demand_bytes=cuda_demand,
                                   wave_reservation=getattr(args, 'wave_reservation', None),
                                   peer_active_pids=children.peer_pids(device),
                                   peer_finished=bool(getattr(args, 'peer_name', None) and
                                                      (out/'receipts'/(args.peer_name+'.json')).is_file()),
                                   require_pid_evidence=(policy == 'v2' and device.startswith('cuda')
                                                         and os.name == 'posix'))
            workers = 1
            if enabled:
                plan['threads'],workers = prefetch_cpu_allocation(
                    plan['cpu_slots']//concurrent,plan['threads'],requested_workers)
            host = plan['host_per_stage_bytes']
            prefetch = prefetch_budget(host,demand,minimum,getattr(args,'prefetch_mib',16384)*1024**2) if enabled else 0
        except GpuResourcesChanged:
            raise
        except ResourceUnavailable as exc:
            if time.monotonic() >= until:
                atomic_json(out/'resources'/f'{stage.name}-denied-{time.time_ns()}.json',
                            dict(stage=stage.name, purpose=purpose, device=device, reason=str(exc),
                                 concurrent=concurrent, policy=policy))
                raise
            print(f'Waiting for {stage.name} on {device}: {exc}', flush=True)
            time.sleep(min(2, max(0, until - time.monotonic())))
            continue
        break
    overlay = combined_environment(plan, group)
    # 当前批次和模型之外，为有序供数队列分配独立预算。
    overlay.update(XU_PREFETCH='1' if enabled else '0', XU_PREFETCH_BYTES=str(prefetch),
                   XU_PREFETCH_WORKERS=str(workers),XU_PREFETCH_DEPTH=str(getattr(args,'prefetch_depth',max(1,2*workers))))
    path = out/'resources'/f'{stage.name}-{purpose}-{time.time_ns()}.json'
    atomic_json(path, dict(stage=stage.name, purpose=purpose, device=device, devices=group, concurrent=concurrent,
                           requested_threads=args.threads, prefetch_budget_bytes=prefetch,
                           prefetch_workers=workers,minimum_batch_bytes=minimum,
                           requested_prefetch_bytes=getattr(args,'prefetch_mib',16384)*1024**2,plan=plan))
    return device, overlay
