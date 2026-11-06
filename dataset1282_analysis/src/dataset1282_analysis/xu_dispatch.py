"""Run independent Xu branches on explicitly selected devices with dependency barriers."""
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import replace
import os

import psutil
from subprocess import TimeoutExpired
from pathlib import Path
import threading
import time

from .contracts import atomic_json
from .observability.status import register_workers, write_status

PAIRS = {'crop64': 'crop128', 'all64': 'all128', 'init64': 'init128',
         'evaluate-all64': 'evaluate-all128'}


def device_identity(name, torch):
    device = torch.device(name)
    if device.type == 'cpu' and device.index is None:
        return {'device': 'cpu'}
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError(f'Requested device unavailable: {name}')
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f'Requested GPU is outside the visible allocation: {name}')
    props = torch.cuda.get_device_properties(index)
    return dict(device=f'cuda:{index}', name=props.name, total_bytes=props.total_memory,
                uuid=str(getattr(props, 'uuid', 'unavailable')))


def select_devices(single, multiple, torch):
    names = [part.strip() for part in multiple.split(',')] if multiple is not None else [single or 'cuda']
    if not 1 <= len(names) <= 2 or not all(names):
        raise ValueError('Select one or two visible devices; the largest independent group has two branches')
    devices = [device_identity(name, torch) for name in names]
    if len(devices) == 2:
        if any(row['device'] == 'cpu' or row['uuid'] in ('unavailable', 'None') for row in devices):
            raise ValueError('Parallel branches need two visible CUDA devices with verifiable UUIDs')
        if devices[0]['uuid'] == devices[1]['uuid'] or devices[0]['device'] == devices[1]['device']:
            raise ValueError('Parallel branches need two physically distinct authorized GPU devices')
    return devices


def waves(stages, devices, stop_after):
    stages = iter(stages)
    for first in stages:
        # Never advance over build-targets until its artifacts have been committed.
        if len(devices) == 2 and first.name in PAIRS and first.name != stop_after:
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
        self.lock = threading.Lock()
        self.active = {}
        self.stopping = False
        self.births = {os.getpid(): psutil.Process().create_time()}

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

    def attach(self, child, name, device, log):
        with self.lock:
            if self.stopping:
                raise RuntimeError('Another branch has stopped this stage group')
            self.active[child.pid] = (child, dict(pid=child.pid, stage=name, device=device, log=str(log)))
            try:
                self.births[child.pid] = psutil.Process(child.pid).create_time()
            except psutil.Error:
                pass  # No ownership credit for an unverified process.
            self._snapshot()

    def detach(self, child):
        with self.lock:
            self.active.pop(child.pid, None)
            self.births.pop(child.pid, None)
            self._snapshot()

    def pids(self):
        with self.lock:
            return dict(self.births)

    def peer_pids(self, device):
        with self.lock:
            return {pid: self.births[pid] for pid, (child, row) in self.active.items()
                    if row['device'] != device and pid in self.births and child.poll() is None}

    def stop(self):
        with self.lock:
            self.stopping = True
            running = [child for child, _ in self.active.values()]
        for child in running:
            if child.poll() is None:
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
        for child in running:
            try:
                child.wait(timeout=8)
            except TimeoutExpired:
                child.kill()
                child.wait()

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


def plan_stage(stage, args, out, children, concurrent, purpose, checked):
    from .xu_resources import ResourceUnavailable, limit_environment, stage_resources

    device = (stage.argv[stage.argv.index('--device') + 1]
              if '--device' in stage.argv and stage.name != 'audit-roi' else 'cpu')
    policy = args.resource_policy
    demand = cuda_demand = 0
    if args.profile == 'paper' and device.startswith('cuda'):
        from .xu_capacity import specification
        sizes = ()
        if stage.name == 'blood':
            sizes = ('blood-encoder','blood-sequence')
        elif stage.name in ('crop64','all64','init64','crop128','all128','init128'):
            sizes = ('3d128' if stage.name.endswith('128') else '3d64',)
        for size in sizes:
            spec = specification(size)
            demand = max(demand, spec['estimated_host_bytes'])
            if purpose == 'launch':
                cuda_demand = max(cuda_demand, spec['estimated_working_bytes'], checked.get((device,size),0))
    until = time.monotonic() + args.wait_seconds
    while True:
        children.ensure_running()
        try:
            plan = stage_resources([device], concurrent=concurrent, threads=args.threads, policy=policy,
                                   owned_pids=children.pids(), host_demand_bytes=demand,
                                   cuda_demand_bytes=cuda_demand,
                                   wave_reservation=getattr(args, 'wave_reservation', None),
                                   peer_active_pids=children.peer_pids(device),
                                   peer_finished=bool(getattr(args, 'peer_name', None) and
                                                      (out/'receipts'/(args.peer_name+'.json')).is_file()),
                                   require_pid_evidence=(policy == 'v2' and device.startswith('cuda')
                                                         and os.name == 'posix'))
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
    overlay = limit_environment(plan, device)
    # The current batch, model and I/O compete with the one optional read-ahead batch.
    prefetch = min(getattr(args, 'prefetch_mib', 4096) * 1024**2,
                   max(0, (plan['host_per_stage_bytes'] - demand) // 4))
    overlay.update(XU_PREFETCH='1' if args.prefetch and prefetch else '0', XU_PREFETCH_BYTES=str(prefetch))
    path = out/'resources'/f'{stage.name}-{purpose}-{time.time_ns()}.json'
    atomic_json(path, dict(stage=stage.name, purpose=purpose, device=device, concurrent=concurrent,
                           requested_threads=args.threads, prefetch_budget_bytes=prefetch, plan=plan))
    return device, overlay
