"""Portable allocation checks, including visible ancestor cgroup limits.

Ancestor traversal follows the local chd-ct-reproduction CPU resource approach;
this module has no dependency on that project. Limits are caps, not reservations.
"""
import math
import os
from pathlib import Path
import threading
import time

import psutil


def group_paths(root, membership):
    root = Path(root)
    paths = {root, root / 'memory', root / 'cpu', root / 'cpu,cpuacct'}
    try:
        for line in Path(membership).read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            relative = relative.lstrip('/')
            if '..' in Path(relative).parts:
                continue
            bases = [root] if not controllers else [root / controllers, *(root / c for c in controllers.split(','))]
            for base in bases:
                current = base / relative
                while current == base or current.is_relative_to(base):
                    paths.add(current)
                    if current == base:
                        break
                    current = current.parent
    except (OSError, ValueError):
        pass
    return sorted(paths)


def allocation(config, root='/sys/fs/cgroup', membership='/proc/self/cgroup'):
    process = psutil.Process()
    try:
        cpu = float(len(process.cpu_affinity()))
    except (AttributeError, psutil.Error):
        cpu = float(os.cpu_count() or 1)
    affinity = cpu
    memory = int(psutil.virtual_memory().available)
    for folder in group_paths(root, membership):
        try:
            quota, period = (folder / 'cpu.max').read_text().split()
            if quota != 'max' and int(quota) > 0 and int(period) > 0:
                cpu = min(cpu, int(quota) / int(period))
        except (OSError, ValueError):
            pass
        try:
            quota = int((folder / 'cpu.cfs_quota_us').read_text())
            period = int((folder / 'cpu.cfs_period_us').read_text())
            if quota > 0 and period > 0:
                cpu = min(cpu, quota / period)
        except (OSError, ValueError):
            pass
        for limit_name, used_name in [('memory.max', 'memory.current'),
                                      ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            try:
                limit = int((folder / limit_name).read_text())
                used = int((folder / used_name).read_text())
                memory = min(memory, max(0, limit - used))
            except (OSError, ValueError):
                pass
    if config.cpu_limit is not None:
        cpu = min(cpu, config.cpu_limit)
    budget = int(memory * config.memory_fraction)
    if config.memory_budget_gib is not None:
        budget = min(budget, int(config.memory_budget_gib * 2**30))
    slots = min(config.workers, max(1, math.floor(cpu / config.compute_threads)))
    return {'cpu_affinity_count': int(affinity), 'cpu_budget': cpu,
            'available_memory_bytes': memory, 'memory_budget_bytes': budget,
            'case_slots': slots, 'read_slots': min(config.read_workers, slots),
            'max_active_cases': 0, 'note': 'estimates and visible limits; not a hard RSS limit or CPU reservation'}


class RuntimeSampler:
    def __enter__(self):
        self.process = psutil.Process()
        self.start = time.monotonic()
        self.cpu = self.process.cpu_times()
        self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        def sample():
            while not self.done.wait(.1):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        cpu = self.process.cpu_times()
        self.result = {'wall_seconds': time.monotonic() - self.start,
                       'cpu_seconds': cpu.user + cpu.system - self.cpu.user - self.cpu.system,
                       'sampled_peak_process_rss_bytes': max(self.peak, self.process.memory_info().rss)}
