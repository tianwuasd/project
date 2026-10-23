"""Throughput-oriented, observable estimates; never an optimality claim.

Case workers share the memory-aware queue in pipeline.run. Native library
threads are kept separate from explicitly allocated slab threads.
"""
from dataclasses import asdict, replace
import math

import numpy as np
import psutil

from .config import Config
from .pipeline import plan
from .resources import allocation


FIELDS = {'cpu_limit', 'memory_budget_gib', 'workers', 'compute_threads', 'read_workers', 'max_case_gib'}


def idle_fraction():
    """Sample cores visible to this process, not whole-host load average."""
    try:
        affinity = psutil.Process().cpu_affinity()
        usage = psutil.cpu_percent(interval=.2, percpu=True)
        visible = [usage[i] for i in affinity if i < len(usage)]
        return max(0., min(1., 1 - sum(visible) / len(visible) / 100)) if visible else None
    except (psutil.Error, AttributeError, OSError):
        return None


def select_resources(input_root, target_spacing, request=None, manifest=None):
    """Return concrete Config, evidence, and header-only plan.

None in a request field means auto. Explicit values are ceilings subject to
visible allocation. Automatic settings are recalculated on each job attempt;
target_spacing is already resolved and cannot be refitted by this function.
"""
    request = dict(request or {})
    if set(request) - FIELDS:
        raise ValueError('unknown resource settings: ' + ', '.join(sorted(set(request) - FIELDS)))
    explicit = {k: v for k, v in request.items() if v is not None}
    base = Config(target_spacing=target_spacing, memory_fraction=.9, **explicit)
    visible = allocation(base)
    idle = idle_fraction()
    cpu = max(1, math.floor(visible['cpu_budget'] * (.9 * (idle if idle is not None else 1))))
    if base.cpu_limit is not None:
        cpu = max(1, math.floor(visible['cpu_budget']))
    memory = visible['memory_budget_bytes']
    if memory < 2**20:
        raise ValueError('available memory budget is below 1 MiB; retry when resources are available')
    initial = plan(input_root, replace(base, compute_threads=1), manifest)
    sizes = [r['estimated_working_bytes'] for r in initial['cases'] if r['status'] == 'ready']
    if not sizes:
        raise ValueError('no valid CT images in the header plan')
    fitting = [s for s in sizes if s <= min(memory, explicit.get('max_case_gib', memory / 2**30) * 2**30)]
    typical = float(np.percentile(fitting or sizes, 75))
    workers = explicit.get('workers', min(32, len(sizes), cpu, max(1, int(memory / typical))))
    threads = explicit.get('compute_threads', min(8, max(1, cpu // workers)))
    if threads > max(1, math.floor(visible['cpu_budget'])):
        raise ValueError('compute_threads exceeds visible CPU allocation')
    # Explicit threads imply at least that CPU allowance, within the user's cap.
    cpu = max(cpu, threads)
    workers = min(workers, max(1, cpu // threads), len(sizes))
    reads = min(explicit.get('read_workers', min(8, max(1, cpu // 4))), workers)
    config = replace(base, cpu_limit=cpu, workers=workers, compute_threads=threads,
                     read_workers=reads, memory_budget_gib=memory / 2**30,
                     max_case_gib=explicit.get('max_case_gib', memory / 2**30))
    report = plan(input_root, config, manifest)
    # Slab buffers grow with threads. Reduce automatic threading before rejecting
    # a large case that would fit with one thread.
    if 'compute_threads' not in explicit:
        while config.compute_threads > 1 and any(
                a.get('estimated_working_bytes', memory + 1) <= min(memory, config.max_case_gib * 2**30)
                < b.get('estimated_working_bytes', 0)
                for a, b in zip(initial['cases'], report['cases'])):
            config = replace(config, compute_threads=max(1, config.compute_threads // 2))
            report = plan(input_root, config, manifest)
    oversized = sum(r.get('estimated_working_bytes', 0) > min(memory, config.max_case_gib * 2**30)
                    for r in report['cases'] if r['status'] == 'ready')
    evidence = {'policy': 'throughput_heuristic_v1', 'requested': request,
                'visible': visible, 'observed_idle_fraction': idle,
                'automatic_cpu_fraction': .9, 'automatic_memory_fraction': .9,
                'case_memory_p75_bytes': typical, 'oversized_cases': oversized,
                'selected': asdict(config),
                'note': 'CPU and RSS are estimates, not OS reservations; memory-aware queue limits simultaneous cases'}
    return config, evidence, report
