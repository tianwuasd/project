"""Stage-specific admission inspired by server-engineering V2, implemented for local analysis."""
from __future__ import annotations
import math
import os
import psutil

GIB = 1024**3


def estimate_case_bytes(voxels: int) -> int:
    """Conservative reservation for decoded arrays, masks, components and libraries."""
    return int(voxels*18 + 256*1024**2)


def choose_workers(cpu_slots: int, available_bytes: int, smallest_case_bytes: int,
                   requested: int | None = None, cap: int = 4) -> dict:
    """CPU80/RAM80 with >=4GiB reserve; requests cannot bypass bounds."""
    reserve = max(4*GIB, int(available_bytes*.2))
    usable = max(0, min(int(available_bytes*.8), available_bytes-reserve))
    cpu_budget = max(1, int(cpu_slots*.8))
    workers = min(cpu_budget, cap, usable//max(1,smallest_case_bytes))
    if requested is not None:
        if requested < 1:
            raise ValueError("workers must be positive")
        workers = min(workers, requested)
    return dict(cpu_slots=cpu_slots, cpu_budget=cpu_budget, available_memory_bytes=available_bytes,
        reserve_bytes=reserve, memory_budget_bytes=usable, max_workers=int(workers),
        requested_workers=requested or "auto", worker_cap=cap, threads_per_worker=1,
        selection="V2 stage CPU80/RAM80 principles; weighted case admission; local Windows implementation")


def live_budget(smallest_case_bytes: int, requested: int | None = None, cap: int = 4) -> dict:
    """Collect current allowed CPU and host RAM; no GPU request."""
    try:
        slots = len(psutil.Process().cpu_affinity())
    except (AttributeError, OSError):
        slots = os.cpu_count() or 1
    result = choose_workers(slots, psutil.virtual_memory().available, smallest_case_bytes, requested, cap)
    result.update(platform=os.name, cpu_percent_sample=psutil.cpu_percent(interval=.2),
                  cgroup_applicable=os.name != "nt")
    if os.name != "nt":
        raise RuntimeError("Linux cgroup admission must be adapted and tested before server execution")
    return result