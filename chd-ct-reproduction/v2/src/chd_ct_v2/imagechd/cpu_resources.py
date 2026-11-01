"""Conservative loader budgets from the current process allocation, not host totals."""

import os
import platform
import shutil
from pathlib import Path

GIB = 1024**3


def cgroup_dirs(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Walk v2 ancestors as well: a parent may impose the actual quota."""
    found = [root]
    try:
        for line in membership.read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            bases = (
                [root]
                if not controllers
                else [root / controllers, *[root / c for c in controllers.split(",")]]
            )
            for base in bases:
                path = (base / relative.lstrip("/")).resolve()
                if path.is_relative_to(base.resolve()):
                    found.extend([path, *[p for p in path.parents if p.is_relative_to(base.resolve())]])
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(found))


def resources(concurrency):
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    memory = 2 * GIB  # Conservative fallback when OS information is unavailable.
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory = int(fields["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        pass
    for folder in cgroup_dirs():
        try:
            quota, period = (folder / "cpu.max").read_text().split()
            if quota != "max":
                cpus = min(cpus, max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            pass
        try:
            quota = int((folder / "cpu.cfs_quota_us").read_text())
            period = int((folder / "cpu.cfs_period_us").read_text())
            if quota > 0:
                cpus = min(cpus, max(1, quota // period))
        except (OSError, ValueError):
            pass
        for limit, current in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            try:
                available = int((folder / limit).read_text()) - int((folder / current).read_text())
                memory = min(memory, max(0, available))
            except (OSError, ValueError):
                pass
    shared = shutil.disk_usage("/dev/shm").free if Path("/dev/shm").is_dir() else memory
    return dict(
        cpus=cpus,
        memory_available=memory,
        shared_available=shared,
        concurrency=max(1, concurrency),
        host=platform.node(),
    )


def batch_bytes(stage, spec):
    pixels = spec["size"] ** (2 if stage.startswith("blood") else 3)
    channels = spec["sequence"] if stage == "blood_lstm" else 1
    return spec["batch"] * pixels * (4 * channels + 8)


def worker_limit(config, hardware, stage):
    """Budget both persistent train/val pools; leave CPU and RAM for training."""
    tasks = hardware["concurrency"]
    spare = max(0, hardware["cpus"] // tasks - config["threads"])
    cpu_limit = spare // (2 * config["data_loading"]["worker_threads"])
    per_worker = batch_bytes(stage, config["stages"][stage]) * config["data_loading"]["prefetch_factor"]
    queue_budget = min(hardware["memory_available"] * 0.15, hardware["shared_available"] * 0.5) / tasks
    # 256 MiB per spawned interpreter plus two queued batches per split; mmap
    # pages are reclaimable OS cache, not a promised hard RSS limit.
    ram_limit = max(0, int(hardware["memory_available"] * 0.25 / tasks // (2 * (256 * 1024**2 + per_worker))))
    queue_limit = max(
        0, int((queue_budget - 4 * batch_bytes(stage, config["stages"][stage])) // (2 * per_worker))
    )
    return min(8, cpu_limit, ram_limit, queue_limit)
