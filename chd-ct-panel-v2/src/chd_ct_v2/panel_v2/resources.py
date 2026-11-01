"""Live VISTA capacity adapted to CT's independent experiments, not DDP."""

import copy
import math
import os
from pathlib import Path

from ..recovery import write_json
from .live_resources import case_policy, snapshot


def normalized(facts):
    facts = dict(facts)
    if "cpu_available" not in facts and facts.get("cpus") is not None:
        idle = facts.get("idle_fraction")
        facts["cpu_available"] = facts["cpus"] * (1 if idle is None else max(0, min(1, idle)))
    if "cpu_budget" not in facts and facts.get("cpu_available") is not None:
        available = facts["cpu_available"]
        if facts.get("cpu_permitted") is not None:
            available = min(available, facts["cpu_permitted"])
        facts["cpu_budget"] = math.floor(0.8 * available)
    if "available_memory_bytes" not in facts and "memory_available" in facts:
        facts["available_memory_bytes"] = facts["memory_available"]
    return facts


def case_budget(facts, cases, case_bytes, concurrency=1, parallel=True):
    facts = normalized(facts)
    count = max(1, int(concurrency))
    if facts.get("cpu_budget") is not None:
        facts["cpu_budget"] //= count
    else:
        facts.pop("cpu_budget", None)
    ram = facts.get("memory_budget_bytes")
    if ram is None and facts.get("available_memory_bytes") is not None:
        ram = int(0.8 * facts["available_memory_bytes"])
    if ram is not None:
        facts["memory_budget_bytes"] = ram // count
    selected = case_policy(facts, cases if parallel else min(1, cases), max(1, case_bytes))
    if cases and not selected["workers"]:
        raise ValueError("当前 CPU/RAM 80% 预算不足，保留任务后重试：" + selected["capacity_status"])
    return dict(
        selected,
        processes=selected["workers"],
        threads=selected["threads_per_worker"],
        concurrency=count,
        strategy="vista_v2_capacity80_ct",
        hardware=facts,
    )


def loader_budget(config, rows, stage, facts=None, concurrency=1):
    """Account both resident pools, whole-case decoding, batches and queues."""
    from ..cpu_runtime import volume_bytes

    facts = normalized(snapshot() if facts is None else facts)
    spec = config["stages"][stage]
    voxels = spec["size"] ** (2 if stage.startswith("blood") else 3)
    sequence = spec.get("sequence", 1)
    batch = max(1, spec["batch"] * voxels * (4 * sequence + 8) * 4)
    private = max(512 * 2**20, volume_bytes([r for split in rows.values() for r in split]), batch * 2)
    count = max(1, int(concurrency))
    known_cpu = facts.get("cpu_budget")
    cpu = (1 if known_cpu is None else known_cpu) // count
    ram = facts.get("memory_budget_bytes")
    if ram is None and facts.get("available_memory_bytes") is not None:
        ram = int(0.8 * facts["available_memory_bytes"])
    if ram is not None:
        ram //= count
    # Two parent datasets can hold one case each; workers hold private case buffers.
    base = 2 * private + 3 * batch
    if cpu < 1 or (ram is not None and ram < base):
        raise ValueError("当前 CPU/RAM 预算无法容纳 CT 加载器，未启动训练进程")
    main = min(4, cpu)
    workers = min(64, max(0, cpu - main))
    if ram is None:
        workers, main = 0, 1
    else:
        workers = min(workers, max(0, (ram - base) // (2 * (private + batch))))
    selected = dict(
        cpu_budget=cpu,
        memory_budget_bytes=ram,
        main_threads=main,
        workers=workers,
        worker_threads=1,
        prefetch_factor=1,
        estimated_ram_bytes=base + 2 * workers * (private + batch),
        total_cpu_slots=main + workers,
        concurrency=count,
        strategy="vista_v2_capacity80_ct",
        hardware=facts,
    )
    adapted = copy.deepcopy(config)
    adapted["data_loading"] = dict(
        config.get("data_loading", {}),
        workers=workers,
        worker_threads=1,
        prefetch_factor=1,
        stage_workers={},
        persistent_workers=True,
        auto_tune=False,
        cache="none",
    )
    return adapted, selected


def direct_config(config, output, concurrency=1):
    """Stable scientific contract; ephemeral resource choices stay out of checkpoints."""
    os.environ["CHD_V2_CONCURRENCY"] = str(max(1, concurrency))
    result = copy.deepcopy(config)
    result["data_loading"] = dict(result.get("data_loading", {}), auto_tune=False, cache="none")
    write_json(
        Path(output) / "cpu-policy.json",
        {
            "policy": "vista_v2_capacity80_ct",
            "candidate_benchmark": False,
            "scientific_parameters_changed": False,
        },
    )
    return result
