import math
from contextlib import suppress

import nibabel as nib
import numpy as np
import psutil

from .host import host_capacity

GIB = 1024 ** 3


def private_bytes(records):
    expected = {m["pid"]: m["birth"] for row in records for m in [row, *row.get("members", [])]}
    total = 0
    for process in psutil.process_iter(["pid", "create_time"], ad_value=None):
        info = process.info
        if info["pid"] in expected and info["create_time"] == expected[info["pid"]]:
            # 退出或无权读取的已登记进程不抵扣预留；不扫描其他用户进程的 USS。
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                total += process.memory_full_info().uss
    return total


def case_estimates(config, prepared, stage):
    roi = config["technical"]["roi_xyz"]
    target = np.array(prepared["plan"]["configurations"]["3d_fullres"]["spacing"])
    order = prepared["plan"]["transpose_forward"]
    reports = []
    for row in prepared["manifest"]["cases"]:
        if stage == "train" and not set(row["roles"]) & {"train", "validation"}:
            continue
        native = list(nib.load(row["image"]).shape)
        shape = np.array([end - begin for begin, end in roi] if roi else native)
        output = np.ceil(shape[::-1][order] * np.array(row["spacing_xyz"])[::-1][order] / target).astype(int)
        reports.append({"case_id": row["case_id"], "raw_shape_xyz": native, "input_shape_xyz": shape.tolist(),
                        "cropped_shape": None, "resampled_upper_shape": output.tolist(),
                        "raw_workspace_bytes": int(np.prod(shape)) * 32,
                        "resample_workspace_bytes": int(np.prod(output)) * 64})
    if not reports:
        raise ValueError("资源规划没有可用病例")
    for row in reports:
        row["workspace_bytes"] = row["raw_workspace_bytes"] + row["resample_workspace_bytes"]
    return reports


def cpu_plan(config, prepared, world, peers=(), stage="train", own=(), estimates=None):
    r = config["resources"]
    facts = host_capacity()
    credit = private_bytes(own)
    pending = sum(max(0, p["ram_bytes"] - private_bytes([p])) for p in peers)
    budget = max(0, min(int(facts["total_bytes"] * r["ram_fraction"]),
                        facts["available_bytes"] + credit - r["reserve_ram_bytes"]) - pending)
    cores = max(1, math.floor(facts["cores"] * r["cpu_fraction"]) - sum(p["cpu_slots"] for p in peers))
    reports = estimates if estimates is not None else case_estimates(config, prepared, stage)
    case_bytes = max(row["workspace_bytes"] for row in reports)
    # 队列结果、堆叠批次和 pinned/传输中的副本分别占用内存。
    patch_bytes = int(np.prod(config["training"]["patch"])) * 16 * config["training"]["batch"]
    cap = r["max_workers"] if r["workers"] == "auto" else min(r["workers"], r["max_workers"])
    cap = max(0, min(cap, cores - world))
    readers_cap = r["max_readers"] if r["readers"] == "auto" else min(r["readers"], r["max_readers"])
    base = (world + 1) * GIB
    minimum = base + case_bytes + patch_bytes * 3
    result = {"admitted": False, "facts": facts, "budget_bytes": budget, "own_uss_credit_bytes": credit,
              "peer_pending_bytes": pending, "minimum_bytes": minimum, "case_estimate_bytes": case_bytes,
              "case_estimates": reports, "requested": {key: r[key] for key in
                 ("workers", "max_workers", "cache_bytes", "prefetch", "readers", "max_readers")}}
    if budget < minimum:
        result["reason"] = "单个病例构建及零 worker 的最小预算仍不足，等待主机内存释放"
        return result
    for count in range(cap, -1, -1):
        workers = [count // world + int(rank < count % world) for rank in range(world)]
        instances = sum(max(1, w) for w in workers)
        depth = max(r["prefetch"], max(workers))
        # 先确保每个实际 worker 能取得任务，再按 RAM 减少预取。
        minimum_depth = max(1, max(workers))
        fixed = base + count * GIB
        if fixed + case_bytes + patch_bytes * (minimum_depth + 2) > budget:
            continue
        depth = max(minimum_depth, min(depth, (budget - fixed - case_bytes) // patch_bytes - 2))
        queues = patch_bytes * (depth + 2)
        readers = max(1, min(readers_cap, instances, (budget - fixed - queues) // case_bytes))
        remaining = budget - fixed - queues - readers * case_bytes
        cache = max(0, min(r["cache_bytes"], remaining // instances))
        used = fixed + queues + readers * case_bytes + instances * cache
        if used > budget:
            raise AssertionError("供数预算计算未闭合")
        result.update(admitted=True, reason="资源预算允许", workers=workers, cache_bytes=[int(cache)] * world,
                      max_workers=count, cpu_slots=world + count, ram_bytes=int(used), reader_slots=int(readers),
                      prefetch=int(depth), breakdown={"processes_bytes": fixed, "queue_bytes": int(queues),
                      "case_builds": int(readers), "case_workspaces_bytes": int(readers * case_bytes),
                      "cache_instances": instances, "cache_total_bytes": int(instances * cache)})
        return result
    raise AssertionError("零 worker 最小预算未被实际规划覆盖")
