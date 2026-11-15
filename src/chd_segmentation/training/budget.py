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
    if config.get('model') in ('medsam2','q_route1'):
        from ..models.memory import case_estimates as model_estimates
        return model_estimates(config,prepared,config['model_stage'])
    roi = config["technical"]["roi_xyz"]
    target = np.array(prepared["plan"]["configurations"]["3d_fullres"]["spacing"])
    order = prepared["plan"]["transpose_forward"]
    reports = []
    for row in prepared["manifest"]["cases"]:
        if stage == "train" and not set(row["roles"]) & {"train", "validation"}:
            continue
        if prepared.get("storage") is not None and stage in ("train", "predict"):
            item = prepared["storage"]["cases"][row["case_id"]]
            patch = int(np.prod(config["training"]["patch"])) * 16 * config["training"]["batch"]
            reports.append({"case_id": row["case_id"], "storage": "readonly_mmap", "input_shape_xyz": item["shape_czyx"][1:][::-1],
                            "raw_workspace_bytes": 0, "resample_workspace_bytes": 0,
                            "mapped_file_bytes": item["array_bytes"], "workspace_bytes": patch * 3 + item["locations_bytes"]})
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
        row.setdefault("workspace_bytes", row["raw_workspace_bytes"] + row["resample_workspace_bytes"])
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
    if stage == "preprocess":
        requested = r["max_workers"] if r["workers"] == "auto" else r["workers"]
        readers = r["max_readers"] if r["readers"] == "auto" else r["readers"]
        cap = max(0, min(requested, readers, len(reports), cores - 1))
        minimum = 2 * GIB + case_bytes
        result = {"admitted": False, "facts": facts, "budget_bytes": budget, "own_uss_credit_bytes": credit,
                  "peer_pending_bytes": pending, "minimum_bytes": minimum, "case_estimate_bytes": case_bytes,
                  "case_estimates": reports, "requested": {"workers": r["workers"], "readers": r["readers"]},
                  "reason": "离线预处理等待 CPU/RAM 安全余量"}
        for count in range(cap, -1, -1):
            fixed = (count + 2) * GIB
            if fixed + case_bytes > budget:
                continue
            slots = max(1, count)
            workspace = min(slots * case_bytes, budget - fixed)
            result.update(admitted=True, reason="离线病例构建预算允许", workers=[count], cpu_slots=count + 1,
                          ram_bytes=fixed + workspace, workspace_capacity_bytes=workspace,
                          reader_slots=slots, prefetch=1, cache_bytes=[0], max_workers=count,
                          breakdown={"processes_bytes": fixed, "case_workspaces_bytes": workspace, "queue_bytes": 0,
                                     "cache_total_bytes": 0, "cache_instances": 0})
            break
        return result
    profile = None
    retained,worker_bytes,cache_limit = 0,GIB,r['cache_bytes']
    fixed_workers = None
    # 队列结果、堆叠批次和 pinned/传输中的副本分别占用内存。
    patch_bytes = int(np.prod(config["training"]["patch"])) * 16 * config["training"]["batch"]
    cap = r["max_workers"] if r["workers"] == "auto" else min(r["workers"], r["max_workers"])
    cap = max(0, min(cap, cores - world))
    batch = config['training']['batch']
    shares = r['batch_layout'] if r['batch_layout'] is not None else [batch//world+int(rank<batch%world) for rank in range(world)]
    readers_cap = r["max_readers"] if r["readers"] == "auto" else min(r["readers"], r["max_readers"])
    base = (world + 1) * GIB
    if config.get('model') in ('medsam2','q_route1'):
        from ..models.memory import memory_profile
        profile = memory_profile(config,prepared,config['model_stage'],shares,reports)
        base = profile['coordinator_bytes']+sum(profile['rank_fixed_bytes'])+sum(profile['rank_case_bytes'])
        retained,worker_bytes = profile['retained_case_bytes'],profile['worker_fixed_bytes']
        patch_bytes = profile['queue_per_sample_bytes']*batch
        cache_limit = min(cache_limit,profile['cache_limit_bytes'])
        fixed_workers = profile['worker_layout']
        if fixed_workers is not None and (len(fixed_workers)!=world or any(type(w) is not int or w<0 for w in fixed_workers)):
            raise ValueError('模型恢复的worker布局无效')
    if fixed_workers is None:
        cap = min(cap,batch)
    minimum_workers = fixed_workers if fixed_workers is not None else [0]*world
    minimum = (base+sum(minimum_workers)*worker_bytes+sum(max(1,w) for w in minimum_workers)*retained+
               case_bytes+patch_bytes*3)
    result = {"admitted": False, "facts": facts, "budget_bytes": budget, "own_uss_credit_bytes": credit,
              "peer_pending_bytes": pending, "minimum_bytes": minimum, "case_estimate_bytes": case_bytes,
              "case_estimates": reports, "requested": {key: r[key] for key in
                 ("workers", "max_workers", "cache_bytes", "prefetch", "readers", "max_readers")}}
    if profile is not None:
        result['model_memory'] = {key:value for key,value in profile.items() if key!='records'}
    if budget < minimum:
        result["reason"] = "单个病例构建及零 worker 的最小预算仍不足，等待主机内存释放"
        return result
    if fixed_workers is not None and sum(fixed_workers)>cap:
        result['reason'] = '模型恢复需要原worker数量，等待相应CPU供数额度'
        return result
    counts = [sum(fixed_workers)] if fixed_workers is not None else range(cap,-1,-1)
    for count in counts:
        if fixed_workers is not None:
            workers = fixed_workers
        else:
            workers = [0]*world
            for _ in range(count):
                candidates = [rank for rank in range(world) if workers[rank]<shares[rank]]
                index = min(candidates,key=lambda rank:(workers[rank],rank))
                workers[index] += 1
        instances = sum(max(1, w) for w in workers)
        fixed = base + count * worker_bytes + instances * retained
        queues = patch_bytes*3
        if fixed + case_bytes + queues > budget:
            continue
        readers = max(1,min(readers_cap,instances,batch))
        cache = max(0,min(cache_limit,(budget-fixed-queues-case_bytes)//instances))
        workspace = min(readers*case_bytes,budget-fixed-queues-instances*cache)
        used = fixed+queues+workspace+instances*cache
        if used > budget:
            raise AssertionError("供数预算计算未闭合")
        result.update(admitted=True, reason="资源预算允许", workers=workers, cache_bytes=[int(cache)] * world,
                      max_workers=count, cpu_slots=world + count, ram_bytes=int(used), reader_slots=int(readers),
                      prefetch=1,workspace_capacity_bytes=int(workspace), breakdown={"processes_bytes": fixed, "queue_bytes": int(queues),
                      "case_builds": int(readers), "case_workspaces_bytes": int(workspace),
                      "cache_instances": instances, "cache_total_bytes": int(instances * cache)})
        if profile is not None:
            result['breakdown'].update(retained_case_instances=instances,retained_cases_bytes=instances*retained,
                rank_prediction_bytes=sum(profile['rank_case_bytes']))
        return result
    raise AssertionError("零 worker 最小预算未被实际规划覆盖")
