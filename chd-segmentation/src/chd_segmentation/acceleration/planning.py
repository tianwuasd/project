import math
from contextlib import suppress

import nibabel as nib
import numpy as np
import psutil

from .host import host_capacity
from .limits import reserve_margin

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
    prediction = stage in ('predict','prediction')
    source = None
    if prediction and prepared.get('storage') is not None:
        from ..preprocessing.inputs import resolve_preprocessed
        source = resolve_preprocessed(prepared['storage']['root'],dataset=prepared['dataset'],
            case_ids=[row['case_id'] for row in prepared['manifest']['cases']],storage=prepared['storage'])
    for row in prepared["manifest"]["cases"]:
        if stage == "train" and not set(row["roles"]) & {"train", "validation"}:
            continue
        if prepared.get("storage") is not None and (stage=='train' or prediction):
            item = prepared["storage"]["cases"][row["case_id"]]
            patch = int(np.prod(config["training"]["patch"])) * 16 * config["training"]["batch"]
            reports.append({"case_id": row["case_id"], "storage": "readonly_mmap", "input_shape_xyz": item["shape_czyx"][1:][::-1],
                            "raw_workspace_bytes": 0, "resample_workspace_bytes": 0,
                            "mapped_file_bytes": item["array_bytes"], "workspace_bytes": patch * 3 + item["locations_bytes"]})
            if prediction:
                from ..training.stage_memory import prediction_workspace
                entry = source.open(row['case_id'],with_label=False)
                memory = prediction_workspace(item['shape_czyx'],entry['source_shape_xyz'],entry['roi_xyz'],
                    len(prepared['dataset_json']['labels']),config['prediction'],config['training']['patch'])
                reports[-1].update(memory,workspace_bytes=memory['prediction_workspace_bytes']+item['locations_bytes'])
            continue
        native = list(nib.load(row["image"]).shape)
        shape = np.array([end - begin for begin, end in roi] if roi else native)
        output = np.ceil(shape[::-1][order] * np.array(row["spacing_xyz"])[::-1][order] / target).astype(int)
        legacy_bytes = int(np.prod(shape)) * 32 + int(np.prod(output)) * 64
        streamed = stage == "preprocess"
        reports.append({"case_id": row["case_id"], "raw_shape_xyz": native, "input_shape_xyz": shape.tolist(),
                        "cropped_shape": None, "resampled_upper_shape": output.tolist(),
                        "legacy_whole_case_bytes": legacy_bytes, "storage": "bounded_disk_blocks" if streamed else "source",
                        "threshold_exceeded": legacy_bytes > config["preprocessing"]["block_threshold_bytes"],
                        "raw_workspace_bytes": 2 * GIB if streamed else int(np.prod(shape)) * 32,
                        "resample_workspace_bytes": config["preprocessing"]["block_bytes"] * 4 if streamed else int(np.prod(output)) * 64,
                        "output_array_bytes": int(np.prod(output)) * 5})
        if prediction:
            from ..training.stage_memory import prediction_workspace
            memory = prediction_workspace([1,*output.tolist()],native,
                roi if roi else [[0,int(size)] for size in native],len(prepared['dataset_json']['labels']))
            reports[-1].update(memory,workspace_bytes=legacy_bytes+memory['prediction_workspace_bytes'])
    if not reports:
        raise ValueError("资源规划没有可用病例")
    for row in reports:
        row.setdefault("workspace_bytes", row["raw_workspace_bytes"] + row["resample_workspace_bytes"])
    return reports


def cpu_plan(config, prepared, world, peers=(), stage="train", own=(), estimates=None, allocation=None, bounds=None):
    r = config["resources"]
    if bounds is not None and allocation is not None:
        raise ValueError('bounds和allocation只能选择一个资源来源')
    facts = bounds['facts'] if bounds is not None else host_capacity()
    credit,pending = 0,0
    if bounds is not None:
        budget,cores = bounds['memory_budget_bytes'],bounds['cpu_budget_slots']
    elif allocation is not None:
        budget = allocation.memory_bytes
        cores = allocation.cpu_threads
    else:
        credit = private_bytes(own) if own else 0
        pending = sum(max(0,p['ram_bytes']-private_bytes([p])) for p in peers)
        budget = max(0,min(int(facts['total_bytes']*r['ram_fraction']),
            facts['available_bytes']+credit-r['reserve_ram_bytes'])-pending)
        cores = max(0,max(1,math.floor(facts['cores']*r['cpu_fraction']))-sum(p['cpu_slots'] for p in peers))
    available_budget = budget
    requested_margin = r.get('memory_margin_fraction', .5)
    margin = requested_margin
    budget = math.floor(budget / (1 + margin))
    def apply_margin(minimum):
        nonlocal budget, margin
        planned = reserve_margin(minimum, available_budget, requested_margin)
        margin = planned['memory_margin_fraction']
        budget = max(0, int(available_budget)*minimum//planned['memory_bytes']) if minimum else max(0,int(available_budget))
    def finish(result):
        minimum = reserve_margin(result['minimum_bytes'],available_budget,requested_margin)
        result.update(budget_bytes=available_budget, planning_budget_bytes=budget,
                      requested_memory_margin_fraction=requested_margin, memory_margin_fraction=margin,
                      raw_minimum_bytes=result['minimum_bytes'], minimum_bytes=minimum['memory_bytes'])
        if result['admitted']:
            estimated = result['ram_bytes']
            result.update(estimated_ram_bytes=estimated,
                          ram_bytes=min(int(available_budget),math.ceil(estimated * (1 + margin))))
            result['margin_bytes'] = result['ram_bytes'] - estimated
        return result
    reports = estimates if estimates is not None else case_estimates(config, prepared, stage)
    if stage == "preprocess" and config.get("pending_case_ids") is not None:
        reports = [row for row in reports if row["case_id"] in config["pending_case_ids"]]
    if not reports:
        raise ValueError("资源规划没有待处理病例")
    case_bytes = max(row["workspace_bytes"] for row in reports)
    if stage == "preprocess":
        requested = r["max_workers"] if r["workers"] == "auto" else r["workers"]
        readers = r["max_readers"] if r["readers"] == "auto" else r["readers"]
        cap = max(0, min(requested, readers, len(reports), cores - 1))
        minimum = 2 * GIB + case_bytes
        apply_margin(minimum)
        result = {"admitted": False, "facts": facts, "budget_bytes": budget, "own_uss_credit_bytes": credit,
                  "peer_pending_bytes": pending, "minimum_bytes": minimum, "case_estimate_bytes": case_bytes,
                  "case_estimates": reports, "requested": {"workers": r["workers"], "readers": r["readers"]},
                  "workers": [0], "cache_bytes": [0], "prefetch": 1, "cpu_slots": 1,
                  "reader_slots": 1, "workspace_capacity_bytes": case_bytes,
                  "reason": "离线预处理等待 CPU/RAM 安全余量"}
        if cores < 1:
            result.update(reason="已登记CPU额度不足，等待其他任务释放", available_cpu_slots=cores)
            return finish(result)
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
        return finish(result)
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
    stage_reserve = None
    if config.get("model", "stunet") == "stunet":
        from ..training.stage_memory import stage_memory
        stage_reserve = stage_memory(config, prepared, prefetch=1)
        base += stage_reserve["checkpoint_reserve_bytes"] + world * stage_reserve["model_load_per_rank_bytes"]
        if r["device"] == "cpu" and stage not in ('predict','prediction'):
            base += stage_reserve["batch_bytes"]
    if config.get('model', 'stunet') == 'stunet' and stage in ('predict','prediction'):
        # 在扣除进程、模型和50%余量后的实际额度内选择整例或分块。
        reports = [dict(row) for row in reports]
        for row in reports:
            available_case = max(0, budget-base-patch_bytes*4)
            wanted = row['prediction_execution_requested']
            execution = 'whole' if wanted == 'whole' or wanted == 'auto' and row['whole_workspace_bytes'] <= available_case else 'blocks'
            block = max(config['prediction']['block_bytes'], min(config['prediction']['max_block_bytes'],
                max(0, available_case-row['prediction_window_bytes']-256*1024**2)//4))
            row.update(prediction_execution=execution, prediction_block_bytes=block)
            row['workspace_bytes'] = (row['whole_workspace_bytes'] if execution == 'whole' else
                row['prediction_window_bytes']+4*block+256*1024**2)
        case_bytes = max(row['workspace_bytes'] for row in reports)
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
    apply_margin(minimum)
    result = {"admitted": False, "facts": facts, "budget_bytes": budget, "own_uss_credit_bytes": credit,
              "peer_pending_bytes": pending, "minimum_bytes": minimum, "case_estimate_bytes": case_bytes,
              "workers": list(minimum_workers), "cache_bytes": [0]*world, "prefetch": 1,
              "cpu_slots": world+sum(minimum_workers), "reader_slots": 1,
              "workspace_capacity_bytes": case_bytes,
              "case_estimates": reports, "requested": {key: r[key] for key in
                 ("workers", "max_workers", "cache_bytes", "prefetch", "readers", "max_readers")}}
    if profile is not None:
        result['model_memory'] = {key:value for key,value in profile.items() if key!='records'}
    if stage_reserve is not None:
        result["stage_memory"] = stage_reserve
    if cores < world+sum(minimum_workers):
        result.update(reason="已登记CPU额度不足，等待其他任务释放", available_cpu_slots=cores)
        return finish(result)
    if budget < minimum:
        result["reason"] = "单个病例构建及零 worker 的最小预算仍不足，等待主机内存释放"
        return finish(result)
    if fixed_workers is not None and sum(fixed_workers)>cap:
        result['reason'] = '模型恢复需要原worker数量，等待相应CPU供数额度'
        return finish(result)
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
        prefetch = 1
        queues = patch_bytes*3
        extra_stage = 0 if stage_reserve is None or r["device"] != "cpu" or stage in ('predict','prediction') else (
            stage_memory(config, prepared, prefetch=2)["batch_bytes"] - stage_reserve["batch_bytes"])
        local_threads = sum(w == 0 for w in workers)
        if (profile is None and r["prefetch"] >= 2 and world + count + local_threads <= cores and
                fixed + extra_stage + case_bytes + patch_bytes * 4 <= budget):
            prefetch, queues, fixed = 2, patch_bytes * 4, fixed + extra_stage
        if fixed + case_bytes + queues > budget:
            continue
        readers = max(1,min(readers_cap,instances,batch))
        cache = max(0,min(cache_limit,(budget-fixed-queues-case_bytes)//instances))
        workspace = min(readers*case_bytes,budget-fixed-queues-instances*cache)
        used = fixed+queues+workspace+instances*cache
        if used > budget:
            raise AssertionError("供数预算计算未闭合")
        result.update(admitted=True, reason="资源预算允许", workers=workers, cache_bytes=[int(cache)] * world,
                      max_workers=count, cpu_slots=world + count + (local_threads if prefetch == 2 else 0),
                      ram_bytes=int(used), reader_slots=int(readers),
                      prefetch=prefetch,workspace_capacity_bytes=int(workspace), breakdown={"processes_bytes": fixed, "queue_bytes": int(queues),
                      "case_builds": int(readers), "case_workspaces_bytes": int(workspace),
                      "cache_instances": instances, "cache_total_bytes": int(instances * cache)})
        if profile is not None:
            result['breakdown'].update(retained_case_instances=instances,retained_cases_bytes=instances*retained,
                rank_prediction_bytes=sum(profile['rank_case_bytes']))
        if stage_reserve is not None:
            result["stage_memory"] = stage_memory(config, prepared, prefetch=prefetch)
        return finish(result)
    raise AssertionError("零 worker 最小预算未被实际规划覆盖")
