from contextlib import contextmanager
import csv
import io
import math
import os
from pathlib import Path
import subprocess
import time

from filelock import FileLock
import numpy as np
import psutil
from scipy.optimize import linear_sum_assignment

from ..artifacts.store import digest, file_digest, read_json, write_json
from ..observability.events import event
from .budget import case_estimates, cpu_plan
from .host import host_capacity

GIB = 1024 ** 3


def layout(batch, world, manual=None):
    shares = manual if manual is not None else [batch // world + int(i < batch % world) for i in range(world)]
    if not 1 <= world <= 4 or len(shares) != world or min(shares) < 1 or sum(shares) != batch:
        raise ValueError("batch 分担必须为 1–4 个非空份额且合计等于全局 batch")
    return shares


def inventory(owned_pids=(), visible_devices="inherited"):
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    rows = [{"index": int(i), "uuid": u.strip(), "total": int(t) * 1024 ** 2,
             "free": int(f) * 1024 ** 2, "utilization": int(v)}
            for i, u, t, f, v in csv.reader(io.StringIO(raw))]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout
    applications = list(csv.reader(io.StringIO(apps)))
    busy = {row[0].strip() for row in applications if row and int(row[1]) not in owned_pids}
    owned = {row[0].strip() for row in applications if row and int(row[1]) in owned_pids}
    for row in rows:
        row["busy"] = row["uuid"] in busy
        row["owned"] = row["uuid"] in owned
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices == "inherited" else visible_devices
    if visible is not None:
        tokens = [v.strip() for v in visible.split(",") if v.strip()]
        if any(not (v.isdigit() or v.startswith("GPU-")) for v in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES 仅支持物理编号或完整 GPU UUID")
        rows = [r for r in rows if str(r["index"]) in tokens or r["uuid"] in tokens]
    return rows


def capacity_key(config, spec, local_batch, world=None):
    from importlib.metadata import version
    return digest({"model": "stunet", "spec": spec, "batch": local_batch,
                   "precision": config["training"]["precision"],
                   "world": config["resources"]["gpu_count"] if world is None else world,
                   "implementation": [file_digest(Path(__file__).with_name("engine.py")),
                                      file_digest(Path(__file__).parents[1] / "losses" / "partial.py")],
                   "versions": {name: version(name) for name in ("torch", "monai", "nnunetv2")}})


def process_vram(uuid, pid):
    raw = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    entries = [int(amount) * 1024 ** 2 for gpu, process, amount in csv.reader(io.StringIO(raw))
               if gpu.strip() == uuid and int(process) == pid]
    if len(entries) != 1:
        raise RuntimeError("无法读取本训练进程的实际显存占用，停止记录共享准入依据")
    return entries[0]


def gpu_decision(config, spec, history, peaks, peers, world=None, require_measured=False):
    r = config["resources"]
    current = history[-1][1]
    requested = r["gpu_ids"]
    selected = current if requested is None else [next((g for g in current if str(g["index"]) == str(token)
                       or g["uuid"] == str(token)), None) for token in requested]
    if any(g is None for g in selected):
        raise ValueError("手动显卡超出 CUDA_VISIBLE_DEVICES 或当前服务器范围")
    reserved = {u for p in peers for u in p["uuids"]}
    world = r["gpu_count"] if world is None else world
    shares = layout(config["training"]["batch"], world, r["batch_layout"])
    reports = [{"index": g["index"], "uuid": g["uuid"], "busy": g["busy"], "free_bytes": g["free"],
                "utilization": g["utilization"], "ranks": {}} for g in selected]
    result = {"gpus": None, "candidates": reports, "visible_indices": [g["index"] for g in current],
              "requested_ids": requested, "reason": "没有满足全部训练份额的显卡组合"}
    if len(selected) < world:
        result["reason"] = "当前可见显卡数量不足请求数量"
        return result
    allowed = np.zeros((len(shares), len(selected)), dtype=bool)
    keys = {}
    for rank, local_batch in enumerate(shares):
        for column, g in enumerate(selected):
            reserve = max(r["vram_reserve_bytes"], int(g["total"] * r["vram_reserve_fraction"]))
            if requested is not None and world == r["gpu_count"] and column != rank:
                reason = "手动显卡顺序对应其他训练进程"
            elif g["uuid"] in reserved:
                reason = "本项目其他活动任务已预留"
            elif not g["busy"]:
                measured = peaks.get(g["uuid"], {}).get(capacity_key(config, spec, local_batch, world)) if require_measured else 0
                if measured is None:
                    reason = "自动调整卡数需要此卡同配方和每卡 batch 的实测容量"
                elif require_measured and (type(measured) not in (int, float) or not math.isfinite(measured) or measured <= 0):
                    raise ValueError("GPU 显存峰值记录必须为有限正数")
                elif g["free"] < measured * r["peak_margin"] + reserve:
                    reason = "空闲显卡未达到训练容量及显存安全预留"
                else:
                    reason = "eligible_idle"
            elif not r["share"]:
                reason = "当前配置关闭显卡共享"
            else:
                if local_batch not in keys:
                    keys[local_batch] = capacity_key(config, spec, local_batch, world)
                measured = peaks.get(g["uuid"], {}).get(keys[local_batch])
                samples = [next((a for a in snapshot if a["uuid"] == g["uuid"]), None) for _, snapshot in history]
                if measured is None:
                    reason = "缺少此显卡同配方的显存峰值记录"
                elif type(measured) not in (int, float) or not math.isfinite(measured) or measured <= 0:
                    raise ValueError("GPU 显存峰值记录必须为有限正数")
                elif history[-1][0] - history[0][0] < r["share_window_seconds"] or any(a is None or not a["busy"] for a in samples):
                    reason = "繁忙显卡的连续观察窗口不足"
                elif np.percentile([a["utilization"] for a in samples], 90) > r["share_utilization"]:
                    reason = "繁忙显卡的窗口利用率超过共享阈值"
                elif min(a["free"] for a in samples) < measured * r["peak_margin"] + reserve:
                    reason = "繁忙显卡的窗口剩余显存不足"
                else:
                    reason = "eligible_shared"
            reports[column]["ranks"][str(rank)] = {"batch": local_batch, "reason": reason}
            allowed[rank, column] = reason in ("eligible_idle", "eligible_shared")
    # 有效组合先最少使用繁忙卡，再比较空闲显存；同时解决不同每卡 batch 的完整匹配。
    size = len(shares)
    maximum_free = max(1, max(g["free"] for g in selected))
    costs = np.full(allowed.shape, (size + 1) ** 3, dtype=float)
    for column, g in enumerate(selected):
        costs[allowed[:, column], column] = ((size + 1 if g["busy"] else 0) -
            (2 if g.get("owned") else 0) - g["free"] / maximum_free)
    ranks, columns = linear_sum_assignment(costs)
    if not allowed[ranks, columns].all():
        return result
    result["gpus"] = [selected[column] for column in columns]
    result["reason"] = "eligible"
    return result


def choose_gpus(config, spec, history, peaks, peers):
    return gpu_decision(config, spec, history, peaks, peers)["gpus"]


def fleet_decision(config, spec, history, peaks, peers, previous_world=None):
    r = config["resources"]
    minimum = r["gpu_count"] if r["min_gpus"] is None else r["min_gpus"]
    decisions = []
    for world in range(r["gpu_count"], minimum - 1, -1):
        decision = gpu_decision(config, spec, history, peaks, peers, world,
                                require_measured=world != r["gpu_count"] or
                                previous_world is not None and world != previous_world)
        decisions.append({"world": world, "reason": decision["reason"], "candidates": decision["candidates"]})
        if decision["gpus"] is not None:
            decision["alternatives"] = decisions
            return decision
    decision["alternatives"] = decisions
    return decision


def refresh_supply(config, prepared, allocation, apply=False):
    root = Path(config["work_root"]) / "resources"
    with FileLock(str(root / ".resource.lock")):
        rows = active_peers(root / "active.json")
        owner = allocation["owner"]
        own = [p for p in rows if p["pid"] == owner["pid"] and p["birth"] == owner["birth"]]
        if len(own) != 1:
            raise RuntimeError("训练资源所有权已经失效")
        peers = [p for p in rows if p not in own]
        result = cpu_plan(config, prepared, allocation["world"], peers, own=own,
                          estimates=allocation["supply"]["case_estimates"])
        if apply and result["admitted"]:
            own[0].update(cpu_slots=result["cpu_slots"], ram_bytes=result["ram_bytes"])
            write_json(root / "active.json", rows)
        return result


def available_world(config, spec, allocation):
    if not allocation["gpus"]:
        return allocation["world"]
    root = Path(config["work_root"]) / "resources"
    with FileLock(str(root / ".resource.lock")):
        rows = active_peers(root / "active.json")
        owner = allocation["owner"]
        own = [p for p in rows if p["pid"] == owner["pid"] and p["birth"] == owner["birth"]]
        if len(own) != 1:
            raise RuntimeError("训练资源所有权已经失效")
        peers = [p for p in rows if p not in own]
        owned = {m["pid"] for p in own for m in [p, *p.get("members", [])] if alive(m)}
        peaks_path = root / "gpu-peaks.json"
        peaks = read_json(peaks_path) if peaks_path.exists() else {}
        history = allocation.setdefault("fleet_history", [])
        history.append((time.monotonic(), inventory(owned, allocation.get("allowed_visible_devices"))))
        window = config["resources"]["share_window_seconds"]
        while len(history) > 1 and history[-1][0] - history[1][0] >= window:
            history.pop(0)
        decision = fleet_decision(config, spec, history, peaks, peers, allocation["world"])
        desired = len(decision["gpus"]) if decision["gpus"] else 0
        allocation["fleet_replacement"] = desired == allocation["world"] and (
            {g["uuid"] for g in decision["gpus"]} != {g["uuid"] for g in allocation["gpus"]})
        if desired > allocation["world"]:
            signature = [g["uuid"] for g in decision["gpus"]]
            previous = own[0].get("growth_candidate")
            if previous is None or previous["uuids"] != signature:
                own[0]["growth_candidate"] = {"uuids": signature, "since": time.monotonic()}
                write_json(root / "active.json", rows)
                return allocation["world"]
            if time.monotonic() - previous["since"] < config["resources"]["growth_seconds"]:
                return allocation["world"]
        elif own and "growth_candidate" in own[0]:
            del own[0]["growth_candidate"]
            write_json(root / "active.json", rows)
    return desired


def alive(record):
    return psutil.pid_exists(record["pid"]) and psutil.Process(record["pid"]).create_time() == record["birth"]


def active_peers(path):
    values = read_json(path) if path.exists() else []
    return [row for row in values if alive(row) or any(alive(m) for m in row.get("members", []))]


def register_member(config, owner):
    root = Path(config["work_root"]) / "resources"
    with FileLock(str(root / ".resource.lock")):
        rows = active_peers(root / "active.json")
        record = next(p for p in rows if p["pid"] == owner["pid"] and p["birth"] == owner["birth"])
        me = psutil.Process()
        record.setdefault("members", []).append({"pid": me.pid, "birth": me.create_time()})
        write_json(root / "active.json", rows)


def close_members(config, owner):
    path = Path(config["work_root"]) / "resources" / "active.json"
    rows = read_json(path)
    record = next(p for p in rows if p["pid"] == owner["pid"] and p["birth"] == owner["birth"])
    members = [psutil.Process(m["pid"]) for m in record.get("members", []) if alive(m)]
    for process in members:
        process.terminate()
    _, living = psutil.wait_procs(members, timeout=5)
    for process in living:
        process.kill()
    psutil.wait_procs(living, timeout=5)


@contextmanager
def resource_lease(config, prepared, spec, stage="train", previous_world=None):
    root = Path(config["work_root"]) / "resources"
    root.mkdir(parents=True, exist_ok=True)
    registry, lock = root / "active.json", root / ".resource.lock"
    me = psutil.Process()
    started, history, last_log = time.monotonic(), [], None
    resources = config["resources"]
    limit = resources["wait_seconds"]
    estimates = case_estimates(config, prepared, stage) if prepared else None
    event(config["run_dir"], "resources_requested", stage=stage, device=config["resources"]["device"],
          gpu_count=config["resources"]["gpu_count"], gpu_ids=config["resources"]["gpu_ids"],
          workers=resources["workers"], batch=config["training"]["batch"], wait_seconds=limit,
          inherited_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    while True:
        if config["resources"]["device"] == "cuda":
            history.append((time.monotonic(), inventory()))
            window = config["resources"]["share_window_seconds"]
            while len(history) > 1 and history[-1][0] - history[1][0] >= window:
                history.pop(0)
        with FileLock(str(lock)):
            peers = active_peers(registry)
            peaks_path = root / "gpu-peaks.json"
            peaks = read_json(peaks_path) if peaks_path.exists() else {}
            if history:
                history.append((time.monotonic(), inventory()))
            decision = fleet_decision(config, spec, history, peaks, peers, previous_world) if history else {
                "gpus": [], "candidates": [], "visible_indices": [], "requested_ids": None, "reason": "CPU 准入"}
            gpus = decision["gpus"]
            supply = None
            if gpus is not None:
                world = len(gpus) or config["resources"]["cpu_processes"]
                supply = cpu_plan(config, prepared, world, peers, stage, estimates=estimates)
            if supply is not None and supply["admitted"]:
                record = {"pid": me.pid, "birth": me.create_time(), "uuids": [g["uuid"] for g in gpus],
                          "cpu_slots": supply["cpu_slots"], "ram_bytes": supply["ram_bytes"]}
                write_json(registry, [*peers, record])
                event(config["run_dir"], "resources_allocated", stage=stage, devices=gpus,
                      cpu=supply, world_size=world, seconds=time.monotonic() - started)
                break
        elapsed = time.monotonic() - started
        interval = config.get("logging", {}).get("heartbeat_seconds", 15)
        if last_log is None or elapsed - last_log >= interval:
            event(config["run_dir"], "waiting_resources", stage=stage, seconds=elapsed, timeout_seconds=limit,
                  waiting_forever=limit is None, observed_gpus=history[-1][1] if history else [], candidates=decision["candidates"],
                  visible_indices=decision["visible_indices"], requested_ids=decision["requested_ids"],
                  reason=supply["reason"] if supply is not None else decision["reason"], cpu=supply,
                  alternatives=decision.get("alternatives"), sharing=resources["share"])
            last_log = elapsed
        if limit is not None and elapsed >= limit:
            raise TimeoutError("已达到手动设置的资源等待时限；显卡和 CPU/RAM 缺口见 waiting_resources 日志")
        delay = resources["poll_seconds"] if limit is None else min(resources["poll_seconds"], limit - elapsed)
        time.sleep(delay)
    try:
        yield {"gpus": gpus, "supply": supply, "world": world, "owner": {"pid": me.pid, "birth": me.create_time()}}
    finally:
        with FileLock(str(lock)):
            write_json(registry, [p for p in active_peers(registry) if p["pid"] != me.pid or p["birth"] != me.create_time()])
        event(config["run_dir"], "resources_released", stage=stage, seconds=time.monotonic() - started)


def record_peaks(config, spec, gpus, shares, measurements):
    if not gpus:
        return
    root = Path(config["work_root"]) / "resources"
    with FileLock(str(root / ".resource.lock")):
        path = root / "gpu-peaks.json"
        peaks = read_json(path) if path.exists() else {}
        for gpu, batch, measurement in zip(gpus, shares, measurements):
            key = capacity_key(config, spec, batch, len(gpus))
            previous = peaks.setdefault(gpu["uuid"], {}).get(key, 0)
            peaks[gpu["uuid"]][key] = max(previous, measurement)
        write_json(path, peaks)
