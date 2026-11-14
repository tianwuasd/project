from contextlib import contextmanager
import csv
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock
import nibabel as nib
import numpy as np
import psutil

from ..artifacts.store import digest, file_digest, read_json, write_json

GIB = 1024 ** 3


def layout(batch, world, manual=None):
    shares = manual if manual is not None else [batch // world + int(i < batch % world) for i in range(world)]
    if not 1 <= world <= 4 or len(shares) != world or min(shares) < 1 or sum(shares) != batch:
        raise ValueError("batch 分担必须为 1–4 个非空份额且合计等于全局 batch")
    return shares


def host_capacity():
    process = psutil.Process()
    cores = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else psutil.cpu_count()
    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    if sys.platform == "linux":
        quota = Path("/sys/fs/cgroup/cpu.max")
        if quota.is_file():
            maximum, period = quota.read_text().split()
            if maximum != "max":
                cores = min(cores, max(1, int(maximum) // int(period)))
        limit, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if limit.is_file() and limit.read_text().strip() != "max":
            total = min(total, int(limit.read_text()))
            available = min(available, max(0, int(limit.read_text()) - int(current.read_text())))
    return {"cores": cores, "total_bytes": total, "available_bytes": available}


def cpu_plan(config, prepared, world, peers=()):
    r = config["resources"]
    facts = host_capacity()
    cores = max(1, math.floor(facts["cores"] * r["cpu_fraction"])) - sum(p["cpu_slots"] for p in peers)
    available = min(int(facts["total_bytes"] * r["ram_fraction"]), facts["available_bytes"] - r["reserve_ram_bytes"])
    # 已占用的内存体现在 available 中，只额外扣除其他任务尚未使用的承诺额度。
    for p in peers:
        private = sum(psutil.Process(m["pid"]).memory_full_info().uss for m in [p, *p.get("members", [])] if alive(m))
        available -= max(0, p["ram_bytes"] - private)
    roi = config["technical"]["roi_xyz"]
    target = np.array(prepared["plan"]["configurations"]["3d_fullres"]["spacing"])
    estimates = []
    for row in prepared["manifest"]["cases"]:
        shape = np.array([end - begin for begin, end in roi] if roi else nib.load(row["image"]).shape)
        output_shape = np.ceil(shape[::-1][prepared["plan"]["transpose_forward"]] *
            np.array(row["spacing_xyz"])[::-1][prepared["plan"]["transpose_forward"]] / target)
        estimates.append(int(np.prod(shape)) * 32 + int(np.prod(output_shape)) * 64)
    case_bytes = max(estimates)
    patch_bytes = int(np.prod(config["training"]["patch"])) * 16 * config["training"]["batch"]
    resident = GIB + world * (GIB + case_bytes)
    if available < resident + patch_bytes * r["prefetch"] or cores < world + 1:
        raise MemoryError("CPU/内存预算不足以启动训练进程；请减少卡数或等待资源释放")
    cap = r["max_workers"] if r["workers"] == "auto" else min(r["workers"], r["max_workers"])
    cache = min(r["cache_bytes"], (available - resident - patch_bytes * r["prefetch"]) // world)
    shares = layout(config["training"]["batch"], world, r["batch_layout"])
    def estimate(workers):
        queues = sum(patch_bytes * shares[i] / config["training"]["batch"] * max(r["prefetch"], w)
                     for i, w in enumerate(workers))
        return int(resident + queues + sum(w * (GIB + case_bytes + cache) + (cache if w == 0 else 0) for w in workers))
    workers = [0] * world
    for number in range(min(cap, max(0, cores - world - 1))):
        candidate = workers.copy()
        candidate[number % world] += 1
        if estimate(candidate) > available:
            break
        workers = candidate
    maximum, caches = sum(workers), [int(cache)] * world
    return {"facts": facts, "workers": workers, "cache_bytes": caches, "max_workers": maximum,
            "cpu_slots": world + maximum + 1, "ram_bytes": estimate(workers),
            "reader_slots": min(r["max_readers"] if r["readers"] == "auto" else r["readers"],
                                max(1, maximum + world)), "case_estimate_bytes": case_bytes}


def inventory():
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    rows = [{"index": int(i), "uuid": u.strip(), "total": int(t) * 1024 ** 2,
             "free": int(f) * 1024 ** 2, "utilization": int(v)}
            for i, u, t, f, v in csv.reader(io.StringIO(raw))]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout
    busy = {row[0].strip() for row in csv.reader(io.StringIO(apps)) if row}
    for row in rows:
        row["busy"] = row["uuid"] in busy
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [v.strip() for v in visible.split(",") if v.strip()]
        if any(not (v.isdigit() or v.startswith("GPU-")) for v in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES 仅支持物理编号或完整 GPU UUID")
        rows = [r for r in rows if str(r["index"]) in tokens or r["uuid"] in tokens]
    return rows


def capacity_key(config, spec, local_batch):
    from importlib.metadata import version
    return digest({"model": "stunet", "spec": spec, "batch": local_batch,
                   "precision": config["training"]["precision"],
                   "world": config["resources"]["gpu_count"],
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


def choose_gpus(config, spec, history, peaks, peers):
    r = config["resources"]
    current = history[-1][1]
    requested = r["gpu_ids"]
    selected = current if requested is None else [next((g for g in current if str(g["index"]) == str(token)
                       or g["uuid"] == str(token)), None) for token in requested]
    if any(g is None for g in selected):
        raise ValueError("手动显卡超出 CUDA_VISIBLE_DEVICES 或当前服务器范围")
    if len(selected) < r["gpu_count"]:
        raise ValueError("可见显卡不足请求数量")
    reserved = {u for p in peers for u in p["uuids"]}
    available = [g for g in selected if g["uuid"] not in reserved]
    idle = [g for g in available if not g["busy"] and
            g["free"] >= max(r["vram_reserve_bytes"], int(g["total"] * r["vram_reserve_fraction"]))]
    if len(idle) >= r["gpu_count"]:
        return idle[:r["gpu_count"]] if requested else sorted(idle, key=lambda g: -g["free"])[:r["gpu_count"]]
    # 只在全部允许显卡均有任务时启用共享；存在空闲卡时等待足够数量的空闲卡。
    if not r["share"] or not all(g["busy"] for g in current) or any(not g["busy"] for g in selected):
        return None
    if history[-1][0] - history[0][0] < r["share_window_seconds"]:
        return None
    shares = layout(config["training"]["batch"], r["gpu_count"], r["batch_layout"])
    admitted = []
    for rank, local_batch in enumerate(shares):
        candidates = []
        for g in ([selected[rank]] if requested else available):
            if g not in available:
                continue
            if g in admitted:
                continue
            key = capacity_key(config, spec, local_batch)
            measured = peaks.get(g["uuid"], {}).get(key)
            samples = [next((a for a in snapshot if a["uuid"] == g["uuid"]), None) for _, snapshot in history]
            if measured is None or any(a is None or not a["busy"] for a in samples):
                continue
            reserve = max(r["vram_reserve_bytes"], int(g["total"] * r["vram_reserve_fraction"]))
            if (np.percentile([a["utilization"] for a in samples], 90) <= r["share_utilization"] and
                    min(a["free"] for a in samples) >= measured * r["peak_margin"] + reserve):
                candidates.append(g)
        if not candidates:
            return None
        admitted.append(candidates[0] if requested else max(candidates, key=lambda g: g["free"]))
    if requested and [g["uuid"] for g in admitted] != [g["uuid"] for g in selected]:
        return None
    return admitted


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
def resource_lease(config, prepared, spec):
    root = Path(config["work_root"]) / "resources"
    root.mkdir(parents=True, exist_ok=True)
    registry, lock = root / "active.json", root / ".resource.lock"
    me = psutil.Process()
    started, history = time.monotonic(), []
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
            gpus = choose_gpus(config, spec, history, peaks, peers) if history else []
            if gpus is not None:
                world = len(gpus) or config["resources"]["cpu_processes"]
                supply = cpu_plan(config, prepared, world, peers)
                record = {"pid": me.pid, "birth": me.create_time(), "uuids": [g["uuid"] for g in gpus],
                          "cpu_slots": supply["cpu_slots"], "ram_bytes": supply["ram_bytes"]}
                write_json(registry, [*peers, record])
                break
        elapsed = time.monotonic() - started
        if elapsed >= config["resources"]["wait_seconds"]:
            raise TimeoutError("没有符合条件的显卡：空闲优先；全部繁忙时需要同配方实测显存及低负载窗口")
        print(f"等待显卡资源，已等待 {elapsed:.0f} 秒", file=sys.stderr, flush=True)
        time.sleep(min(config["resources"]["poll_seconds"], config["resources"]["wait_seconds"] - elapsed))
    try:
        yield {"gpus": gpus, "supply": supply, "world": world, "owner": {"pid": me.pid, "birth": me.create_time()}}
    finally:
        with FileLock(str(lock)):
            write_json(registry, [p for p in active_peers(registry) if p["pid"] != me.pid or p["birth"] != me.create_time()])


def record_peaks(config, spec, gpus, shares, measurements):
    if not gpus:
        return
    root = Path(config["work_root"]) / "resources"
    with FileLock(str(root / ".resource.lock")):
        path = root / "gpu-peaks.json"
        peaks = read_json(path) if path.exists() else {}
        for gpu, batch, measurement in zip(gpus, shares, measurements):
            key = capacity_key(config, spec, batch)
            previous = peaks.setdefault(gpu["uuid"], {}).get(key, 0)
            peaks[gpu["uuid"]][key] = max(previous, measurement)
        write_json(path, peaks)
