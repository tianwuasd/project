from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from time import perf_counter, sleep

import psutil


RELEASE = "b134e621943f60f20caf6a9272aa85ceeeb441f8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def signature(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_config(path):
    path = Path(path).resolve(strict=True)
    value = read_json(path)
    value["project_root"] = (path.parent / value.get("state_root", ".")).resolve()
    for key in ("aaoca_root", "aaoca_pairs", "imagechd_root", "imagechd_manifest"):
        value[key] = (path.parent / value[key]).resolve()
    if not 0 < value["host_memory_fraction"] <= 0.8:
        raise ValueError("本任务整机内存预算必须在0到80%之间")
    if not 0 < value["max_memory_bytes"] <= 100_000_000_000:
        raise ValueError("任务内存预算必须为正数且不超过100GB")
    if value["imagechd_orientation_policy"] != "preserve_header_axis_directions_with_author_spacing":
        raise ValueError("需要明确ImageCHD的作者间距和源头方向约定")
    if any(len(value[key]) != 3 for key in ("roi_mm", "offset_index_mm")):
        raise ValueError("裁剪尺寸和偏移必须为三个轴的毫米值")
    if any(x <= 0 for x in value["roi_mm"]):
        raise ValueError("裁剪尺寸必须为正")
    return value


def event(name, **fields):
    print(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                      "event": name, **fields}, ensure_ascii=False), flush=True)


def wait_memory(config, additional_bytes):
    if additional_bytes > config["max_memory_bytes"]:
        raise MemoryError("本例预计占用超过配置的任务内存预算")
    while True:
        memory = psutil.virtual_memory()
        reserve = memory.total * (1 - config["host_memory_fraction"])
        if memory.available - reserve >= additional_bytes:
            return
        event("waiting_memory", requested_bytes=additional_bytes,
              available_bytes=memory.available, host_reserve_bytes=int(reserve))
        sleep(5)


def check_work(config):
    root = config["project_root"] / "work"
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    # 硬链接只增加目录条目，原始数据的字节不计为本轮新副本。
    size = sum(p.stat().st_size for p in files if p.stat().st_nlink == 1)
    if len(files) > config["max_work_files"] or size > config["max_work_bytes"]:
        raise OSError("本任务work文件数或磁盘预算超出已登记范围")
    return {"files": len(files), "new_bytes": size}


@contextmanager
def measure():
    process = psutil.Process()
    stop = threading.Event()
    result = {"peak_rss_bytes": process.memory_info().rss}

    def sample():
        while not stop.wait(0.05):
            result["peak_rss_bytes"] = max(result["peak_rss_bytes"], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    start = perf_counter()
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join()
        result["seconds"] = perf_counter() - start
