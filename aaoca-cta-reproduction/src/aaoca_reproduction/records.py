from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from time import perf_counter, sleep

import psutil


RELEASE = "b134e621943f60f20caf6a9272aa85ceeeb441f8"
CLASSIFICATION_TARGET = (215, 215, 85)


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
    value["data_root"] = (path.parent / value.get("data_root", str(value["project_root"] / "data"))).resolve()
    value["course_root"] = (path.parent / value.get("course_root", str(value["project_root"] / "course"))).resolve()
    if value.get("input_manifest") is not None:
        value["input_manifest"] = (path.parent / value["input_manifest"]).resolve()
    for key in ("aaoca_root", "aaoca_pairs", "imagechd_root", "imagechd_manifest"):
        value[key] = (path.parent / value[key]).resolve()
    for key in ("chd1282_root", "chd1282_manifest", "geometry_bindings", "author_archive"):
        if value.get(key) is not None:
            value[key] = (path.parent / value[key]).resolve()
    if value.get("diagnoses") is not None:
        value["diagnoses"] = str((path.parent / value["diagnoses"]).resolve())
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


def classification_for_task(existing, task, prediction_case_ids):
    value = deepcopy(existing)
    # 公共参数继续复用；旧运行输出目录不属于新数据批次。
    for key in ("run_directory", "prediction_directory"):
        value.pop(key, None)
    if existing.get("task") != task:
        for key in ("strategy", "pretrained_detection", "weights"):
            value.pop(key, None)
    else:
        strategies = {"local_from_scratch", "author_detector_finetune"} if task == "aaoca_present" else {"author_detection_transfer", "local_detection_transfer"}
        if "strategy" in value and value["strategy"] not in strategies:
            raise ValueError("已有初始化策略不适用于当前分类任务")
        if any(weight["task"] != task for weight in value.get("weights", [])):
            raise ValueError("已有预测权重的任务声明与当前任务不符")
        if value.get("pretrained_detection") is not None and (
                (task == "aaoca_present" and value.get("strategy") != "author_detector_finetune") or value["pretrained_detection"]["task"] != "aaoca_present"):
            raise ValueError("检测父权重只适用于子任务且必须声明检测任务")
    value.update(task=task, prediction_case_ids=list(prediction_case_ids))
    return value


def event(name, **fields):
    print(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                      "event": name, **fields}, ensure_ascii=False), flush=True)


def wait_memory(config, additional_bytes):
    if additional_bytes + psutil.Process().memory_info().rss > config["max_memory_bytes"]:
        raise MemoryError("本例预计占用超过配置的任务内存预算")
    while True:
        memory = psutil.virtual_memory()
        reserve = memory.total * (1 - config["host_memory_fraction"])
        if memory.available - reserve >= additional_bytes:
            return
        event("waiting_memory", requested_bytes=additional_bytes,
              available_bytes=memory.available, host_reserve_bytes=int(reserve))
        sleep(5)


def admit_stage(config, stage, arrays):
    raw = sum(int(value) for value in arrays.values())
    additional = int(raw * 1.5) + 64 * 2**20
    record = {"stage": stage, "allocation_estimates": arrays, "estimated_additional_bytes": additional,
              "resident_rss_before_bytes": psutil.Process().memory_info().rss,
              "safety_factor": 1.5, "fixed_overhead_bytes": 64 * 2**20, "measured_allocation": False}
    event("memory_admission", **record)
    wait_memory(config, additional)
    return record


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
