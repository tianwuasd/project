from datetime import datetime, timezone
import os
from pathlib import Path
import threading
from time import perf_counter

import psutil
from filelock import FileLock

from ..artifacts.store import write_json, read_json
from ..training.resources import inventory
from .events import event


_FIELDS = ("done", "new", "reused", "failures", "pending", "active", "current_case",
           "block_done", "block_total", "epoch", "step", "storage_bytes", "phase", "waiting_reason")


def resource_summary(snapshot):
    budget = snapshot.get("budget", {})
    own = budget.get("own") or {}
    return {"host": {key: snapshot.get("host", {}).get(key) for key in
            ("cores", "total_bytes", "available_bytes", "host_total_bytes", "host_available_bytes")},
        "project": {key: snapshot.get("project", {}).get(key) for key in
            ("process_count", "accounted_bytes", "rss_bytes", "pss_bytes", "uss_bytes", "complete")},
        "budget": {**{key: budget.get(key) for key in ("headroom_bytes", "system_headroom_bytes", "pending_bytes",
                  "target_fraction", "memory_limit_bytes", "reason")},
                   "own": {key: own.get(key) for key in ("process_count", "accounted_bytes", "rss_bytes", "pss_bytes", "uss_bytes")}},
        "peaks": snapshot.get("peaks"), "paused": snapshot.get("paused"), "stop_requested": snapshot.get("stop_requested")}


def phase_name(phase):
    names = {"read_decompress": "读取", "decompress": "解压", "normalize": "归一化", "crop_bounds": "裁剪",
             "prefilter_copy": "预滤波准备", "image_resample": "图像重采样", "label_resample": "标签重采样",
             "write_array": "写盘", "readback": "读回核验", "locations_scan": "前景位置扫描",
             "intensity_bounds": "灰度范围扫描", "label_values": "标签类别扫描",
             "native_window_bounds": "原生灰度窗范围", "native_gray_window": "原生灰度窗转换",
             "image_inplane": "图像平面插值", "label_inplane": "标签平面插值"}
    labels = []
    for step in phase.split(":"):
        if step.startswith("prefilter_axis_"):
            labels.append(f"第{int(step.removeprefix('prefilter_axis_')) + 1}轴预滤波")
        elif step.startswith("fill_holes_pass_"):
            labels.append(f"孔洞填充第{int(step.removeprefix('fill_holes_pass_'))}次")
        else:
            labels.append(names.get(step, step))
    return "／".join(labels)


def console_progress(record):
    def gb(value):
        return "未知" if value is None else f"{value / 1_000_000_000:.2f}GB"
    total = record.get("total")
    done = record.get("done", 0)
    unit = {'cases': '病例', 'steps': '训练步', 'updates': '训练更新', 'epochs': '训练轮次'}.get(
        record.get('unit'), "病例" if record.get("stage") == "preprocess" else "训练步")
    parts = [f"[{record.get('stage')}] {unit} {done}/{total if total is not None else '未知'}",
        f"复用{record.get('reused', 0)} 新完成{record.get('new', 0)} 失败{record.get('failures', 0)} 剩余{record.get('remaining', '未知')}"]
    phase = record.get("phase")
    if record.get("current_case"):
        parts.append(str(record["current_case"]))
    if phase:
        parts.append(phase_name(phase))
    if record.get("plane_index") is not None:
        parts.append(f"切片 {record['plane_index']}/{record['plane_total']}")
    if record.get("block_done") is not None:
        count, limit = record["block_done"], record.get("block_total")
        parts.append(f"块 {count}/{limit if limit is not None else '未知'} 剩余{limit - count if limit is not None else '未知'}")
    execution = record.get("execution")
    if execution:
        parts.append(f"病例线程{execution['applied_case_threads']} 块预算{execution['applied_block_bytes'] / 1024 ** 2:.1f}MiB")
    if record.get("epoch") is not None:
        parts.append(f"epoch {record['epoch']} step {record.get('step')}")
    memory = record.get("resource", {})
    budget = memory.get("budget", {})
    own = budget.get("own") or {}
    if budget:
        parts.append(f"任务内存 {gb(own.get('accounted_bytes'))}/{gb(budget.get('memory_limit_bytes'))} 安全余量{gb(budget.get('headroom_bytes'))}")
    disk = record.get("disk", {})
    if disk:
        parts.append(f"数据已写{gb(disk.get('task_storage_bytes'))} 磁盘剩余{gb(disk.get('data_free_bytes', disk.get('free_bytes')))}")
    if record.get("elapsed_seconds") is not None:
        parts.append(f"耗时{record['elapsed_seconds']:.1f}秒")
    reason = record.get("waiting_reason")
    complete = total is not None and done == total and not record.get('active')
    if not complete and (reason or memory.get("paused")):
        parts.append("等待：" + (reason or "整机或项目内存安全余量"))
    print(" | ".join(parts), flush=True)


def _raise_walk_error(error):
    raise error


class Progress:
    def __init__(self, config, stage, total=None, reused=0, interval_seconds=15,
                 scan_run_dir=False, resource=None, started=None, unit=None):
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage 必须为非空名称")
        if total is not None and (type(total) is not int or total < 0):
            raise ValueError("total 必须为非负整数或 null")
        if type(reused) is not int or reused < 0:
            raise ValueError("reused 必须为非负整数")
        if type(interval_seconds) not in (int, float) or interval_seconds < 15:
            raise ValueError("进度写入间隔不得短于 15 秒")
        self.config = config
        self.stage = stage
        self.unit = unit or ('cases' if stage == 'preprocess' else 'steps')
        if self.unit not in ('cases', 'steps', 'updates', 'epochs'):
            raise ValueError('进度单位必须为cases、steps、updates或epochs')
        self.directory = Path(config["run_dir"])
        self.started = perf_counter() if started is None else started
        self.last_snapshot = 0.0
        self.interval = interval_seconds
        self.guard = threading.RLock()
        self.closed = False
        self.resource = resource
        self.values = {name: None for name in _FIELDS}
        self.values.update(done=0, new=0, reused=reused, failures=0, pending=None, active=0,
                           current_case=None, block_done=None, block_total=None,
                           epoch=None, step=None, storage_bytes=0)
        self.total = total
        self.disk, self.disk_path = self._disk_snapshot(scan_run_dir)
        self.cpu_percent = None
        self.gpus = None
        self._metrics_at = None
        self._last_new = 0
        self._last_new_at = self.started

    def _disk_snapshot(self, enabled):
        run_dir = self.directory
        existing = run_dir
        while not existing.exists():
            if existing.parent == existing:
                raise FileNotFoundError(str(existing))
            existing = existing.parent
        usage = psutil.disk_usage(str(existing))
        result = {"free_bytes": usage.free, "task_storage_bytes": 0,
                  "run_dir_bytes": None, "run_dir_scan_partial": None}
        if enabled and run_dir.is_dir():
            started = perf_counter()
            size, count, visited_dirs, partial = 0, 0, 0, False
            for root, dirs, files in os.walk(run_dir, onerror=_raise_walk_error):
                dirs.sort()
                visited_dirs += 1
                for name in files:
                    path = Path(root) / name
                    size += path.stat().st_size
                    count += 1
                    if count >= 1000 or visited_dirs >= 1000 or perf_counter() - started >= 0.1:
                        partial = True
                        break
                if visited_dirs >= 1000 or perf_counter() - started >= 0.1:
                    partial = True
                if partial:
                    break
            result["run_dir_bytes"] = size
            result["run_dir_scan_partial"] = partial
        return result, existing

    def update(self, *, total=None, resource=None, **values):
        unknown = set(values) - set(_FIELDS)
        if unknown:
            raise ValueError("未知进度字段：" + ", ".join(sorted(unknown)))
        with self.guard:
            if self.closed:
                raise RuntimeError("进度记录已关闭")
            if total is not None:
                if type(total) is not int or total < 0:
                    raise ValueError("total 必须为非负整数")
                self.total = total
            if resource is not None:
                if not isinstance(resource, dict):
                    raise TypeError("resource 必须为字典")
                self.resource = resource
            for key, value in values.items():
                if key in ("done", "new", "reused", "failures", "pending", "active", "block_done", "block_total", "epoch", "step", "storage_bytes"):
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError(key + " 必须为非负整数或 null")
                self.values[key] = value

    def snapshot(self, force=False):
        now = perf_counter()
        with self.guard:
            if self.closed and not force:
                raise RuntimeError("进度记录已关闭")
            if not force and now - self.last_snapshot < self.interval:
                return None
            elapsed = now - self.started
            new = self.values["new"]
            since = now - self._last_new_at
            new_delta = new - self._last_new
            rate = new_delta / since if new_delta > 0 and since > 0 else None
            if new_delta > 0:
                self._last_new, self._last_new_at = new, now
            remaining = self.total - self.values["done"] if self.total is not None else None
            eta = remaining / rate if remaining is not None and remaining >= 0 and rate and rate > 0 else None
            if self._metrics_at is None or now - self._metrics_at >= 15:
                self.cpu_percent = psutil.cpu_percent(interval=None)
                if self.config.get("resources", {}).get("device") == "cuda":
                    self.gpus = inventory()
                self._metrics_at = now
            memory_path = self.directory / "memory.json"
            raw_resource = self.resource or (read_json(memory_path) if memory_path.exists() else {})
            resource = resource_summary(raw_resource) if raw_resource else {}
            resource.update(cpu_percent=self.cpu_percent, gpus=self.gpus)
            usage = psutil.disk_usage(str(self.disk_path))
            disk = {**self.disk, "free_bytes": usage.free,
                    "task_storage_bytes": self.values["storage_bytes"]}
            data = self.config.get("preprocessing", {}).get("root")
            if data:
                existing = Path(data)
                while not existing.exists():
                    existing = existing.parent
                disk["data_free_bytes"] = psutil.disk_usage(str(existing)).free
            record = {"schema": 1, "stage": self.stage, "time_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": elapsed, "total": self.total, **self.values,
                      "remaining": remaining, 'unit': self.unit, "rate_unit": self.unit+'/second',
                      "storage_bytes_estimated": True,
                      "rate_new_per_second": rate, "eta_seconds": eta,
                      "resource": resource, "disk": disk}
            path = self.directory / f"progress-{self.stage}.json"
            with FileLock(str(self.directory / f".progress-{self.stage}.lock")):
                previous = read_json(path) if path.exists() else {}
                active_cases = (record.get('current_case') or '').split(', ')
                if (self.stage == 'preprocess' and record['active'] and
                        previous.get('current_case') in active_cases and previous.get('done') == record['done']):
                    record.update({key: previous[key] for key in ('current_case', 'phase', 'block_done',
                        'block_total', 'block_remaining', 'peak_block_bytes', 'execution', 'plane_index', 'plane_total') if key in previous})
                    record['active_cases'] = active_cases
                write_json(path, record)
            event(self.directory, "progress_snapshot", console=False, **record)
            console_progress(record)
            self.last_snapshot = now
            return record

    def close(self):
        with self.guard:
            if self.closed:
                return
            self.snapshot(force=True)
            self.closed = True
