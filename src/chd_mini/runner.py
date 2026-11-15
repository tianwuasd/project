from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from time import perf_counter, sleep
import traceback

from filelock import FileLock
import psutil

from .contracts import (TOOL_ROOT, confined, file_sha, git, load_settings, read_json, reference_snapshot,
                        verify_engine, verify_references, write_json)
from .processes import stop_owned
from .retention import cap_logs


MAX_FILES = 500
MAX_WORK_BYTES = 4 * 1024 ** 3
MAX_LOG_BYTES = 20 * 1024 ** 2
STAGE_SECONDS = 900
LOG_RESERVE = 1024 ** 2


def copy_bounded(stream, destination, limit, overflow=None):
    written, omitted = 0, 0
    for block in iter(lambda: stream.read(4096), b""):
        allowed = min(len(block), max(0, limit - written))
        destination.write(block[:allowed])
        destination.flush()
        written += allowed
        omitted += len(block) - allowed
        if omitted and overflow is not None:
            overflow.set()
    return {"written_bytes": written, "omitted_bytes": omitted}


def inventory(root):
    root = Path(root)
    if not root.exists():
        return []
    result = []
    for path in root.rglob("*"):
        confined(path, root)
        if path.is_file():
            result.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    return result


def processes():
    return {p.pid: p for p in psutil.process_iter(["pid", "ppid", "create_time", "memory_info"], ad_value=None)
            if p.info["create_time"] is not None}


class ProcessGroup:
    def __init__(self, logs):
        self.logs = logs
        self.owned = {}
        self.peak_rss = 0

    def sample(self, child_pid):
        table = processes()
        if child_pid in table:
            self.owned.setdefault(child_pid, table[child_pid].info["create_time"])
        changed = True
        while changed:
            changed = False
            for pid, proc in table.items():
                parent = proc.info["ppid"]
                if (pid not in self.owned and parent in self.owned and parent in table and
                        table[parent].info["create_time"] == self.owned[parent]):
                    self.owned[pid] = proc.info["create_time"]
                    changed = True
        live = [p for pid, p in table.items() if pid in self.owned and p.info["create_time"] == self.owned[pid]]
        rss = sum(p.info["memory_info"].rss for p in live if p.info["memory_info"] is not None)
        self.peak_rss = max(self.peak_rss, rss)
        write_json(self.logs / "processes.json", {"supervisor": {"pid": os.getpid(), "birth": psutil.Process().create_time()},
                                                  "owned": [{"pid": p, "birth": b} for p, b in self.owned.items()],
                                                  "sampled_peak_rss_bytes": self.peak_rss})
        return live

    def stop(self):
        stop_owned(self.owned)


def event_totals(path):
    events = []
    if path.exists():
        for segment in [*sorted(path.parent.glob("events.jsonl.[0-9]"), reverse=True), path]:
            events.extend(json.loads(line) for line in segment.read_text(encoding="utf-8").splitlines() if line.strip())
    stages = defaultdict(lambda: {"count": 0, "seconds": 0.0})
    for row in events:
        if row["event"] == "stage_completed":
            item = stages[row["stage"]]
            item["count"] += 1
            item["seconds"] += row["elapsed_seconds"]
    return events, dict(stages)


class RunWorkspace:
    def __init__(self, settings, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise ValueError("运行编号格式无效")
        self.settings, self.run_id = settings, run_id
        self.base = confined(TOOL_ROOT / ".runs" / run_id, TOOL_ROOT)
        if self.base.exists():
            raise FileExistsError("运行编号已使用，请指定新的编号")
        self.work, self.logs = self.base / "work", self.base / "logs"
        self.stages, self.peak_bytes, self.peak_files = [], 0, 0

    def __enter__(self):
        self.started = perf_counter()
        self.baseline = verify_engine(self.settings["engine_root"], self.settings["engine_commit"])
        self.references = {key: self.settings[key] for key in ("manifest", "split", "pretrained")}
        self.input_snapshot = reference_snapshot(self.references)
        self.work.mkdir(parents=True)
        self.logs.mkdir()
        self.group = ProcessGroup(self.logs)
        self.request = {"settings": self.settings, "work": str(self.work), "logs": str(self.logs)}
        self.request_path = self.work / "request.json"
        write_json(self.request_path, self.request)
        write_json(self.logs / "ownership.json", {"run_id": self.run_id, "scope": "mini-workflow",
                    "owned_paths": [str(self.work)], "shared_references": {key: self.settings[key] for key in
                    ("engine_root", "python", "manifest", "split", "data_root", "pretrained")},
                    "retention": {"work_files": MAX_FILES, "work_bytes": MAX_WORK_BYTES, "logs_bytes": MAX_LOG_BYTES,
                                  "cleanup": "完成或失败后自动清理工作文件，必要日志保留至用户主动删除"},
                    "cleanup_state": "pending", "engine": self.baseline, "input_snapshot": self.input_snapshot})
        return self

    def budget(self):
        rows = inventory(self.work)
        amount = sum(r["bytes"] for r in rows)
        self.peak_bytes, self.peak_files = max(self.peak_bytes, amount), max(self.peak_files, len(rows))
        if len(rows) > MAX_FILES or amount > MAX_WORK_BYTES:
            raise RuntimeError("本任务工作文件超过已批准数量或容量")
        if sum(r["bytes"] for r in inventory(self.logs)) > MAX_LOG_BYTES:
            raise RuntimeError("本任务日志超过已批准容量")

    def stage(self, name):
        verify_references(self.references, self.input_snapshot)
        if verify_engine(self.settings["engine_root"], self.settings["engine_commit"]) != self.baseline:
            raise ValueError("阶段前的生产依赖已经改变")
        command = [self.settings["python"], "-u", "-B", str(TOOL_ROOT / "run.py"), "_worker",
                   "--request", str(self.request_path), "--stage", name]
        env = dict(os.environ)
        cache = self.work / "cache"
        cache.mkdir(exist_ok=True)
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   MPLCONFIGDIR=str(cache / "matplotlib"), XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache / "torch"),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TEMP=str(cache), TMP=str(cache),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        mark = perf_counter()
        invocation = {"stage": name, "state": "started", "command": command}
        self.stages.append(invocation)
        print(json.dumps({"stage": name, "state": "started", "time_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False), flush=True)
        available = MAX_LOG_BYTES - LOG_RESERVE - sum(r["bytes"] for r in inventory(self.logs))
        if available <= 0:
            raise RuntimeError("日志容量不足以启动下一个阶段")
        overflow = threading.Event()
        with (self.logs / (name + ".log")).open("wb") as output, ThreadPoolExecutor(max_workers=1) as pool:
            child = subprocess.Popen(command, cwd=self.work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            reader = pool.submit(copy_bounded, child.stdout, output, available, overflow)
            try:
                while child.poll() is None:
                    self.group.sample(child.pid)
                    self.budget()
                    if overflow.is_set():
                        raise RuntimeError("阶段输出超过日志容量，已停止并保留有界诊断")
                    if perf_counter() - mark > STAGE_SECONDS:
                        raise TimeoutError("工作阶段超过900秒有界等待：" + name)
                    sleep(0.25)
                code = child.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                self.group.stop()
                child.wait()
                invocation.update(wall_seconds=perf_counter() - mark, returncode=child.returncode,
                                  state="completed" if child.returncode == 0 and not overflow.is_set() else "failed",
                                  console=reader.result())
                child.stdout.close()
        if overflow.is_set():
            raise RuntimeError("阶段输出超过日志容量，已保留有界诊断")
        elapsed = perf_counter() - mark
        self.budget()
        verify_references(self.references, self.input_snapshot)
        if verify_engine(self.settings["engine_root"], self.settings["engine_commit"]) != self.baseline:
            raise ValueError("阶段后的生产依赖已经改变")
        record = read_json(self.logs / (name + ".json"))
        invocation.update(worker_seconds=record["worker_seconds"], startup_and_exit_seconds=elapsed - record["worker_seconds"])
        print(json.dumps({"stage": name, "state": "completed", "wall_seconds": elapsed}, ensure_ascii=False), flush=True)

    def __exit__(self, kind, error, trace):
        write_json(self.logs / "summary.json", {"state": "FAIL", "run_id": self.run_id,
                   "error": None if kind is None else "".join(traceback.format_exception(kind, error, trace)),
                   "stages": self.stages, "settings": self.settings, "input_snapshot": self.input_snapshot,
                   "total_seconds": perf_counter() - self.started, "finalization": "pending"})
        try:
            self.group.stop()
            self.retain_raw_events()
            self.finish_report(kind, error, trace)
        finally:
            active_error = sys.exc_info()
            if active_error[0] is not None:
                details = "".join(traceback.format_exception(*active_error))
                previous = read_json(self.logs / "summary.json")
                previous.update(state="FAIL", finalization="failed", finalization_error=details)
                write_json(self.logs / "summary.json", previous)
                write_json(self.logs / "finalization-error.json", {"error": details})
            self.cleanup()
        return False

    def retain_raw_events(self):
        source = self.work / "workspace" / "training" / "events.jsonl"
        if not source.exists():
            return
        available = max(0, MAX_LOG_BYTES - LOG_RESERVE - sum(r["bytes"] for r in inventory(self.logs)))
        omitted = 0
        with (self.logs / "production-events.raw.log").open("wb") as output:
            for segment in [*sorted(source.parent.glob("events.jsonl.[0-9]"), reverse=True), source]:
                with segment.open("rb") as stream:
                    receipt = copy_bounded(stream, output, available)
                available -= receipt["written_bytes"]
                omitted += receipt["omitted_bytes"]
        if omitted:
            raise RuntimeError("原始事件超出保留容量，已保留有界记录；省略字节数：" + str(omitted))

    def finish_report(self, kind, error, trace):
        events, totals = event_totals(self.work / "workspace" / "training" / "events.jsonl")
        if events:
            (self.logs / "production-events.raw.log").replace(self.logs / "production-events.jsonl")
        source_unchanged = None
        selected = self.work / "inputs" / "selected.json"
        if selected.exists():
            source_unchanged = all(file_sha(row[key]) == row["sha256"][key]
                                   for row in read_json(selected)["cases"] for key in ("image", "label"))
        engine_unchanged = verify_engine(self.settings["engine_root"], self.settings["engine_commit"]) == self.baseline
        current_inputs = reference_snapshot(self.references)
        frozen_inputs_unchanged = current_inputs == self.input_snapshot
        training_path = self.logs / "train_resume.json"
        training = read_json(training_path)["result"] if training_path.exists() else None
        rows = inventory(self.work)
        before = sum(r["bytes"] for r in rows)
        summary = {"run_id": self.run_id, "state": "PASS" if kind is None and source_unchanged and engine_unchanged and frozen_inputs_unchanged else "FAIL",
                   "settings": self.settings, "engine": self.baseline, "tool_commit": git(TOOL_ROOT, "rev-parse", "HEAD"),
                   "tool_modified": bool(git(TOOL_ROOT, "status", "--porcelain")),
                   "stages": self.stages, "total_seconds": perf_counter() - self.started,
                   "source_unchanged": source_unchanged, "engine_unchanged": engine_unchanged,
                   "frozen_inputs_unchanged": frozen_inputs_unchanged, "input_snapshot": self.input_snapshot,
                   "final_input_snapshot": current_inputs,
                   "sampled_process_group_peak_rss_bytes": self.group.peak_rss,
                   "ram_sampling_seconds": 0.25, "production_stage_totals": totals,
                   "stage_total_note": "含嵌套、重复读取及并行阶段，不能相加替代总墙钟", 
                   "training": training,
                   "updates_per_second": training["updates"] / sum(s["wall_seconds"] for s in self.stages if s["stage"].startswith("train_")) if training else None,
                   "work_peak_sampled_bytes": self.peak_bytes, "work_peak_sampled_files": self.peak_files,
                   "error": None if kind is None else "".join(traceback.format_exception(kind, error, trace)),
                   "measurement_limits": "RSS按0.25秒采样，显存来自生产训练记录；CPU与CUDA结果各按实际设备标记"}
        write_json(self.logs / "summary.json", summary)
        if kind is None and summary["state"] != "PASS":
            raise ValueError("来源或版本最终核验失败")

    def cleanup(self):
        rows = inventory(self.work)
        before = sum(r["bytes"] for r in rows)
        write_json(self.logs / "cleanup.json", {"state": "pending", "files": rows, "bytes": before})
        confined(self.work, self.base)
        if self.group.sample(-1):
            raise RuntimeError("仍有本任务进程，拒绝清理")
        shutil.rmtree(self.work)
        if self.work.exists():
            raise RuntimeError("工作文件清理不完整")
        retained = inventory(self.logs)
        cleanup = {"state": "pending", "deleted_files": len(rows), "released_bytes": before,
                  "work_removed": True, "owned_processes_exited": True,
                  "retained_files_before_receipt": len(retained), "retained_bytes_before_receipt": sum(r["bytes"] for r in retained)}
        write_json(self.logs / "cleanup.json", cleanup)
        cap_logs(self.logs)
        cleanup["state"] = "completed"
        write_json(self.logs / "cleanup.json", cleanup)
        ownership = read_json(self.logs / "ownership.json")
        ownership["cleanup_state"] = "completed"
        write_json(self.logs / "ownership.json", ownership)


def run(config_path, run_id):
    settings = load_settings(config_path)
    verify_engine(settings["engine_root"], settings["engine_commit"])
    runs = confined(TOOL_ROOT / ".runs", TOOL_ROOT)
    runs.mkdir(exist_ok=True)
    with FileLock(str(runs / ".tool.lock"), timeout=0):
        with RunWorkspace(settings, run_id) as task:
            for name in ("select", "fit", "prepare", "preprocess", "train_start", "train_resume", "test"):
                task.stage(name)
    return read_json(task.logs / "summary.json")
