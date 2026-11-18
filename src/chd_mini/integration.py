from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from time import perf_counter, sleep

from filelock import FileLock
import psutil

from .acceptance import summarize_run
from .attempts import AttemptLedger
from .contracts import TOOL_ROOT, confined, git, read_json, verify_engine, write_json
from .model_routes import build_steps, command_for, load_acceptance
from .processes import stop_owned
from .runner import copy_bounded, inventory


def check_approval(settings, path):
    approval = read_json(path)
    if approval["settings"] != settings or not approval["approval_reference"]:
        raise ValueError("产物许可须对应当前完整生效配置和明确批准记录")
    for key in ("max_files", "max_bytes", "max_report_bytes"):
        if type(approval[key]) is not int or approval[key] <= 0:
            raise ValueError("产物许可额度必须为正整数")
    if approval["max_report_bytes"] > 20 * 1024 ** 2:
        raise ValueError("本轮必要报告不得超过20MiB")
    return approval


class IntegrationRun:
    def __init__(self, settings, run_id, approval):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise ValueError("运行编号格式无效")
        self.settings, self.run_id, self.approval = settings, run_id, approval
        self.work = Path(settings["work_root"]) / run_id
        self.reports = Path(settings["report_root"]) / run_id
        self.invocations, self.owned = [], {}
        self.peak_files = self.peak_bytes = 0

    def budget(self):
        rows = inventory(self.work)
        reports = inventory(self.reports)
        total = sum(r["bytes"] for r in rows + reports)
        count = len(rows) + len(reports)
        self.peak_files, self.peak_bytes = max(count, self.peak_files), max(total, self.peak_bytes)
        if total > self.approval["max_bytes"] or count > self.approval["max_files"]:
            raise RuntimeError("实际产物超过批准额度，停止后续步骤")
        if sum(r["bytes"] for r in reports) > self.approval["max_report_bytes"]:
            raise RuntimeError("报告超过批准保留额度")

    def __enter__(self):
        self.started = perf_counter()
        if self.work.exists() or self.reports.exists():
            raise FileExistsError("运行编号已经使用；累计记录保留，禁止以新目录重置本轮训练额度")
        self.versions = {"engine_start": verify_engine(self.settings["engine_root"], self.settings["engine_commit"]),
                         "tool_start": git(TOOL_ROOT, "rev-parse", "HEAD"),
                         "tool_status": git(TOOL_ROOT, "status", "--porcelain")}
        self.work.mkdir(parents=True)
        self.reports.mkdir(parents=True)
        self.ledger = AttemptLedger(self.reports / "attempts.json")
        self.ledger.initialize(self.run_id)
        write_json(self.reports / "ownership.json", {"run_id": self.run_id, "work": str(self.work),
                   "settings": self.settings, "approval": self.approval, "cleanup": "pending"})
        return self

    def stage(self, step):
        self.budget()
        command = command_for(self.settings, step, self.work, self.reports)
        cache = self.work / "cache"
        cache.mkdir(exist_ok=True)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
               "PYTHONIOENCODING": "utf-8", "TEMP": str(cache), "TMP": str(cache),
               "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch"),
               "MPLCONFIGDIR": str(cache / "matplotlib"),
               "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
               "CHD_MINI_ATTEMPT_LEDGER": str(self.ledger.path),
               "CHD_MINI_TOOL_ROOT": str(TOOL_ROOT),
               "CHD_MINI_STAGE": step.get("stage", ""),
               "CHD_MINI_ROLES": str(self.reports / "roles.json")}
        write_json(self.reports / "roles.json", self.settings["roles"])
        row = {"name": step["name"], "kind": step["kind"], "stage": step.get("stage"),
               "command": command, "state": "running"}
        self.invocations.append(row)
        write_json(self.reports / "steps.json", self.invocations)
        started = perf_counter()
        succeeded = False
        overflow = threading.Event()
        available = self.approval["max_report_bytes"] - sum(r["bytes"] for r in inventory(self.reports)) - 1024 ** 2
        if available <= 0:
            raise RuntimeError("缺少日志与收尾报告空间")
        with (self.reports / (step["name"] + ".log")).open("wb") as output, ThreadPoolExecutor(max_workers=1) as pool:
            child = subprocess.Popen(command, cwd=self.settings["engine_root"], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.owned[child.pid] = psutil.Process(child.pid).create_time()
            reader = pool.submit(copy_bounded, child.stdout, output, available, overflow)
            try:
                while child.poll() is None:
                    if overflow.is_set():
                        raise RuntimeError("输出超过批准容量")
                    if perf_counter() - started > step["timeout_seconds"]:
                        raise TimeoutError("阶段超过批准执行时限：" + step["name"])
                    sleep(1)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
                if overflow.is_set():
                    raise RuntimeError("阶段日志发生截断")
                succeeded = True
            finally:
                stop_owned(self.owned)
                child.wait()
                child.stdout.close()
                row.update(wall_seconds=perf_counter() - started, returncode=child.returncode,
                           console=reader.result(), state="awaiting_receipt" if succeeded else "failed")
                write_json(self.reports / "steps.json", self.invocations)
        if overflow.is_set():
            raise RuntimeError("阶段日志发生截断")
        self.budget()
        receipt = read_json(self.reports / step["receipt"])
        if receipt["roles"] != self.settings["roles"] or receipt["state"] != "completed":
            raise ValueError("步骤回执角色或执行状态不符")
        row["receipt"] = receipt
        if step["kind"] == "train":
            state = self.ledger.read()
            rows = [r for r in state["attempts"] if r["stage"] == step["stage"]]
            if not rows or any(r["outcome"] == "started" for r in rows):
                raise RuntimeError("生产入口没有提交本阶段累计尝试，禁止标记验收通过")
        row["state"] = "completed"
        write_json(self.reports / "steps.json", self.invocations)

    def __exit__(self, kind, error, trace):
        stop_owned(self.owned)
        for row in list(self.ledger.read()["attempts"]):
            if row["outcome"] == "started":
                self.ledger.finish(row["stage"], row["number"], "interrupted")
        summary = summarize_run(self.invocations, self.ledger.read(), self.versions, perf_counter() - self.started)
        summary.update(error=None if error is None else str(error),
                       state=summary["state"] if kind is None else "INCOMPLETE",
                       sampled_peak_bytes=self.peak_bytes, sampled_peak_files=self.peak_files,
                       capacity_sampling="阶段边界目录元数据，未测量瞬时峰值",
                       ram_peak="未测量", vram_peak="未测量")
        try:
            self.versions["engine_end"] = verify_engine(self.settings["engine_root"], self.settings["engine_commit"])
            self.versions["tool_end"] = git(TOOL_ROOT, "rev-parse", "HEAD")
            write_json(self.reports / "summary.json", summary)
        finally:
            self.cleanup()
        return False

    def cleanup(self):
        stop_owned(self.owned)
        confined(self.work, Path(self.settings["work_root"]))
        rows = inventory(self.work)
        write_json(self.reports / "cleanup.json", {"state": "pending", "files": rows})
        shutil.rmtree(self.work)
        if self.work.exists():
            raise RuntimeError("本次工作目录清理不完整")
        write_json(self.reports / "cleanup.json", {"state": "completed", "deleted_files": len(rows),
                   "released_bytes": sum(r["bytes"] for r in rows), "work_removed": True})
        if sum(r["bytes"] for r in inventory(self.reports)) > self.approval["max_report_bytes"]:
            raise RuntimeError("最终必要报告超过批准额度")


def run_integration(config, run_id, approval_path):
    settings = load_acceptance(config)
    approval = check_approval(settings, approval_path)
    # 锁和累计记录按本轮批准文件固定；更换运行编号不重置训练预算。
    lock = Path(approval_path).resolve().with_suffix(".execution.lock")
    with FileLock(str(lock), timeout=0):
        reports = Path(settings["report_root"])
        if reports.exists() and any(reports.glob("*/attempts.json")):
            raise RuntimeError("该报告根目录已有本轮尝试记录，须核对已有进度后继续")
        with IntegrationRun(settings, run_id, approval) as task:
            for step in build_steps(settings):
                task.stage(step)
    return read_json(task.reports / "summary.json")
