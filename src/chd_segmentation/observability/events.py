from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import threading
from time import perf_counter
import traceback

from filelock import FileLock

from ..artifacts.store import require_output


def event(directory, kind, console=True, **fields):
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "event": kind, **fields}
    line = json.dumps(record,
                      ensure_ascii=False, allow_nan=False) + "\n"
    if directory is not None:
        path = Path(directory) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 所有 rank 和供数进程共用文件锁，轮转与追加构成一次完整操作。
        with FileLock(str(path.with_suffix(".lock")), timeout=30):
            if path.exists() and path.stat().st_size + len(line.encode("utf-8")) > 16 * 1024 ** 2:
                for index in range(3, 0, -1):
                    old = path.with_suffix(f".jsonl.{index}")
                    if old.exists():
                        if index == 3:
                            old.unlink()
                        else:
                            old.replace(path.with_suffix(f".jsonl.{index + 1}"))
                path.replace(path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
    if console:
        print(line, end="", flush=True)


def log_directory(config):
    return require_output(config["run_dir"], config["work_root"],
                          [config["manifest"], config["pretrained"], Path(__file__).parents[3]])


class Stage:
    def __init__(self, directory, name, heartbeat_seconds=15, enabled=True, console=True, **fields):
        self.directory, self.name, self.interval = directory, name, heartbeat_seconds
        self.enabled, self.fields = enabled, fields
        self.console = console
        self.stop = threading.Event()
        self.guard = threading.Lock()

    def emit(self, kind, console=None, **fields):
        if self.enabled:
            with self.guard:
                current = dict(self.fields)
            event(self.directory, kind, stage=self.name, elapsed_seconds=perf_counter() - self.started,
                  console=self.console if console is None else console, **{**current, **fields})

    def update(self, **fields):
        with self.guard:
            self.fields.update(fields)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit("heartbeat", console=True, message="进程仍在等待当前阶段结束；心跳不代表病例或训练步已完成")

    def __enter__(self):
        self.started = perf_counter()
        if self.enabled:
            self.emit("stage_started")
            if self.interval > 0:
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="progress")
                self.future = self.executor.submit(self._heartbeat)
        return self

    def __exit__(self, exception_type, exception, trace):
        if self.enabled:
            if self.interval > 0:
                self.stop.set()
                self.executor.shutdown(wait=True)
                # 日志线程的写入错误也必须传播给调用方。
                self.future.result()
            fields = {} if exception is None else {
                "error_type": exception_type.__name__, "error": str(exception),
                "traceback": "".join(traceback.format_exception(exception_type, exception, trace))}
            self.emit("stage_completed" if exception is None else "stage_failed", **fields)
        return False


def stage(config, name, **fields):
    settings = config.get("logging", {})
    return Stage(log_directory(config), name, settings.get("heartbeat_seconds", 15), **fields)


def logged(name, config_argument=0, config_key=None):
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            config = args[config_argument] if len(args) > config_argument else kwargs["config"]
            if config_key is not None:
                config = config[config_key]
            fields = {"rank": args[0]} if config_key is not None else {}
            with stage(config, name, dataset=config["dataset"], fold=config["fold"], **fields):
                return function(*args, **kwargs)
        return invoke
    return decorate


def logged_directory(name):
    def decorate(function):
        @wraps(function)
        def invoke(directory, *args, **kwargs):
            with Stage(directory, name):
                return function(directory, *args, **kwargs)
        return invoke
    return decorate
