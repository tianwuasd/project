"""Read-only, session-local SHA256 verification of completed workflow steps."""

import hashlib
import math
import os
import re
import signal
import stat
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

MIB = 1024**2
BLOCK = 4 * MIB
CHANGED = "已完成步骤的输入或输出改变（包含新增/缺失文件），不能直接续跑"


@contextmanager
def finish_before_interrupt():
    """A repeated Ctrl+C must not interrupt join and leave live file readers."""
    interrupted = False
    main = threading.current_thread() is threading.main_thread()

    def later(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGINT, later) if main else None
    try:
        yield
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
        if interrupted:
            raise KeyboardInterrupt()


def fingerprint(path):
    try:
        info = path.stat()
    except OSError as error:
        raise ValueError("校验无法读取文件，不能直接续跑：" + str(path)) from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(CHANGED + "：" + str(path))
    return signature(info)


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def tree_files(root, tick=lambda: None):
    """Keep legacy tree_hashes membership, but prune hidden directories early."""
    root = Path(root)
    if root.is_file():
        return {str(root.resolve())}
    if not root.is_dir():
        raise ValueError("恢复所需路径不存在：" + str(root))
    found = set()

    def fail(error):
        raise error

    for folder, dirs, files in os.walk(root, onerror=fail):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        tick()
        for name in files:
            path = Path(folder) / name
            if (
                not name.startswith(".")
                and path.suffix not in {".log", ".tmp"}
                and ".interrupted-" not in str(path)
                and path.is_file()
            ):
                found.add(str(path.resolve()))
            tick()
    return found


def storage_kind(path, mounts=None, sys_root=Path("/sys/dev/block")):
    """Use the actual Linux mount/device, never infer NAS from a directory name."""
    try:
        lines = mounts if mounts is not None else Path("/proc/self/mountinfo").read_text().splitlines()
        selected = None
        for line in lines:
            left, right = line.split(" - ", 1)
            fields = left.split()
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if path.is_relative_to(mount) and (selected is None or len(str(mount)) > selected[0]):
                selected = (len(str(mount)), fields[2], right.split()[0])
        if selected:
            _, device, fs = selected
            if fs in {"nfs", "nfs4", "cifs", "smb3", "ceph", "lustre", "glusterfs"} or fs.startswith("fuse"):
                return "network"
            if fs in {"tmpfs", "ramfs"}:
                return "memory"
            device = (sys_root / device).resolve()
            for folder in (device, device.parent):
                rotational = folder / "queue/rotational"
                if rotational.is_file():
                    return "hdd" if rotational.read_text().strip() == "1" else "ssd"
    except (OSError, ValueError, IndexError):
        pass
    return "unknown"


def choose_threads(hardware, kinds, count, total):
    idle = hardware.get("idle_fraction")
    cpus = max(1, hardware["cpus"])
    budget = max(1, math.floor(cpus * (0.75 if idle is None else min(0.75, max(0, idle)))))
    caps = {"network": 4, "ssd": 8, "memory": 8, "hdd": 1, "unknown": 2}
    storage_cap = min((caps[k] for k in kinds), default=2)
    memory_cap = max(1, int(hardware["memory_available"] * 0.1 // (2 * BLOCK)))
    workers = max(1, min(budget, storage_cap, memory_cap, count))
    if total < 32 * MIB:
        workers = 1
    return dict(workers=workers, cpus=cpus, budget=budget, storage_cap=storage_cap)


class ReadProgress:
    """A heartbeat also remains visible during a slow directory/stat/read syscall."""

    def __init__(self, enabled=True, interval=1):
        self.enabled = enabled
        self.interval = interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.phase = "扫描文件清单"
        self.detail = ""
        self.discovered = self.total = self.done = self.count = self.finished = 0
        self.started = None

    def __enter__(self):
        self.thread = None
        if self.enabled:
            self.render()
            self.thread = threading.Thread(target=self.heartbeat, name="chd-verify-progress", daemon=True)
            self.thread.start()
        return self

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            self.render()

    def stage(self, phase, detail=""):
        with self.lock:
            self.phase, self.detail = phase, detail

    def found(self):
        with self.lock:
            self.discovered += 1

    def reading(self, total, count, selected, kinds, references):
        with self.lock:
            self.phase = "读取并校验"
            self.total, self.count, self.started = total, count, time.monotonic()
            self.detail = ""
            if self.enabled:
                labels = {
                    "network": "网络盘",
                    "ssd": "SSD",
                    "memory": "内存盘",
                    "hdd": "机械块设备",
                    "unknown": "未识别",
                }
                print(
                    f"校验自动配置：{selected['workers']} 个读取线程；可用 CPU {selected['cpus']}，"
                    f"空闲预算 {selected['budget']}，存储 {','.join(labels[k] for k in sorted(kinds)) or '未识别'}"
                    f"（上限 {selected['storage_cap']}，启发式选择）。\n"
                    f"共 {count} 个唯一文件，{total / MIB:.1f} MiB；"
                    f"合并 {references - count} 次重复文件引用，只完整读取一遍。",
                    flush=True,
                )
        self.render()

    def advance(self, size):
        with self.lock:
            self.done += size

    def completed_file(self):
        with self.lock:
            self.finished += 1

    def render(self):
        if not self.enabled:
            return
        with self.lock:
            if self.phase == "读取并校验":
                elapsed = max(0.001, time.monotonic() - self.started)
                speed = self.done / elapsed
                ratio = min(1, self.done / self.total) if self.total else 1
                bar = "#" * int(20 * ratio) + "-" * (20 - int(20 * ratio))
                eta = f"约 {(self.total - self.done) / speed:.0f} 秒" if speed else "估算中"
                line = (
                    f"校验读取 [{bar}] {ratio:.1%} | {self.done / MIB:.1f}/{self.total / MIB:.1f} MiB"
                    f" | 文件 {self.finished}/{self.count} | {speed / MIB:.1f} MiB/s"
                    f" | 读取剩余 {eta}"
                )
            else:
                line = f"校验：{self.phase} {self.detail}"
                if self.phase == "扫描文件清单":
                    line += f" | 已扫描目录项 {self.discovered}，剩余时间暂不可估算"
                elif self.phase == "复核文件状态":
                    line += " | 读取已结束，正在复核新增/缺失文件与修改时间"
            print(line, flush=True)

    def __exit__(self, error_type, error, traceback):
        with finish_before_interrupt():
            self.stop.set()
            if self.thread:
                self.thread.join()
        self.stage(
            "校验中断，未提交续跑"
            if error_type is KeyboardInterrupt
            else "校验失败，未提交续跑"
            if error_type
            else "校验通过"
        )
        self.render()


def hash_one(path, expected_stat, progress, cancelled):
    if cancelled.is_set():
        raise InterruptedError("校验已取消")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = signature(os.fstat(stream.fileno()))
        # Windows Python can expose different ctime semantics for stat and fstat.
        # Compare ctime only within the same API; identity, size and mtime still match.
        if opened[:4] != expected_stat[:4]:
            raise ValueError(CHANGED + "：" + str(path))
        remaining = expected_stat[2]
        while not cancelled.is_set():
            block = stream.read(min(BLOCK, remaining + 1))
            if not block:
                break
            if len(block) > remaining:
                raise ValueError(CHANGED + "：" + str(path))
            remaining -= len(block)
            digest.update(block)
            progress.advance(len(block))
        if cancelled.is_set():
            raise InterruptedError("校验已取消")
        if remaining or signature(os.fstat(stream.fileno())) != opened:
            raise ValueError(CHANGED + "：" + str(path))
    if fingerprint(path) != expected_stat:
        raise ValueError(CHANGED + "：" + str(path))
    progress.completed_file()
    return digest.hexdigest()


def read_hashes(files, stamps, workers, progress, expected):
    """Keep at most workers open/pending files, and join all readers on exit."""
    cancelled = threading.Event()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chd-verify-reader")
    pending = {}
    iterator = iter(sorted(files))
    try:
        while True:
            while len(pending) < workers:
                name = next(iterator, None)
                if name is None:
                    break
                pending[pool.submit(hash_one, Path(name), stamps[name], progress, cancelled)] = name
            if not pending:
                break
            ready, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in ready:
                name = pending.pop(future)
                if future.result() != expected[name]:
                    raise ValueError(CHANGED + "：" + name)
    except BaseException:
        progress.stage("正在停止校验", "等待读取线程退出；未提交续跑")
        raise
    finally:
        with finish_before_interrupt():
            cancelled.set()
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


def verify_snapshots(records, show_progress=True):
    """Verify all successful steps together; no persistent/metadata-only hash cache."""
    from ..cpu_runtime import inspect

    with ReadProgress(show_progress) as progress:
        roots, files, expected, aliases = {}, set(), {}, {}
        references = 0
        for label, record in records:
            for group, root_key in (("inputs", "input_roots"), ("outputs", "output_roots")):
                progress.stage("扫描文件清单", f"{label} / {group}")
                recorded = record[group]
                if root_key in record:
                    actual = set()
                    for root in record[root_key]:
                        # Do not resolve the root here: retargeted symlinks must be caught at final scan.
                        if root not in roots:
                            roots[root] = tree_files(root, progress.found)
                        actual.update(roots[root])
                    if actual != set(recorded):
                        raise ValueError(CHANGED + "：" + label)
                for name, digest in recorded.items():
                    canonical = str(Path(name).resolve())
                    aliases[name] = canonical
                    if canonical in expected and expected[canonical] != digest:
                        raise ValueError(CHANGED + "（步骤之间记录冲突）：" + name)
                    expected[canonical] = digest
                    files.add(canonical)
                    references += 1
        stamps = {name: fingerprint(Path(name)) for name in sorted(files)}
        # Classify each unique filesystem once, even when the cache has thousands of files.
        devices = {}
        for name, stamp in stamps.items():
            devices.setdefault(stamp[0], Path(name))
        kinds = {storage_kind(path) for path in devices.values()}
        total = sum(stamp[2] for stamp in stamps.values())
        hardware = inspect() if files else dict(cpus=1, memory_available=0)
        selected = choose_threads(hardware, kinds, len(files), total)
        progress.reading(total, len(files), selected, kinds, references)
        read_hashes(files, stamps, selected["workers"], progress, expected)
        progress.render()
        progress.stage("复核文件状态")
        progress.render()
        for root, members in roots.items():
            if tree_files(root) != members:
                raise ValueError(CHANGED + "：" + root)
        for name, stamp in stamps.items():
            if fingerprint(Path(name)) != stamp:
                raise ValueError(CHANGED + "：" + name)
        for name, canonical in aliases.items():
            if str(Path(name).resolve()) != canonical:
                raise ValueError(CHANGED + "：" + name)
        return dict(files=len(files), bytes=total, references=references, **selected)
