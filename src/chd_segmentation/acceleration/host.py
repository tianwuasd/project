import ctypes
from ctypes import CDLL, POINTER, byref, c_char_p, c_int, c_void_p
from ctypes.util import find_library
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import psutil


def windows_job_capacity():
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("read_operations", "write_operations", "other_operations", "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    class MemoryUsage(ctypes.Structure):
        _fields_ = [("current", ctypes.c_uint64), ("peak", ctypes.c_uint64)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, c_void_p,
                                                wintypes.DWORD, POINTER(wintypes.DWORD)]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    included = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, byref(included)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not included.value:
        return None
    limits, usage = ExtendedLimits(), MemoryUsage()
    if not kernel.QueryInformationJobObject(None, 9, byref(limits), ctypes.sizeof(limits), None):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.QueryInformationJobObject(None, 28, byref(usage), ctypes.sizeof(usage), None):
        raise ctypes.WinError(ctypes.get_last_error())
    job_limit = int(limits.job_memory) if limits.basic.flags & 0x200 else None
    process_limit = int(limits.process_memory) if limits.basic.flags & 0x100 else None
    current_process = psutil.Process().memory_info().private
    return {"limit_bytes": job_limit, "current_bytes": int(usage.current),
            "peak_bytes": int(usage.peak), "process_limit_bytes": process_limit,
            "process_current_bytes": current_process, "scope": "current_job"}


@lru_cache(maxsize=1)
def native_systemd():
    systemd = CDLL(find_library("systemd") or "libsystemd.so.0")
    systemd.sd_pid_get_cgroup.argtypes = [c_int, POINTER(c_void_p)]
    systemd.sd_pid_get_cgroup.restype = c_int
    return systemd


def current_cgroup():
    if sys.platform != "linux":
        return None
    pointer = c_void_p()
    status = native_systemd().sd_pid_get_cgroup(0, byref(pointer))
    if status < 0:
        raise OSError(-status, "无法确定当前进程的 cgroup")
    libc = CDLL(None)
    libc.free.argtypes = [c_void_p]
    try:
        return c_char_p(pointer.value).value.decode()
    finally:
        libc.free(pointer)


@lru_cache(maxsize=8)
def cgroup_directories(membership):
    if membership is None:
        return ()
    # systemd 和 util-linux 分别解析进程成员关系和 mountinfo。
    member = PurePosixPath(membership)
    result = subprocess.run(["findmnt", "--json", "--types", "cgroup2", "--output", "TARGET,FSROOT"],
                            check=True, text=True, capture_output=True)
    mounts = json.loads(result.stdout)["filesystems"]
    pending = list(mounts)
    while pending:
        mount = pending.pop()
        pending.extend(mount.get("children", []))
        point, root = Path(mount["target"]), PurePosixPath(mount["fsroot"])
        candidates = [point.joinpath(*member.parts[1:])]
        if member.is_relative_to(root):
            candidates.insert(0, point.joinpath(*member.relative_to(root).parts))
        for current in dict.fromkeys(candidates):
            witness = current / "cgroup.procs"
            if not witness.is_file():
                continue
            members = np.atleast_1d(np.loadtxt(witness, dtype=np.int64, ndmin=1))
            if os.getpid() not in members:
                continue
            directories = []
            while True:
                directories.append(current)
                if current == point:
                    return tuple(directories)
                current = current.parent
    raise RuntimeError("未找到包含当前进程的 cgroup v2；禁止按主机容量扩大供数")


def host_capacity():
    process = psutil.Process()
    cores = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else psutil.cpu_count()
    memory = psutil.virtual_memory()
    total, available, quota = memory.total, memory.available, float(cores)
    groups = []
    for path in cgroup_directories(current_cgroup()):
        row = {"path": str(path), "limit_bytes": None, "current_bytes": None, "cpu_quota": None}
        maximum, current = path / "memory.max", path / "memory.current"
        if maximum.is_file():
            value = maximum.read_text().strip()
            row["current_bytes"] = int(current.read_text())
            if value != "max":
                row["limit_bytes"] = int(value)
                total = min(total, row["limit_bytes"])
                available = min(available, max(0, row["limit_bytes"] - row["current_bytes"]))
        cpu = path / "cpu.max"
        if cpu.is_file():
            values = cpu.read_text().split()
            if values[0] != "max":
                row["cpu_quota"] = int(values[0]) / int(values[1])
                quota = min(quota, row["cpu_quota"])
        groups.append(row)
    job = windows_job_capacity()
    if job is not None:
        for limit, used in ((job["limit_bytes"], job["current_bytes"]),
                            (job["process_limit_bytes"], job["process_current_bytes"])):
            if limit is not None:
                total = min(total, limit)
                available = min(available, max(0, limit - used))
    return {"cores": max(1, int(quota)), "cpu_quota": quota, "affinity_cores": cores,
            "host_total_bytes": memory.total, "host_available_bytes": memory.available,
            "total_bytes": total, "available_bytes": available, "cgroups": groups, "windows_job": job}
