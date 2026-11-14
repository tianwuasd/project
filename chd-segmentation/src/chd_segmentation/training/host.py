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
    return {"cores": max(1, int(quota)), "cpu_quota": quota, "affinity_cores": cores,
            "host_total_bytes": memory.total, "host_available_bytes": memory.available,
            "total_bytes": total, "available_bytes": available, "cgroups": groups}
