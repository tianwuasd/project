"""Per-process allocator budget and bounded fail-fast GPU memory checks."""

import os
import subprocess
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from ..recovery import write_json

GIB = 2**30
_active_guard = ContextVar("chd_gpu_memory_guard", default=None)


def check_memory():
    guard = _active_guard.get()
    if guard is not None:
        guard.check()


def budget_bytes(free, total, fraction=0.85, reserve_gib=2.0):
    if not 0 < fraction < 1 or reserve_gib < 2 or not 0 < free <= total:
        raise ValueError("GPU 显存预算参数无效")
    budget = min(int(total * fraction), int(free - max(reserve_gib * GIB, free * (1 - fraction))))
    if budget < GIB:
        raise RuntimeError("GPU 可用显存不足以保留安全余量；请等待获准空闲卡")
    return budget


class MemoryGuard:
    def __init__(self, device, path, config):
        self.enabled = str(device).startswith("cuda")
        self.device = torch.device(device)
        self.previous_fraction = None
        self.last_scan = float("-inf")
        self.path, self.record = path, {"device": device, "status": "cpu"}

    def start(self, config):
        if not self.enabled:
            return
        options = config.get("gpu_guard", {})
        # Safe only before this stage creates live model tensors, never per batch.
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        self.record = {
            "status": "starting",
            "device": str(self.device),
            "initial_free_bytes": free,
            "total_bytes": total,
            "scope": "PyTorch allocator; not an exclusive GPU reservation",
        }
        self.limit = budget_bytes(free, total, options.get("fraction", 0.85), options.get("reserve_gib", 2.0))
        self.minimum_free = int(options.get("minimum_free_gib", 0.5) * GIB)
        getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        self.previous_fraction = getter(self.device) if getter else 1.0
        torch.cuda.set_per_process_memory_fraction(self.limit / total, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.record.update(status="running", budget_bytes=self.limit, minimum_free_bytes=self.minimum_free)
        self.check()
        write_json(self.path, self.record)
        print(f"[gpu-memory] budget={self.limit / GIB:.2f} GiB; free={free / GIB:.2f} GiB", flush=True)

    def check(self):
        if not self.enabled:
            return
        if time.monotonic() - self.last_scan >= 5:
            foreign = foreign_processes(self.device)
            self.last_scan = time.monotonic()
            if foreign:
                self.record["other_processes"] = foreign
                raise RuntimeError("GPU 出现非本任务计算进程，保护性停止；不终止其他进程")
        free, _ = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        if free < self.minimum_free or allocated > self.limit:
            raise RuntimeError("GPU 显存保护停止：余量不足或本任务超预算；保留最后完整 epoch，不自动改 batch")

    def finish(self, error=None):
        if not self.enabled:
            return
        self.record.update(status="failed" if error else "complete", error=str(error) if error else None)
        diagnostics = []
        for key, fn in (
            ("peak_allocated_bytes", torch.cuda.max_memory_allocated),
            ("peak_reserved_bytes", torch.cuda.max_memory_reserved),
            ("free_bytes", lambda device: torch.cuda.mem_get_info(device)[0]),
        ):
            try:
                self.record[key] = fn(self.device)
            except Exception as problem:
                diagnostics.append(f"{key}: {problem}")
        try:
            if self.previous_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(self.previous_fraction, self.device)
        except Exception as problem:
            diagnostics.append(f"restore_budget: {problem}")
        self.record["diagnostic_errors"] = diagnostics
        try:
            write_json(self.path, self.record)
        except Exception as problem:
            if error is None:
                raise
            warnings.warn(f"GPU failure snapshot could not be saved: {problem}", RuntimeWarning)


def foreign_processes(device):
    from ..server.gpu_hold import owned_processes

    gpu = str(torch.cuda.get_device_properties(device).uuid)
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    allowed = {os.getpid(), *owned_processes()}
    return [
        int(pid.strip())
        for line in query.stdout.splitlines()
        if "," in line
        for uuid, pid in [line.split(",", 1)]
        if uuid.strip().lower() == gpu.lower() and pid.strip().isdigit() and int(pid.strip()) not in allowed
    ]


@contextmanager
def gpu_budget(device, path, config):
    guard = MemoryGuard(device, path, config)
    token = _active_guard.set(guard)
    try:
        guard.start(config)
        yield guard
    except BaseException as error:
        guard.finish(error)
        raise
    else:
        guard.finish()
    finally:
        _active_guard.reset(token)
