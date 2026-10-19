"""Bounded, observable CPU budgets for non-training tasks only."""

import argparse
import importlib
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from .imagechd.cpu_resources import GIB, resources
from .recovery import output_lock, read_json, write_json

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def cpu_sample():
    """Sample allocated cores, not the average of unrelated host cores."""
    try:
        allowed = os.sched_getaffinity(0)
        rows = {}
        for line in Path("/proc/stat").read_text().splitlines():
            fields = line.split()
            if fields[0].startswith("cpu") and fields[0][3:].isdigit():
                core = int(fields[0][3:])
                if core in allowed:
                    values = list(map(int, fields[1:9]))  # guest time is already included
                    rows[core] = (sum(values), values[3])
        return rows
    except (OSError, AttributeError, ValueError):
        return {}


def inspect(delay=0.2):
    hardware = resources(1)
    before = cpu_sample()
    if before:
        time.sleep(delay)
        after = cpu_sample()
        total = sum(after[k][0] - v[0] for k, v in before.items() if k in after)
        idle = sum(after[k][1] - v[1] for k, v in before.items() if k in after)
        hardware["idle_fraction"] = max(0, min(1, idle / total)) if total else None
    else:
        hardware["idle_fraction"] = None
    return hardware


def choose(hardware, cases=1, case_bytes=GIB, concurrency=1, parallel=False):
    cpus = max(1, hardware["cpus"])
    idle = hardware.get("idle_fraction")
    budget = max(1, math.floor(cpus * (0.75 if idle is None else min(0.75, idle))))
    # The same budget is divided by all concurrently permitted GPU workers.
    per_task = max(1, budget // max(1, concurrency))
    memory_slots = int(hardware["memory_available"] * 0.5 // max(GIB // 2, case_bytes))
    processes = max(1, min(8, cases, per_task, memory_slots)) if parallel else 1
    threads = 1 if processes > 1 else min(4, per_task)
    return dict(
        processes=processes,
        threads=threads,
        cpu_budget=budget,
        concurrency=max(1, concurrency),
        estimated_case_bytes=int(case_bytes),
        memory_slots=memory_slots,
        strategy="bounded_heuristic_v1",
        hardware=hardware,
    )


def thread_env(env, threads):
    return dict(env, **{name: str(threads) for name in THREAD_VARS}, CHD_CPU_THREADS=str(threads))


def configure_env(task, output, env, concurrency=1):
    """Called inside an existing step. Training must retain its original environment."""
    if task == "train":
        return env
    selected = choose(inspect(), concurrency=concurrency)
    write_json(Path(output) / "cpu-allocation.json", dict(task=task, **selected))
    print(
        f"CPU 自动分配 {task}：预算 {selected['cpu_budget']}，每任务线程 {selected['threads']}，"
        f"并发预算 {concurrency}（启发式；训练调优不变）",
        flush=True,
    )
    return dict(thread_env(env, selected["threads"]), CHD_CPU_AUTO="1", CHD_CPU_REPORT_DIR=str(output))


def allocation(task, output, cases, case_bytes=GIB, parallel=True, concurrency=1):
    enabled = os.environ.get("CHD_CPU_AUTO") == "1"
    selected = (
        choose(inspect(), cases, case_bytes, concurrency, parallel)
        if enabled
        else dict(processes=1, threads=2, strategy="local_serial", concurrency=concurrency)
    )
    if enabled:
        records = record_directory(output)
        write_json(records / (task + ".json"), dict(task=task, **selected))
    print(f"CPU {task}：{selected['processes']} 个病例进程 × {selected['threads']} 线程", flush=True)
    return selected


def record_directory(output):
    directory = Path(os.environ.get("CHD_CPU_REPORT_DIR", output)) / ".cpu-runtime"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def volume_bytes(rows, factor=48, native=False):
    shapes = [row.get("cache_shape", [256] * 3) for row in rows]
    if native:
        for row in rows:
            shapes.extend(row[key] for key in ("native_shape", "canonical_shape") if row.get(key))
            if row.get("resampling", {}).get("resampled_shape"):
                shapes.append(row["resampling"]["resampled_shape"])
    return max((math.prod(shape) * factor for shape in shapes), default=GIB)


def invoke(module, function, kwargs):
    return getattr(importlib.import_module(module), function)(**kwargs)


def iter_cases(module, function, items, output, selected, lock_fds=()):
    """Bounded subprocesses, JSON metadata only; arrays stay in case-specific files.

    Results are yielded in input order, keeping manifests and split assignment stable.
    No unbounded prefetch, no multiprocessing array copies, no nested worker pools.
    """
    items = list(items)
    started = time.monotonic()
    timing = record_directory(output) if os.environ.get("CHD_CPU_AUTO") == "1" else None
    from .progress import Progress

    progress = Progress(timing.parent, function, len(items)) if timing else None
    if progress:
        progress.update(0)
    completed = 0
    if selected["processes"] == 1:
        if timing:
            import torch

            torch.set_num_threads(selected["threads"])
        try:
            for item in items:
                result = invoke(module, function, item)
                completed += 1
                if progress:
                    progress.update(completed, "complete" if completed == len(items) else "running")
                yield result
        finally:
            if timing is not None:
                write_json(
                    timing / "timing.json",
                    dict(seconds=time.monotonic() - started, cases=completed, **selected),
                )
        return
    folder = Path(output) / ".cpu-work" / uuid4().hex
    folder.mkdir(parents=True)
    active, ready = {}, {}
    next_start = next_result = 0
    env = thread_env(os.environ, selected["threads"])
    env.update(CUDA_VISIBLE_DEVICES="", CHD_CPU_AUTO="0")
    handlers, stopped = {}, [False]
    records = {}

    def publish():
        write_json(folder / "queue.json", dict(stages=records))

    def interrupted(sig, frame):
        stopped[0] = True

    with output_lock(folder) as own_fd:
        fds = set(lock_fds) | {own_fd}
        if env.get("CHD_RUNTIME_LOCK_FD"):
            fds.add(int(env["CHD_RUNTIME_LOCK_FD"]))
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is not None:
                handlers[sig] = signal.signal(sig, interrupted)
        try:
            while next_result < len(items):
                if stopped[0]:
                    raise KeyboardInterrupt("CPU 病例队列中断")
                while next_start < len(items) and len(active) + len(ready) < selected["processes"]:
                    number = next_start
                    spec = folder / f"{number}.json"
                    write_json(spec, dict(module=module, function=function, kwargs=items[number]))
                    log = (folder / f"{number}.log").open("a", encoding="utf-8")
                    try:
                        proc = subprocess.Popen(
                            [sys.executable, "-m", "chd_ct.cpu_runtime", str(spec)],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            **({"pass_fds": tuple(fds)} if os.name == "posix" else {}),
                        )
                    except BaseException:
                        log.close()
                        raise
                    active[number] = (proc, log, spec)
                    from .server.jobs import identity

                    records[str(number)] = dict(pid=proc.pid, identity=identity(proc.pid), status="running")
                    publish()
                    next_start += 1
                for number, (proc, log, spec) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[number]
                    records[str(number)]["status"] = "complete" if code == 0 else "failed"
                    publish()
                    result_path = spec.with_suffix(".result.json")
                    if code:
                        error = read_json(result_path) if result_path.exists() else {}
                        if error.get("type") == "CacheReuseError":
                            from .imagechd.cache_reuse import CacheReuseError

                            raise CacheReuseError(error["error"])
                        raise ValueError(
                            error.get("error", f"CPU worker 退出 {code}；查看 {spec.with_suffix('.log')}")
                        )
                    ready[number] = read_json(result_path)
                while next_result in ready:
                    result = ready.pop(next_result)
                    next_result += 1
                    if progress:
                        progress.update(next_result, "complete" if next_result == len(items) else "running")
                    yield result
                if next_result < len(items):
                    time.sleep(0.05)
        finally:
            for proc, log, _ in active.values():
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if timing is not None:
                write_json(
                    timing / "timing.json",
                    dict(seconds=time.monotonic() - started, cases=next_result, **selected),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    args = parser.parse_args()
    spec = Path(args.spec)
    task = read_json(spec)
    import torch

    torch.set_num_threads(int(os.environ.get("CHD_CPU_THREADS", "1")))
    try:
        result = invoke(task["module"], task["function"], task["kwargs"])
    except Exception as error:
        write_json(spec.with_suffix(".result.json"), dict(error=str(error), type=type(error).__name__))
        raise
    write_json(spec.with_suffix(".result.json"), result)


if __name__ == "__main__":
    main()
