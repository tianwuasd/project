"""Automatic, bounded CPU loader calibration within smoke/preflight/train startup.

No public benchmark mode. Uses training cases only and the source configuration's
real input sizes, then freezes settings for all subsequent stages and folds.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from ..recovery import read_json, write_json
from .dataset import StageDataset
from .loading import close_loader, make_loader, settings, start_epoch


def cache_directory(directory, config):
    return Path(settings(config)["cache_dir"] or Path(directory) / ".training-inputs").expanduser().resolve()


def terminate(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    process.wait()


def trial(request, folder, seconds):
    """One private subprocess; timeout kills its loader children as well."""
    folder.mkdir(parents=True, exist_ok=True)
    request_file, result_file = folder / "request.json", folder / "result.json"
    result_file.unlink(missing_ok=True)
    write_json(request_file, request)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # CPU calibration never claims a candidate GPU.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    process, cancelled = None, False
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        # Do not raise inside Popen before the process handle has been assigned.
        nonlocal cancelled
        cancelled = True

    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        with (folder / "trial.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", __name__, str(request_file), str(result_file)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
                **(
                    {"pass_fds": (int(env["CHD_RUNTIME_LOCK_FD"]),)}
                    if os.name == "posix" and env.get("CHD_RUNTIME_LOCK_FD")
                    else {}
                ),
            )
            deadline = time.monotonic() + seconds
            while True:
                if cancelled:
                    raise KeyboardInterrupt("CPU 调优已取消")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"status": "timeout"}
                try:
                    code = process.wait(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        try:
            if process is not None and (cancelled or process.poll() is None):
                terminate(process)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if cancelled:
            raise KeyboardInterrupt("CPU 调优已取消")
    if code or not result_file.is_file():
        return {"status": "failed", "exit_code": code}
    return read_json(result_file)


def measure(request):
    config, stage = request["config"], request["stage"]
    torch.set_num_threads(config["threads"])
    dataset = StageDataset(request["directory"], request["rows"], stage, config, request["cache_root"])
    # Interleave cases instead of timing one patient's hot slices only. Each
    # candidate sees exactly the same bounded sequence, including edge slices.
    indices = []
    by_case = [
        [i for i, (case, _) in enumerate(dataset.index) if case == c] for c in range(len(dataset.rows))
    ]
    count = 12 * config["stages"][stage]["batch"]
    for step in range(count):
        case = step % len(by_case)
        choices = by_case[case]
        per_case = count // len(by_case) + int(case < count % len(by_case))
        position = round((step // len(by_case)) * (len(choices) - 1) / max(1, per_case - 1))
        indices.append(choices[position])
    loader = make_loader(Subset(dataset, indices), config, stage, "val", config["seed"])
    try:
        start = time.perf_counter()
        iterator = start_epoch(loader, True)
        next(iterator)
        startup = time.perf_counter() - start
        for _ in iterator:
            pass
        durations = []
        # Two warm rounds include the normal iterator reset and amortize jitter.
        for _ in range(2):
            start = time.perf_counter()
            samples = sum(len(x) for x, _ in start_epoch(loader, True))
            durations.append((time.perf_counter() - start) / samples)
        return dict(
            status="passed",
            seconds_per_sample=float(np.median(durations)),
            round_seconds_per_sample=durations,
            startup_seconds=startup,
            samples=samples,
        )
    finally:
        close_loader(loader)


def choose(trials, baseline):
    passed = [r for r in trials if r["status"] == "passed"]
    if not passed:
        return 0, "所有候选未通过，回退单进程加载；详见 trial.log"
    best = min(r["seconds_per_sample"] for r in passed)
    # Small timing differences do not justify extra CPU processes.
    selected = min((r for r in passed if r["seconds_per_sample"] <= best * 1.10), key=lambda r: r["workers"])
    return selected["workers"], "两轮热加载耗时；与最快相差不超过 10% 时选更少进程"


def resolve(config, directory, rows, output, concurrency=1, resume=False):
    from ..panel_v2.resources import direct_config
    return direct_config(config, output, concurrency)
