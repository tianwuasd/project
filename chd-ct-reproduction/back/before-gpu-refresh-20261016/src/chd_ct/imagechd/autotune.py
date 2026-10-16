"""Automatic, bounded CPU loader calibration within smoke/preflight/train startup.

No public benchmark mode. Uses training cases only and the source configuration's
real input sizes, then freezes settings for all subsequent stages and folds.
"""

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from ..recovery import file_hash, read_json, write_json
from .cpu_cache import FORMAT, cache_lock
from .cpu_resources import resources, worker_limit
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
    """Called once before smoke shrinks the model, or before a CV coordinator."""
    if not settings(config)["auto_tune"]:
        return config
    original = copy.deepcopy(config)
    directory, output = Path(directory), Path(output)
    requested = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    data_hash = file_hash(directory / "dataset.json")
    saved_path = output / "cpu-tuning.json"
    if saved_path.is_file():
        saved = read_json(saved_path)
        if saved["requested_sha256"] != requested or saved["prepared_sha256"] != data_hash:
            raise ValueError("CPU 调优的配置或数据来源改变，请另建任务")
        config = copy.deepcopy(config)
        config["data_loading"] = saved["selected"]
        return config  # Never retune a run's frozen settings during resume.
    if resume and (output / "training-contract.json").is_file():
        raise ValueError("续训缺少冻结 CPU 调优记录，不能重新选参")
    if not rows:
        raise ValueError("CPU 调优需要训练病例")
    hardware = resources(concurrency)
    config = copy.deepcopy(config)
    config["data_loading"] = settings(config)
    options = config["data_loading"]
    options.update(auto_tune=False, stage_workers={})
    root = cache_directory(directory, config)
    options["cache_dir"] = str(root)
    groups = {
        max(
            (s for s in config["stages"] if not s.startswith("blood")),
            key=lambda s: config["stages"][s]["size"],
        ): [s for s in config["stages"] if not s.startswith("blood")],
        "blood2d": ["blood2d"],
        "blood_lstm": ["blood_lstm"],
    }
    # Match later smoke/preflight/train invocations on this allocation, while
    # checking current memory headroom separately instead of hashing fluctuating free RAM.
    identity = dict(
        requested=requested,
        prepared=data_hash,
        format=FORMAT,
        cpu=hardware["cpus"],
        concurrency=concurrency,
        host=hardware["host"],
        torch=str(torch.__version__),
        cases=[r["case_id"] for r in rows[:2]],
        sources=[file_hash(directory / r["cache"]) for r in rows[:2]],
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    shared = root / "tuning" / key
    shared.mkdir(parents=True, exist_ok=True)
    with cache_lock(shared):
        report_path = shared / "report.json"
        saved = read_json(report_path) if report_path.is_file() else None
        if saved and all(
            n <= worker_limit(config, hardware, s) for s, n in saved["selected"]["stage_workers"].items()
        ):
            saved["reused"] = True
            write_json(saved_path, saved)
            config["data_loading"] = saved["selected"]
            print(
                "[cpu-tuning] 复用小样本推荐：" + json.dumps(saved["selected"]["stage_workers"]), flush=True
            )
            return config
        report = dict(
            requested_sha256=requested,
            prepared_sha256=data_hash,
            hardware=hardware,
            scope="CPU data loading only; GPU compute/transfer checked by preflight",
            training_cases=[r["case_id"] for r in rows[:2]],
            test_cases_used=0,
            groups={},
            reused=False,
        )
        remaining = options["tune_seconds"]
        for stage, members in groups.items():
            limit = min(worker_limit(config, hardware, s) for s in members)
            baseline = min(4, options["workers"], limit)
            for s in members:
                options["stage_workers"][s] = baseline
            print(
                f"[cpu-tuning] {stage}: 初始 workers={baseline}, prefetch={options['prefetch_factor']}, 上限={limit}",
                flush=True,
            )
            # Cold cache construction is reported independently; it is not charged
            # to one candidate or mistaken for steady-state loader throughput.
            started = time.perf_counter()
            StageDataset(directory, rows[:2], stage, config, root)
            cold_seconds = time.perf_counter() - started
            trials = []
            candidates = list(dict.fromkeys([baseline, 0, min(limit, max(1, baseline * 2))]))
            for workers in candidates:
                if remaining < 1:
                    break
                candidate = copy.deepcopy(config)
                candidate["device"] = "cpu"
                candidate["data_loading"]["stage_workers"][stage] = workers
                request = dict(
                    config=candidate,
                    directory=str(directory),
                    rows=rows[:2],
                    stage=stage,
                    cache_root=str(root),
                )
                started = time.perf_counter()
                result = trial(request, shared / f"{stage}-{workers}", min(45, remaining))
                remaining -= time.perf_counter() - started
                trials.append(dict(workers=workers, **result))
                print(f"[cpu-tuning] {stage} workers={workers}: {result}", flush=True)
            selected, reason = choose(trials, baseline)
            for s in members:
                options["stage_workers"][s] = selected
            report["groups"][stage] = dict(
                input_size=config["stages"][stage]["size"],
                initial_workers=baseline,
                cold_cache_seconds=cold_seconds,
                trials=trials,
                selected_workers=selected,
                reason=reason,
            )
        report["selected"] = options
        write_json(report_path, report)
        write_json(saved_path, report)
        print("[cpu-tuning] 后续训练配置：" + json.dumps(options["stage_workers"]), flush=True)
    return config


if __name__ == "__main__":
    write_json(Path(sys.argv[2]), measure(read_json(sys.argv[1])))
