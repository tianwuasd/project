"""Read-only progress accompanying detailed logs; no CUDA or torch imports."""

import time

from .inspection import MODELS, load, model_runs, run_directory, select_step


def duration(seconds):
    if seconds is None:
        return "暂无法估算"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def show_progress(job, step="auto", model="all"):
    output = select_step(job, "auto" if step == "launcher" else step)
    if output is None:
        return
    server = load(output / "status.json")
    current = next(
        (r for r in load(job / "job_state.json").get("steps", []) if r["output"] == str(output)), {}
    )
    status = current.get("status", server.get("status", "unknown"))
    print(f"\n[进度] {current.get('name', output.name)} | {status}", flush=True)
    if server.get("task") == "train" or current.get("name") in {"train", "smoke", "preflight"}:
        run = run_directory(output)
        queue = load(run / "queue.json")
        waiting = server.get("status") == "waiting_gpu" or (
            queue.get("status") == "waiting_gpu"
            and queue.get("sampled_at", 0) >= server.get("phase_started_at", 0)
        )
        if waiting:
            print("  正在等待 GPU，资源等待时间未知；剩余暂无法估算。")
            status = "waiting_gpu"
        for folder in model_runs(run_directory(output)):
            config = load(folder / "effective-config.json").get("stages", {})
            for name in MODELS if model in {"all", "server"} else (model,):
                history = load(folder / f"{name}.history.json", [])
                if not history and name not in config:
                    continue
                completed = history[-1].get("epoch", 0) if history else 0
                total = config.get(name, {}).get("epochs")
                timings = load(folder / f"{name}.timing.json", [])
                samples = [
                    sum(s.get("seconds", 0) for s in t.get("splits", {}).values()) for t in timings[-3:]
                ]
                eta = sum(samples) / len(samples) * max(0, total - completed) if samples and total else None
                elapsed = sum(sum(s.get("seconds", 0) for s in t.get("splits", {}).values()) for t in timings)
                if status in {"failed", "interrupted", "pending", "waiting_gpu"}:
                    eta = None
                print(
                    f"  {folder.name}/{name}：{completed}/{total or '?'} epoch；"
                    f"已记录计算 {duration(elapsed)}；剩余计算约 {duration(eta)}（不含排队）"
                )
        return
    progress = [
        load(output / "progress.json"),
        load(output / "predict/progress.json"),
        load(output / "test/predictions/progress.json"),
    ]
    progress = [p for p in progress if p.get("sampled_at", 0) >= server.get("phase_started_at", 0)]
    if progress:
        latest = max(progress, key=lambda p: p.get("sampled_at", 0))
        completed, total = latest.get("completed", 0), latest.get("total", 0)
        eta = latest.get("eta_seconds")
        if latest.get("status") == "waiting_gpu" or status in {"failed", "interrupted"}:
            eta = None
        elapsed = latest.get("elapsed_seconds", 0)
        if status == "running" and latest.get("status") in {"running", "waiting_gpu"}:
            elapsed += max(0, time.time() - latest.get("sampled_at", time.time()))
        pct = 100 * completed / total if total else 0
        print(
            f"  {latest.get('phase', '')}：{completed}/{total}（{pct:.1f}%）；"
            f"已用 {duration(elapsed)}；剩余约 {duration(eta)}"
        )
        if latest.get("status") == "waiting_gpu":
            print("  正在等待 GPU，可用时间未知。")
    else:
        elapsed = time.time() - server["phase_started_at"] if server.get("phase_started_at") else None
        print(f"  已用 {duration(elapsed)}；剩余暂无法估算（尚无本阶段完成样本或旧版无计时记录）。")
