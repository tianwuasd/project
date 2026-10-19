"""Read-only server inspection; usable with the system Python and no ML packages."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS = ("crop64", "crop128", "all64", "all128", "blood2d", "blood_lstm")
MODEL_STEPS = {"smoke", "preflight", "train"}
FOLDS = ("fold_0", "fold_1", "fold_2")


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def latest(runtime):
    directory = load(Path(runtime) / "last-job.json").get("directory")
    return Path(directory).expanduser().resolve() if directory else None


def name(job):
    return load(job / "job_state.json").get("name") or load(job / "plan.json").get("name") or job.name


def job_active(job):
    from .jobs import identity, job_owner
    from .resume import live_commands

    if job is None or not (job / "job_state.json").is_file():
        return False
    owner = job_owner(job)
    if owner.get("identity") and identity(owner.get("pid")) == owner["identity"]:
        return True
    plan = load(job / "plan.json")
    return bool(plan.get("steps") and live_commands(plan))


def status(job):
    saved = load(job / "job_state.json").get("status", "unknown")
    if job_active(job):
        return "running" if saved in {"running", "queued"} else "running（仍有子进程）"
    return "stale（记录中的进程已退出）" if saved in {"running", "queued"} else saved


def history(runtime, results):
    """Discover only workflows for this runtime, including the last custom result root."""
    runtime = Path(runtime).expanduser().resolve()
    pointer = latest(runtime)
    roots = {Path(results).expanduser().resolve()}
    saved_results = load(runtime / ".server-settings.json").get("results")
    if saved_results:
        roots.add(Path(saved_results).expanduser().resolve())
    if pointer:
        roots.add(pointer.parent)
    candidates = {p for root in roots for p in root.glob("*_workflow") if p.is_dir()}
    candidates.update(Path(p).expanduser().resolve() for p in load(runtime / "job-history.json", []))
    if pointer and pointer.is_dir():
        candidates.add(pointer)
    found = []
    for job in candidates:
        recorded = load(job / "plan.json").get("runtime")
        if recorded and Path(recorded).expanduser().resolve() == runtime:
            found.append(job)
    return sorted(found, key=lambda p: p.name, reverse=True)


def show_history(runtime, results):
    records = history(runtime, results)
    print("\n历史任务（新到旧；完成任务可查询，中断任务可续跑）：")
    for i, job in enumerate(records, 1):
        print(f"  {i}. {name(job)} | {status(job)}\n     {job}")
    if not records:
        print("没有任务记录。可用 --results 指定其他历史结果目录。")
    return records


def select_step(job, requested="auto"):
    steps = load(job / "job_state.json").get("steps", [])
    if requested != "auto":
        matching = [s for s in steps if s["name"] == requested or Path(s["output"]).name == requested]
        if not matching:
            raise ValueError("任务没有此步骤：" + requested)
        return Path(matching[0]["output"])
    started = [s for s in steps if s.get("status") != "pending"]
    active = [s for s in started if s.get("status") == "running"]
    selected = active or started or steps
    return Path(selected[-1]["output"]) if selected else None


def run_directory(step):
    phase = load(step / "status.json").get("phase")
    if phase == "preflight" and (step / "preflight").is_dir():
        return step / "preflight"
    # A standalone preflight, unlike the preflight within formal train, writes train/.
    return step / "train"


def model_runs(run):
    """Return legacy run or the fixed, contained fold paths, including future folds."""
    run = Path(run)
    root = run.resolve()
    folds = [run / fold for fold in FOLDS if (run / fold).resolve().is_relative_to(root)]
    if (
        load(run / "report.json").get("task") == "chd_threefold"
        or load(run / "models.json").get("format") == "imagechd7-threefold-v1"
        or any(fold.is_dir() for fold in folds)
    ):
        return folds
    return [run]


def active_model_run(run):
    """Use the global cross-fold queue, or the legacy current fold's fixed path."""
    run = Path(run)
    runs = model_runs(run)
    if runs == [run]:
        return run
    summary = load(run / "report.json")
    if summary.get("scheduling") == "cross_fold" and summary.get("status") in {
        "running",
        "interrupted",
        "failed",
    }:
        return run
    current = summary.get("current_fold")
    return next((fold for fold in runs if fold.name == current), None)


def show_models(job, step="auto"):
    if step == "auto":
        steps = load(job / "job_state.json").get("steps", [])
        choices = [s for s in steps if s["name"] in MODEL_STEPS and s.get("status") != "pending"]
        if not choices:
            print("尚无模型训练记录；模型步骤尚未开始。")
            return
        step = next((s["name"] for s in choices if s.get("status") == "running"), choices[-1]["name"])
    output = select_step(job, step)
    if output is None:
        print("尚无模型训练记录。")
        return
    run = run_directory(output)
    report = load(output / "status.json")
    print(f"\n任务：{name(job)} | {status(job)}\n模型记录：{run}")
    runs = model_runs(run)
    if runs == [run]:
        show_model_table(run, report)
    else:
        summary = load(run / "report.json")
        folds = summary.get("folds", {})
        complete = sum(folds.get(fold, {}).get("status") == "complete" for fold in FOLDS)
        current = summary.get("current_fold")
        current = current if current in FOLDS else "无"
        if summary.get("scheduling") == "cross_fold":
            active = ", ".join(f for f in summary.get("current_folds", []) if f in FOLDS) or "无"
            position = "跨折共享队列 | 当前运行折：" + active
        else:
            position = "当前折：" + current
        print(f"三折进度：已完成 {complete}/3 | {position} | 状态：{summary.get('status', '未记录')}")
        config = load(run / "effective-config.json").get("stages", {})
        for fold in runs:
            item = folds.get(fold.name, {})
            print(
                f"\n{fold.name} | {item.get('status', '未记录')} | "
                f"train={item.get('train_cases', '?')} val={item.get('val_cases', '?')} "
                f"test={item.get('test_cases', '?')}\n模型记录：{fold}"
            )
            show_model_table(fold, report, config, completed=item.get("status") == "complete")
    print("每轮训练和验证完成后更新 epoch/loss；心跳刷新不代表模型完成了一轮。")


def show_model_table(run, report, fallback_config=None, completed=False):
    queue = load(run / "queue.json")
    old_queue = (
        bool(queue) and not completed and queue.get("sampled_at", 0) < report.get("phase_started_at", 0)
    )
    if old_queue:
        print("队列是恢复前的旧记录，尚无本轮调度状态。历史 epoch/loss 仅作参考。")
    config = load(run / "effective-config.json").get("stages", fallback_config or {})
    stages = {} if old_queue else queue.get("stages", {})
    print("模型          记录状态          已完成 epoch/目标    train loss    val loss    学习率    GPU")
    for model in MODELS:
        records = load(run / f"{model}.history.json", [])
        record = records[-1] if isinstance(records, list) and records else {}
        item = stages.get(model, {})
        label = item.get("status", "未记录")
        if (
            model == "blood_lstm"
            and label == "pending"
            and stages.get("blood2d", {}).get("status") != "complete"
        ):
            label += "（等待 blood2d）"
        epoch = f"{record.get('epoch', 0)}/{config.get(model, {}).get('epochs', '?')}"
        loss = record.get("loss", {})
        print(
            f"{model:<13} {label:<14} {epoch:<18} {loss.get('train', '—')}    {loss.get('val', '—')}    {record.get('learning_rate', '—')}    {item.get('gpu_uuid', '—')}"
        )


def log_paths(job, step="auto", model="all"):
    if step == "launcher":
        return [job / "launcher.log"]
    output = select_step(job, step)
    if output is None:
        return [job / "launcher.log"]
    if model == "server":
        return [output / "server.log"]
    task = load(output / "status.json").get("task")
    if task in {"test", "predict"}:
        root = output / ("test/predictions" if task == "test" else "predict")
        return [output / "server.log", *sorted((root / "stage_logs").glob("*.log"))]
    run = run_directory(output)
    runs = model_runs(run)
    if model != "all" and runs == [run]:
        return [run / "stage_logs" / f"{model}.log"]
    models = MODELS if model == "all" else (model,)
    paths = [fold / "stage_logs" / f"{m}.log" for fold in runs for m in models]
    existing = [p for p in paths if p.is_file() and p.resolve().is_relative_to(run.resolve())]
    return existing or [output / "server.log"]


def tail_logs(supplier, lines=60, follow=False, progress=None):
    """Bound initial reads; follow newly created files and tolerate replacement/truncation."""
    positions, missing = {}, set()
    last_progress = float("-inf")
    try:
        while True:
            paths = supplier()
            for path in paths:
                try:
                    with path.open("rb") as stream:
                        stat = os.fstat(stream.fileno())
                        token = (stat.st_dev, stat.st_ino)
                        previous = positions.get(path)
                        if previous is None or previous[0] != token or stat.st_size < previous[1]:
                            stream.seek(max(0, stat.st_size - 262144))
                            data = stream.read(262144).splitlines(keepends=True)[-lines:]
                            text = b"".join(data).decode("utf-8", errors="replace")
                        else:
                            stream.seek(previous[1])
                            text = stream.read(262144).decode("utf-8", errors="replace")
                        positions[path] = (token, stream.tell())
                    if text:
                        print(
                            f"\n==> {path} <==\n{text}", end="" if text.endswith("\n") else "\n", flush=True
                        )
                    missing.discard(path)
                except FileNotFoundError:
                    positions.pop(path, None)
                    if path not in missing:
                        print(f"日志尚未生成：{path}" + ("；继续等待。" if follow else ""))
                        missing.add(path)
            if progress and time.monotonic() - last_progress >= 10:
                progress()
                last_progress = time.monotonic()
            if not follow:
                return
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n退出查看，后台任务继续运行。")


def show_gpu():
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    print(result.stdout or result.stderr)
    if result.returncode:
        raise ValueError("显卡查询失败，请检查 NVIDIA 驱动与 nvidia-smi。")


def watch_gpu():
    """Refresh the read-only GPU view once per second until Ctrl+C."""
    try:
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print("显卡状态：每 1 秒刷新；Ctrl+C 返回，后台训练继续。", flush=True)
            show_gpu()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n退出显卡查看，后台任务继续运行。")


def show_processes():
    if not hasattr(os, "getuid"):
        print("进程查询需要在 Linux 服务器上执行。")
        return
    result = subprocess.run(
        ["ps", "-u", str(os.getuid()), "-o", "pid,ppid,stat,etime,pcpu,pmem,args"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    rows = result.stdout.splitlines()
    print("\n".join(rows[:1] + [r for r in rows[1:] if "start_server" in r or "chd_ct" in r]))


def query(args, profile):
    from . import jobs

    runtime = Path(args.runtime or profile["runtime"]).expanduser().resolve()
    job = Path(args.job_dir).expanduser().resolve() if args.job_dir else latest(runtime)
    if args.query == "results":
        from .results import show_results

        if job is None:
            print("没有任务记录；用 --job-dir 指定历史 workflow 目录，或在面板 o 中选择。")
            return 0
        try:
            while True:
                show_results(job, args.step)
                if not args.follow:
                    return 0
                time.sleep(3)
        except KeyboardInterrupt:
            print("\n退出结果查看，后台任务继续运行。")
            return 0
    if args.query not in {"gpu", "processes", "history"} and (
        job is None or not (job / "job_state.json").is_file()
    ):
        print("没有任务记录；先新建任务，或用 --job-dir 指定历史 workflow 目录。")
        return 0
    if args.query == "logs":
        from .log_progress import show_progress

        tail_logs(
            lambda: log_paths(job, args.step, args.model),
            args.lines,
            args.follow,
            progress=lambda: show_progress(job, args.step, args.model),
        )
        return 0
    if args.query == "gpu" and args.follow:
        watch_gpu()
        return 0
    try:
        while True:
            if args.query == "status":
                jobs.show_status(runtime, job)
            elif args.query == "gpu":
                show_gpu()
            elif args.query == "processes":
                show_processes()
            elif args.query == "models":
                show_models(job, args.step)
            elif args.query == "history":
                show_history(runtime, args.results or profile["results"])
            if not args.follow:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n退出查看，后台任务继续运行。")
    return 0
