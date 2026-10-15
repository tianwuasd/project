"""Primary server dispatcher; each job runs exactly one requested CHD function."""

import shutil
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..quickstart.commands import build_command
from ..quickstart.diagnosis import OUTPUT_KEYS
from ..recovery import archive_child
from .bootstrap import ensure_python, environment, run_logged
from .launcher import ROOT, gpu_pool, query_gpus, select_gpus, write_json
from .resources import wait_for_gpus
from .settings import effective_device, resolve_settings, task_configuration


def select_with_handoff(count, requested, visible, minimum, recent_gpus):
    rows = query_gpus()
    try:
        return select_gpus(rows, count, requested, visible, minimum)
    except ValueError:
        # Only wait for utilization left by this workflow; busy/unauthorized cards
        # remain ineligible, and we never wait for another user's compute process.
        candidates = select_gpus(
            [dict(row, util=0) if row["uuid"] in recent_gpus else row for row in rows],
            count,
            requested,
            visible,
            minimum,
        )
        from ..imagechd.stage_queue import check_gpu_handoff

        for gpu in candidates:
            if gpu["uuid"] in recent_gpus:
                check_gpu_handoff(gpu["uuid"], used=True)
        return select_gpus(query_gpus(), count, requested, visible, minimum)


def run(args, profile, *, output_dir=None, lock_held=False, recent_gpus=()):
    if sys.platform != "linux":
        print("服务器入口用于 Linux；本机请使用 start.py 或独立 Python 脚本。")
        return 1
    interactive = sys.stdin.isatty() and not args.non_interactive
    output, state, handlers = None, {}, {}
    interrupt_code = 130

    def on_signal(sig, frame):
        nonlocal interrupt_code
        interrupt_code = 128 + sig
        raise KeyboardInterrupt(f"收到信号 {sig}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        handlers[sig] = signal.signal(sig, on_signal)
    try:
        runtime, results, settings_file, settings = resolve_settings(args, profile, interactive)
        if args.task in {"preprocess", "train", "predict", "evaluate", "test"}:
            if args.task == "preprocess":
                if not Path(args.dataset).exists():
                    raise ValueError(f"数据目录不存在：{args.dataset}；用 --dataset 覆盖默认路径")
                if Path(args.prepared).exists() and not getattr(args, "resume", False):
                    raise ValueError("预处理目录已存在；训练/测试可直接复用，或指定新的 --prepared")
            elif not (Path(args.prepared) / "dataset.json").is_file():
                raise ValueError("缺少预处理结果；先独立运行 preprocess_server.sh")
        if args.task in {"predict", "test"} and (
            not args.models or not Path(args.models).expanduser().exists()
        ):
            raise ValueError("请指定 --models 完整六阶段模型目录")
        if args.task == "evaluate" and (
            not args.predictions or not Path(args.predictions).expanduser().is_dir()
        ):
            raise ValueError("请指定 --predictions")
        if args.task == "train" and not Path(args.config).is_file():
            raise ValueError("模型配置不存在：" + args.config)
        for folder in (runtime, results):
            folder.mkdir(parents=True, exist_ok=True)
        if (
            shutil.disk_usage(runtime).free < (2 if args.python else 20) * 2**30
            or shutil.disk_usage(results).free < 2 * 2**30
        ):
            raise ValueError("环境或结果盘可用空间不足")
        import fcntl

        with nullcontext() if lock_held else (runtime / ".job.lock").open("a") as lock:
            if not lock_held:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output = output_dir or results / (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
            )
            resuming = bool(getattr(args, "resume", False))
            output.mkdir(parents=True, exist_ok=resuming)
            if resuming:
                archive_child(output / "environment")
                archive_child(output / "preflight")
            state = {
                "status": "running",
                "task": args.task,
                "prepared": args.prepared,
                "server_profile_date": profile["source_updated"],
            }
            write_json(output / "status.json", state)
            write_json(settings_file, settings)
            write_json(output / "server-settings.json", task_configuration(args))
            print("日志和结果：" + str(output), flush=True)
            env = environment(runtime, ROOT, profile.get("threads", 2))
            device = effective_device(args)
            visible = env.get("CUDA_VISIBLE_DEVICES")
            count = args.gpu_count if args.task in {"train", "environment"} else 1
            requested = args.gpu
            pool = []
            if device == "cuda":
                pool = [r["uuid"] for r in gpu_pool(query_gpus(), requested, visible)]
            python = (
                Path(args.python).expanduser().resolve()
                if args.python
                else ensure_python(runtime, ROOT, env, output / "server.log", profile.get("existing_conda"))
            )
            if not python.is_file():
                raise ValueError("Python 路径不存在")

            def resource_update(status, detail):
                state.update(status=status, gpu_resources=detail)
                write_json(output / "status.json", state)

            # Training checks each stage's actual device in its worker, including
            # cards that become available later. Do not block on a full pool here.
            gpus = wait_for_gpus(pool, count, query_gpus, resource_update) if pool else []
            state["gpus"] = gpus
            state["gpu_pool"] = pool
            state["gpu_limit"] = count
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                pool if args.task == "train" else [r["uuid"] for r in gpus]
            )
            args.gpu_ids = pool if args.task == "train" and pool else None
            args.gpu_limit = count if args.gpu_ids else None
            write_json(output / "status.json", state)
            # Probe only currently idle cards; the rest remain candidates, not reservations.
            probes = gpus[:1] if args.task == "train" else gpus
            for index, gpu in enumerate(probes or [None]):
                probe_env = dict(env, CUDA_VISIBLE_DEVICES=gpu["uuid"] if gpu else "")
                command = [
                    python,
                    ROOT / "start.py",
                    "--task",
                    "environment",
                    "--non-interactive",
                    "--device",
                    device,
                    "--output",
                    output / "environment" / str(index),
                ]
                code = run_logged(command, probe_env, output / "server.log")
                if code:
                    state.update(status="failed", exit_code=code)
                    return code
            commands = []
            if args.task == "train" and args.mode == "train":
                commands.append(
                    build_command(args, ROOT, python, output / "preflight", device, mode="preflight")
                )
            if args.task != "environment":
                commands.append(build_command(args, ROOT, python, output / args.task, device))
            for command in commands:
                state["phase"] = "preflight" if "--mode" in command and "preflight" in command else args.task
                state["phase_started_at"] = time.time()
                write_json(output / "status.json", state)
                if device == "cuda" and args.task != "train":
                    gpus = wait_for_gpus(pool, 1, query_gpus, resource_update)
                    env["CUDA_VISIBLE_DEVICES"] = gpus[0]["uuid"]
                    state["gpus"] = gpus
                code = run_logged(command, env, output / "server.log")
                if code:
                    state.update(status="failed", exit_code=code)
                    return code
            if args.task == "train" and args.mode in {"train", "smoke", "preflight"}:
                settings["last_models"] = str(output / "train")
            if args.task in {"predict", "test"}:
                settings["last_predictions"] = str(
                    output / ("test/predictions" if args.task == "test" else "predict")
                )
            write_json(settings_file, settings)
            if args.task in OUTPUT_KEYS:
                settings[OUTPUT_KEYS[args.task]] = str(output / args.task)
                write_json(settings_file, settings)
            state["status"] = "passed"
            print("已完成 " + args.task + "：" + str(output))
            return 0
    except KeyboardInterrupt as error:
        state.update(status="interrupted", error=str(error))
        return interrupt_code
    except Exception as error:
        state.update(status="failed", error=str(error))
        print("未通过：" + str(error), file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if output:
            write_json(output / "status.json", state)
