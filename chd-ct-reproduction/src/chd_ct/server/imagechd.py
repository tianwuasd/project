"""Primary server dispatcher; each job runs exactly one requested CHD function."""

import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..quickstart.commands import build_command
from ..quickstart.diagnosis import DIAGNOSIS_TASKS, OUTPUT_KEYS
from .bootstrap import ensure_python, environment, run_logged
from .launcher import ROOT, query_gpus, select_gpus, write_json
from .settings import resolve_settings


def run(args, profile):
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
                if Path(args.prepared).exists():
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

        with (runtime / ".job.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output = results / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
            output.mkdir()
            state = {
                "status": "running",
                "task": args.task,
                "prepared": args.prepared,
                "server_profile_date": profile["source_updated"],
            }
            write_json(output / "status.json", state)
            write_json(settings_file, settings)
            write_json(output / "server-settings.json", settings)
            print("日志和结果：" + str(output), flush=True)
            env = environment(runtime, ROOT, profile.get("threads", 2))
            device = (
                "cpu"
                if args.task in DIAGNOSIS_TASKS
                or args.task in {"preprocess", "evaluate"}
                or (args.task == "train" and args.mode == "check")
                else args.device
            )
            if device == "auto":
                device = "cuda"
            visible = env.get("CUDA_VISIBLE_DEVICES")
            count = args.gpu_count if args.task in {"train", "environment"} else 1
            requested = args.gpu
            if count == 1 and args.task not in {"train", "environment"} and requested != "auto":
                requested = requested.split(",")[0].strip()
            minimum = 6000
            if device == "cuda":
                select_gpus(query_gpus(), count, requested, visible, minimum)
            python = (
                Path(args.python).expanduser().resolve()
                if args.python
                else ensure_python(runtime, ROOT, env, output / "server.log", profile.get("existing_conda"))
            )
            if not python.is_file():
                raise ValueError("Python 路径不存在")
            gpus = select_gpus(query_gpus(), count, requested, visible, minimum) if device == "cuda" else []
            state["gpus"] = gpus
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in gpus)
            args.gpu_ids = [gpu["uuid"] for gpu in gpus] if args.task == "train" and len(gpus) > 1 else None
            # Check every chosen card in a short-lived process before submitting work.
            for index, gpu in enumerate(gpus or [None]):
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
