"""Primary server dispatcher; each job runs exactly one requested CHD function."""

import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..quickstart.commands import build_command
from .bootstrap import ensure_python, environment, run_logged
from .launcher import ROOT, choose, personal_path, query_gpus, select_gpu, write_json


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
        runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", interactive))
        results = personal_path(
            choose(args.results, None, profile["results"], "模型/日志/预测结果目录", interactive)
        )
        if args.task in {"preprocess", "train", "predict", "evaluate"}:
            args.prepared = str(
                personal_path(choose(args.prepared, None, profile["prepared"], "预处理目录", interactive))
            )
            if args.task == "preprocess":
                args.dataset = choose(args.dataset, None, "", "已解压 ImageCHD 目录或待预测影像", interactive)
                if not args.dataset or not Path(args.dataset).expanduser().exists():
                    raise ValueError("预处理需要有效 --dataset")
                if Path(args.prepared).exists():
                    raise ValueError("预处理目录已存在；训练预测可直接复用，或指定新的输出目录")
            elif not (Path(args.prepared) / "dataset.json").is_file():
                raise ValueError("缺少预处理结果；先独立运行 preprocess_server.sh")
        if args.task == "predict":
            args.models = choose(args.models, None, "", "完整六阶段模型目录", interactive)
            if not args.models or not Path(args.models).expanduser().exists():
                raise ValueError("请指定 --models")
        if args.task == "evaluate":
            args.predictions = choose(args.predictions, None, "", "预测结果目录", interactive)
            if not args.predictions or not Path(args.predictions).expanduser().is_dir():
                raise ValueError("请指定 --predictions")
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
            print("日志和结果：" + str(output), flush=True)
            env = environment(runtime, ROOT, profile.get("threads", 2))
            device = (
                "cpu"
                if args.task in {"preprocess", "evaluate"} or (args.task == "train" and args.mode == "check")
                else args.device
            )
            if device == "auto":
                device = "cuda"
            visible = env.get("CUDA_VISIBLE_DEVICES")
            if device == "cuda":
                select_gpu(query_gpus(), args.gpu, visible)
            python = (
                Path(args.python).expanduser().resolve()
                if args.python
                else ensure_python(runtime, ROOT, env, output / "server.log", profile.get("existing_conda"))
            )
            if not python.is_file():
                raise ValueError("Python 路径不存在")
            if device == "cuda":
                gpu = select_gpu(query_gpus(), args.gpu, visible)
                env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
                state["gpu"] = gpu
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            commands = [
                [
                    python,
                    ROOT / "start.py",
                    "--task",
                    "environment",
                    "--non-interactive",
                    "--device",
                    device,
                    "--output",
                    output / "environment",
                ]
            ]
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
