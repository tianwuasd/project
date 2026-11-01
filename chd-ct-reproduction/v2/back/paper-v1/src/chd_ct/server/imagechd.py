"""Server dispatch for independently runnable preprocess/train/predict scripts."""

import signal
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .bootstrap import ensure_python, environment, run_logged
from .launcher import ROOT, choose, personal_path, query_gpus, select_gpu, write_json


def run(args, profile):
    if sys.platform != "linux":
        print("服务器入口面向 Linux；本机请使用三个独立 Python 脚本。")
        return 1
    interactive = sys.stdin.isatty() and not args.non_interactive
    output, state, handlers = None, {}, {}
    code_on_interrupt = 130

    def on_signal(sig, frame):
        nonlocal code_on_interrupt
        code_on_interrupt = 128 + sig
        raise KeyboardInterrupt(f"收到信号 {sig}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        handlers[sig] = signal.signal(sig, on_signal)
    try:
        runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", interactive))
        results = personal_path(
            choose(args.results, None, profile["results"], "日志/模型/预测结果目录", interactive)
        )
        prepared = personal_path(
            choose(
                args.prepared,
                None,
                profile["prepared"],
                "预处理结果目录（预处理时为输出，训练/预测时为输入）",
                interactive,
            )
        )
        source, checkpoint = None, None
        if args.task == "preprocess":
            selected = choose(args.dataset, None, "", "已解压数据目录或待预测 NIfTI", interactive)
            if not selected or not Path(selected).expanduser().exists():
                raise ValueError("请用 --dataset 指定已解压的数据目录或 NIfTI。")
            source = Path(selected).expanduser().resolve()
            if prepared.exists():
                raise ValueError("预处理输出已存在；可直接供训练/预测复用，或指定新的 --prepared。")
        else:
            if not (prepared / "dataset.json").is_file():
                raise ValueError(
                    "预处理结果缺失；先独立运行 preprocess_imagechd.py 或服务器 preprocess 任务。"
                )
            if args.task == "predict":
                value = choose(args.checkpoint, None, "", "七结构模型权重 best.pt", interactive)
                if not value or not Path(value).expanduser().is_file():
                    raise ValueError("预测必须提供 --checkpoint")
                checkpoint = Path(value).expanduser().resolve()
        config = Path(args.config).expanduser().resolve() if args.config else ROOT / "configs/imagechd7.yaml"
        if args.task == "train" and not config.is_file():
            raise ValueError("模型配置不存在")
        for folder in (runtime, results):
            folder.mkdir(parents=True, exist_ok=True)
        import fcntl
        import shutil

        if shutil.disk_usage(runtime).free < (2 if args.python else 20) * 2**30:
            raise ValueError("runtime 可用空间不足：复用 Python 至少 2 GiB，创建环境至少 20 GiB")
        if shutil.disk_usage(results).free < 2 * 2**30:
            raise ValueError("结果盘可用空间不足 2 GiB")
        with (runtime / ".job.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output = results / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
            output.mkdir()
            state = {
                "status": "running",
                "task": args.task,
                "prepared": str(prepared),
                "server_profile_date": profile["source_updated"],
            }
            write_json(output / "status.json", state)
            print(f"日志和结果：{output}", flush=True)
            env = environment(runtime, ROOT, profile.get("threads", 2))
            device = "cpu" if args.task == "preprocess" else args.device
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
                    "--demo",
                    "--mode",
                    "check",
                    "--non-interactive",
                    "--device",
                    device,
                    "--output",
                    output / "environment",
                ]
            ]
            if args.task == "preprocess":
                command = [
                    python,
                    ROOT / "preprocess_imagechd.py",
                    "--input",
                    source,
                    "--output",
                    prepared,
                    "--size",
                    str(args.size),
                    "--seed",
                    str(args.seed),
                ]
                if args.for_prediction:
                    command += ["--for-prediction"]
                if args.split_file:
                    command += ["--split-file", args.split_file]
                if args.limit:
                    command += ["--limit", str(args.limit)]
                commands.append(command)
            elif args.task == "train":

                def training_command(mode, folder):
                    return [
                        python,
                        ROOT / "train_imagechd.py",
                        "--prepared",
                        prepared,
                        "--config",
                        config,
                        "--output",
                        output / folder,
                        "--mode",
                        mode,
                        "--device",
                        device,
                    ]

                if args.mode == "train":
                    commands.append(training_command("preflight", "preflight"))
                commands.append(training_command(args.mode, "training"))
            else:
                command = [
                    python,
                    ROOT / "predict_imagechd.py",
                    "--prepared",
                    prepared,
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    output / "predictions",
                    "--device",
                    device,
                ]
                for flag, value in (("--case-id", args.case_id), ("--split", args.split)):
                    if value:
                        command += [flag, value]
                if args.allow_smoke:
                    command += ["--allow-smoke"]
                commands.append(command)
            for command in commands:
                code = run_logged(command, env, output / "server.log")
                if code:
                    state.update(status="failed", exit_code=code)
                    return code
            state["status"] = "passed"
            print(f"已完成 {args.task}：{output}")
            return 0
    except KeyboardInterrupt as error:
        state.update(status="interrupted", error=str(error))
        return code_on_interrupt
    except Exception as error:
        state.update(status="failed", error=str(error))
        print(f"未通过：{error}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if output:
            write_json(output / "status.json", state)
