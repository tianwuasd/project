"""One primary workflow, with independent task selection and environment checks."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..paths import PACKAGE_PARENT, project_root
from ..server.bootstrap import run_logged
from ..server.launcher import choose
from .commands import add_task_arguments, build_command, choose_task
from .diagnosis import DIAGNOSIS_TASKS, resolve_inputs
from .environment import check_environment

ROOT = project_root()


def choose_path(label):
    print(label + "；输入路径，或输入 browse 打开目录选择窗口。")
    value = input("路径：").strip().strip('"').strip("'")
    if value.lower() == "browse":
        try:
            import tkinter as tk
            from tkinter import filedialog

            window = tk.Tk()
            window.withdraw()
            try:
                value = filedialog.askdirectory(title=label, parent=window)
            finally:
                window.destroy()
        except Exception as error:
            print("无法打开选择窗口：" + str(error))
            value = input("粘贴路径：").strip().strip('"').strip("'")
    if not value:
        raise ValueError("未选择路径")
    return str(Path(value).expanduser().resolve())


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description="ImageCHD 主流程：环境检查、独立预处理/训练/预测/评估")
    add_task_arguments(parser)
    parser.add_argument("--output", help="新的任务日志/结果目录")
    args = parser.parse_args(argv)
    interactive = sys.stdin.isatty() and not args.non_interactive
    output = None
    output_created = False
    report = {"status": "running"}
    try:
        args.task = choose_task(args, interactive)
        if args.task in DIAGNOSIS_TASKS:
            resolve_inputs(args, {}, choose, interactive)
        if interactive:
            if args.task == "preprocess" and not args.dataset:
                args.dataset = choose_path("选择已解压 ImageCHD 数据")
            if args.task in {"train", "predict", "evaluate", "test"} and not args.prepared:
                args.prepared = choose_path("选择含 dataset.json 的预处理目录")
            if args.task in {"predict", "test"} and not args.models:
                args.models = choose_path("选择含 models.json 的模型目录")
            if args.task == "evaluate" and not args.predictions:
                args.predictions = choose_path("选择含 prediction-report.json 的预测目录")
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else ROOT / "runs/quickstart" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
        )
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        report.update(task=args.task, output=str(output))
        print("环境检查与结果：" + str(output), flush=True)
        env = check_environment(output)
        report["environment"] = env
        if not env["ok"]:
            raise ValueError("环境未就绪：" + "；".join(env["errors"]) + "。请按 README 安装依赖。")
        if args.device == "cuda" and not env.get("cuda"):
            raise ValueError("CUDA 实际运算不可用")
        device = "cuda" if args.device != "cpu" and env.get("cuda") else "cpu"
        if args.task in {"preprocess", "evaluate"}:
            device = "cpu"
        report["device"] = device
        if args.task == "environment":
            report["status"] = "checked"
            return 0
        environment = dict(
            os.environ, PYTHONPATH=str(PACKAGE_PARENT), PYTHONIOENCODING="utf-8", PYTHONUTF8="1"
        )
        commands = []
        if args.task == "train" and args.mode == "train":
            commands.append(
                build_command(args, ROOT, sys.executable, output / "preflight", device, mode="preflight")
            )
        commands.append(build_command(args, ROOT, sys.executable, output / args.task, device))
        for command in commands:
            code = run_logged(command, environment, output / "run.log")
            if code:
                report.update(status="failed", exit_code=code)
                return code
        report["status"] = "passed"
        print("已完成 " + args.task + "；结果：" + str(output))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error))
        print("未通过：" + str(error), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if output_created:
            (output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
