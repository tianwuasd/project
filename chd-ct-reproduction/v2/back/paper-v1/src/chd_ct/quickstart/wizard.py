"""Chinese guided launcher. This module must remain importable with no scientific dependencies."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from .datasets import find_manifests
from .environment import check_environment

ROOT = Path(__file__).resolve().parents[3]
STEPS = ("environment", "data_validation", "synthetic_smoke", "dataset_smoke")
TITLES = {
    "environment": "环境检查",
    "data_validation": "数据校验",
    "synthetic_smoke": "合成数据全流程测试",
    "dataset_smoke": "所选数据集小规模测试",
}


def format_command(arguments):
    if os.name == "nt":
        # A PowerShell-safe copyable command, including paths with whitespace or apostrophes.
        return "& " + " ".join("'" + str(x).replace("'", "''") + "'" for x in arguments)
    return shlex.join(map(str, arguments))


def select_path(kind):
    try:
        import tkinter as tk
        from tkinter import filedialog

        window = tk.Tk()
        window.withdraw()
        try:
            window.attributes("-topmost", True)
            path = (
                filedialog.askdirectory(title="选择含 manifest.csv 的数据目录", parent=window)
                if kind == "folder"
                else filedialog.askopenfilename(
                    title="选择数据清单 CSV", parent=window, filetypes=[("数据清单", "*.csv")]
                )
            )
        finally:
            window.destroy()
        if not path:
            raise ValueError("已取消选择数据集。")
        return path
    except (ImportError, RuntimeError) as error:
        print(f"无法打开选择窗口（{error}），可以粘贴路径。")
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        print("图形窗口不可用，可以直接粘贴路径。")
    return input("数据目录或 CSV 路径：").strip()


def choose_dataset(args):
    if args.demo:
        return None
    selection = args.dataset
    if selection is None:
        if args.non_interactive:
            raise ValueError("非交互运行需要 --demo 或 --dataset。")
        print(
            "\n请选择数据集：\n  1. 没有准备好数据，先用合成演示（默认）\n  2. 选择数据文件夹\n  3. 选择 CSV 清单\n  4. 粘贴数据路径"
        )
        choice = input("输入数字 [1]：").strip() or "1"
        if choice == "1":
            return None
        if choice not in ("2", "3", "4"):
            raise ValueError("请选择 1、2、3 或 4。")
        selection = (
            input("粘贴路径：").strip()
            if choice == "4"
            else select_path("folder" if choice == "2" else "file")
        )
    candidates = find_manifests(selection)
    if len(candidates) > 1:
        if args.non_interactive:
            raise ValueError("目录含多个数据清单，请用 --dataset 指定准确 CSV 路径。")
        for index, path in enumerate(candidates, 1):
            print(f"  {index}. {path}")
        try:
            index = int(input("选择数据清单编号：")) - 1
        except ValueError as error:
            raise ValueError("数据清单编号应为数字。") from error
        if not 0 <= index < len(candidates):
            raise ValueError("数据清单编号不在范围内。")
        return candidates[index]
    return candidates[0]


def write_report(output, report):
    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "CHD CT 一键检查报告",
        f"总结果：{report['status']}",
        f"模式：{report['mode']}",
        f"选择的数据：{report.get('dataset') or '合成演示'}",
        "",
    ]
    for name, step in report["steps"].items():
        lines.append(f"{TITLES[name]}：{step['status']} {step.get('message', '')}")
    environment = report.get("environment", {})
    if environment:
        lines.extend(
            [
                "",
                f"Python：{environment.get('python', '未知')}  {environment.get('executable', '')}",
                f"CUDA：{environment.get('cuda', False)}  GPU：{environment.get('gpu', '无')}",
                f"GPU 显存：{environment.get('gpu_memory_gib', '未知')} GiB",
                f"磁盘剩余：{environment.get('disk_free_gib', '未知')} GiB",
            ]
        )
        for name, package in environment.get("packages", {}).items():
            lines.append(f"{name}：{package.get('version') or package.get('error')}")
    lines.extend(["", report.get("message", ""), "", "后续操作：", *report.get("next_steps", [])])
    lines.append("\n小规模测试只验证环境、数据接口和流程，不代表医学效果或正式训练显存足够。")
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_worker(command, root, output, log, manifest=None, device="cpu"):
    arguments = [
        sys.executable,
        "-m",
        "chd_ct.quickstart.worker",
        command,
        "--root",
        str(root),
        "--output",
        str(output),
        "--device",
        device,
    ]
    if manifest:
        arguments.extend(["--manifest", str(manifest)])
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    tail = deque(maxlen=8)
    with log.open("a", encoding="utf-8") as stream:
        with subprocess.Popen(
            arguments,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            try:
                for line in process.stdout:
                    if line.strip():
                        tail.append(line.strip())
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    if code:
        detail = tail[-1] if tail else "没有错误输出"
        raise RuntimeError(f"{command} 执行失败（退出码 {code}）：{detail}。详细日志：{log.name}。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHD CT 一键环境检查、数据选择与试跑")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", help="含清单的数据目录，或 manifest.csv")
    selection.add_argument("--demo", action="store_true", help="用合成数据测试")
    parser.add_argument(
        "--mode", choices=["check", "smoke"], default="smoke", help="check 仅检查；默认 smoke 还会试跑"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--output", help="新的输出目录；默认 runs/quickstart/时间戳")
    args = parser.parse_args(argv)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = Path(args.output).expanduser().resolve() if args.output else ROOT / "runs/quickstart" / tag
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"不能创建输出目录，请选一个新的可写目录：{output}\n{error}")
        return 1
    # Keep temporary files local, including PyTorch's temporary compiler artifacts.
    (output / "tmp").mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(output / "tmp")
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    report = {
        "status": "running",
        "mode": args.mode,
        "steps": {name: {"status": "not_run"} for name in STEPS},
        "next_steps": [],
        "dataset": None,
    }
    current = None
    print("CHD CT 一键启动：选择数据 → 检查环境 → 校验数据 → 小规模试跑", flush=True)
    print(f"报告与日志：{output}", flush=True)
    try:
        manifest = choose_dataset(args)
        report["dataset"] = str(manifest) if manifest else None
        current = "environment"
        print("\n[1/4] 检查 Python、依赖、磁盘、CPU 与 CUDA……", flush=True)
        environment = check_environment(output)
        report["environment"] = environment
        if not environment["ok"]:
            install = format_command([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])
            report["next_steps"] = [
                f"在 PowerShell（Windows）或终端执行：{install}",
                "如果当前 Python 不合适，按 README 的安装步骤创建 .venv 后再双击 start.bat。",
            ]
            raise ValueError("环境未就绪：" + "；".join(environment["errors"]))
        if args.device == "cuda" and not environment.get("cuda"):
            raise ValueError("指定了 CUDA，但实际 CUDA 运算不可用。使用 --device cpu 或检查驱动/PyTorch。")
        device = "cuda" if args.device != "cpu" and environment.get("cuda") else "cpu"
        report["device"] = device
        report["steps"][current] = {"status": "passed", "message": f"本次使用 {device}"}
        print(
            f"Python {environment['python']}；本次设备：{device}；可用磁盘 {environment['disk_free_gib']} GiB"
        )
        for warning in environment.get("warnings", []):
            print(warning)
        if manifest:
            current = "data_validation"
            print("\n[2/4] 校验全部所选数据（网格、标签、患者划分）……", flush=True)
            run_worker("validate", ROOT, output, output / "run.log", manifest, device)
            info = json.loads((output / "data_validation.json").read_text(encoding="utf-8"))
            report["data"] = info
            report["steps"][current] = {
                "status": "passed",
                "message": f"{info['cases']} 例，划分 {info['splits']}",
            }
        else:
            report["steps"]["data_validation"] = {
                "status": "skipped",
                "message": "选择的是自动生成的合成数据",
            }
        if args.mode == "check":
            report.update(
                status="checked", message="检查完成；未训练、未推理。再次运行默认模式可进行小规模测试。"
            )
        else:
            current = "synthetic_smoke"
            print("\n[3/4] 先用合成数据测试完整训练和推理……", flush=True)
            run_worker("synthetic", ROOT, output / "synthetic", output / "run.log", device=device)
            report["steps"][current] = {"status": "passed", "message": "八阶段训练及完整推理通过"}
            if manifest:
                current = "dataset_smoke"
                print("\n[4/4] 对所选数据的小样本副本试跑（最多 4 例，最大边长 32）……", flush=True)
                run_worker("selected", ROOT, output / "selected", output / "run.log", manifest, device)
                result = json.loads((output / "selected/result.json").read_text(encoding="utf-8"))
                report["steps"][current] = result
                report.update(status=result["status"], message=result["message"])
            else:
                report["steps"]["dataset_smoke"] = {"status": "skipped", "message": "未选择真实数据集"}
                report.update(
                    status="passed", message="合成数据完整试跑通过；可以重新启动并选择自己的数据集。"
                )
            if report["status"] == "partial":
                report["next_steps"].append(
                    "补齐 train/val 的 initial_label 后重跑，才能验证七路融合和诊断。"
                )
            elif manifest:
                report["next_steps"].append(
                    "完整小规模流程已通过；准备正式 GPU 训练时按 README 使用 paper 配置。"
                )
        report["next_steps"].append("数据标签语义、疾病诊断准确率和正式大模型资源需求不由本检查证明。")
        write_report(output, report)
        print(f"\n{report['message']}\n报告：{output / 'report.txt'}", flush=True)
        return 2 if report["status"] == "partial" else 0
    except KeyboardInterrupt:
        report.update(status="interrupted", message="用户中止；保留已有日志和结果。")
        if current:
            report["steps"][current] = {"status": "interrupted"}
        write_report(output, report)
        return 130
    except Exception as error:
        report.update(status="failed", message=str(error))
        if current:
            report["steps"][current] = {"status": "failed", "message": str(error)}
        if not report["next_steps"]:
            report["next_steps"] = [
                "查看 run.log 的具体错误，并按 docs/data-guide.md 检查标签、文件路径和患者划分。",
                "数据尚未准备好时，可以重新双击 start.bat 选择“合成演示”。",
            ]
        write_report(output, report)
        print(
            f"\n未通过：{error}\n处理建议：{' '.join(report['next_steps'])}\n报告：{output / 'report.txt'}",
            flush=True,
        )
        return 1
