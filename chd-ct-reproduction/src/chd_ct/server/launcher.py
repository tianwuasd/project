"""zmic44 defaults, one GPU per job, existing quickstart validation and smoke."""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..quickstart.datasets import find_manifests
from .bootstrap import ensure_python, environment, run_logged

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME = "/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime"
DEFAULT_RESULTS = "/data_nas/zhangruichen/chd_ct_results"


def write_json(path, data):
    tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def choose(value, saved, default, label, interactive):
    if value is not None:
        return value
    suggestion = saved or default
    if not interactive:
        return suggestion
    value = input(f"{label} [{suggestion}]：").strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in {"'", chr(34)}:
        value = value[1:-1]
    return value or suggestion


def personal_path(value):
    path = Path(value).expanduser().resolve()
    shared = {
        "/",
        "/home",
        "/data5",
        "/data_nas",
        "/data5/zhougaowei",
        "/tmp",
        "/usr",
        "/etc",
        "/opt",
        "/var",
    }
    if str(path) in shared or path == Path.home() or path == ROOT or path in ROOT.parents:
        raise ValueError("请使用本项目专用的个人子目录。")
    return path


def resolve_manifest(value):
    path = Path(value).expanduser().resolve()
    raw = [path, path / "ImageCHD_dataset"]
    if any(p.is_dir() and next(p.glob("ct_*_label.nii.gz"), None) for p in raw):
        raise ValueError(
            "检测到原始 ImageCHD：7 类标签编号与本项目不同，且缺 SVC/IVC/PV 和 initial_label。"
            "不能直接训练完整复现；见 docs/imagechd-archive.md。可用 --demo 测试服务器。"
        )
    if path.is_dir() and next(path.glob("*.z01"), None):
        raise ValueError("这是未解压的分卷压缩包；见 docs/imagechd-archive.md。")
    manifests = find_manifests(path)
    if len(manifests) != 1:
        raise ValueError("请选择一个明确的 manifest.csv；清单缺失或候选多于一个。")
    return manifests[0]


def query_gpus():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name, free, util = [v.strip() for v in line.split(",")]
        rows.append(dict(index=index, uuid=uuid, name=name, free=int(free), util=int(util)))
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    busy = {line.strip() for line in processes.stdout.splitlines()}
    for row in rows:
        row["busy"] = row["uuid"] in busy
    return rows


def select_gpu(rows, requested="auto", visible=None, minimum=6000):
    allowed = None if visible is None else {s.strip() for s in visible.split(",") if s.strip()}
    candidates = []
    for row in rows:
        if allowed is not None and not ({row["index"], row["uuid"]} & allowed):
            continue
        if requested != "auto" and requested not in (row["index"], row["uuid"]):
            continue
        if not row["busy"] and row["free"] >= minimum and row["util"] <= 10:
            candidates.append(row)
    if not candidates:
        raise ValueError(
            "指定/分配范围内没有满足条件的空闲 GPU。请查看 nvidia-smi，等待或选择其他 GPU；"
            "不会越过 CUDA_VISIBLE_DEVICES，也不会自动改用 CPU。"
        )
    return max(candidates, key=lambda row: row["free"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="zmic44 / Linux CHD CT 引导入口")
    parser.add_argument("--runtime", help="专用环境/缓存目录，默认本地盘")
    parser.add_argument("--results", help="结果目录，默认 NAS")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--dataset", help="符合项目规范的 manifest.csv 或目录")
    data.add_argument("--demo", action="store_true", help="合成数据，不需要真实病例")
    parser.add_argument("--mode", choices=["check", "smoke", "preflight", "train"], default="smoke")
    parser.add_argument("--gpu", default="auto", help="auto、nvidia-smi 物理编号或完整 GPU UUID")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--python", help="复用指定 Python，跳过安装；由环境检查判断是否可用")
    parser.add_argument("--config", default=str(ROOT / "configs/server3090.yaml"))
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        print("此入口面向 Linux x86_64；Windows 请使用 start.bat。")
        return 1
    interactive = sys.stdin.isatty() and not args.non_interactive
    state, output = {}, None
    interrupt_code = 130
    previous_handlers = {}

    def on_signal(signum, frame):
        nonlocal interrupt_code
        interrupt_code = 128 + signum
        raise KeyboardInterrupt(f"收到终止信号 {signum}")

    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, on_signal)
    try:
        settings_path = ROOT / ".server-settings.json"
        saved = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        runtime = personal_path(
            choose(
                args.runtime, saved.get("runtime"), DEFAULT_RUNTIME, "环境与缓存目录（本地盘）", interactive
            )
        )
        results = personal_path(
            choose(args.results, saved.get("results"), DEFAULT_RESULTS, "结果目录（NAS）", interactive)
        )
        selection = (
            "demo"
            if args.demo
            else choose(
                args.dataset,
                saved.get("dataset"),
                "demo",
                "数据清单路径；demo 为合成测试",
                interactive,
            )
        )
        manifest = None if selection == "demo" else resolve_manifest(selection)
        if args.mode in {"preflight", "train"} and (not manifest or args.device != "cuda"):
            raise ValueError("正式尺寸预检/训练需要真实数据 manifest 与 CUDA。先用 smoke 检查软件。")
        config = Path(args.config).expanduser().resolve()
        if args.mode in {"preflight", "train"} and not config.is_file():
            raise ValueError(f"配置文件不存在：{config}")
        for folder, minimum in [
            (runtime, 20 if not args.python else 2),
            (results, 30 if args.mode in {"preflight", "train"} else 2),
        ]:
            folder.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(folder).free < minimum * 1024**3:
                raise ValueError(f"{folder} 可用空间不足 {minimum} GiB")
        import fcntl

        with (runtime / ".job.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("这个 runtime 已有任务运行；请查看其日志或使用独立 runtime。") from None
            tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
            output = results / tag
            output.mkdir(exist_ok=False)
            state = dict(
                status="running",
                mode=args.mode,
                output=str(output),
                dataset=str(manifest),
                started=datetime.now().isoformat(),
            )
            write_json(output / "status.json", state)
            write_json(settings_path, dict(runtime=str(runtime), results=str(results), dataset=selection))
            print(f"本次日志与报告：{output}", flush=True)
            env = environment(runtime, ROOT)
            original_visible = env.get("CUDA_VISIBLE_DEVICES")
            minimum_gpu = 18000 if args.mode in {"preflight", "train"} else 6000
            if args.device == "cuda":
                gpu = select_gpu(query_gpus(), args.gpu, original_visible, minimum_gpu)
                print(f"候选 GPU：{gpu['index']} {gpu['name']}，空闲 {gpu['free']} MiB", flush=True)
            python = (
                Path(args.python).expanduser().resolve()
                if args.python
                else ensure_python(
                    runtime,
                    ROOT,
                    env,
                    output / "server.log",
                )
            )
            if not python.is_file():
                raise ValueError(f"Python 不存在：{python}")
            if args.device == "cuda":
                gpu = select_gpu(query_gpus(), args.gpu, original_visible, minimum_gpu)
                env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
                state["gpu"] = gpu
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            command = [
                python,
                ROOT / "start.py",
                "--non-interactive",
                "--device",
                args.device,
                "--mode",
                "check" if args.mode == "check" else "smoke",
                "--output",
                output / "quickstart",
            ]
            command += ["--dataset", manifest] if manifest else ["--demo"]
            code = run_logged(command, env, output / "server.log")
            if code:
                state["status"] = "partial" if code == 2 else "failed"
                return code
            if args.mode in {"preflight", "train"}:
                code = run_logged(
                    [
                        python,
                        "-m",
                        "chd_ct.server.training",
                        "--manifest",
                        manifest,
                        "--config",
                        config,
                        "--output",
                        output / "preflight",
                        "--preflight",
                    ],
                    env,
                    output / "server.log",
                )
                if code:
                    state["status"] = "failed"
                    return code
                if args.mode == "train":
                    code = run_logged(
                        [
                            python,
                            "-m",
                            "chd_ct.server.training",
                            "--manifest",
                            manifest,
                            "--config",
                            config,
                            "--output",
                            output / "training",
                        ],
                        env,
                        output / "server.log",
                    )
                    if code:
                        state["status"] = "failed"
                        return code
            state["status"] = "checked" if args.mode == "check" else "passed"
            print(f"完成：{state['status']}；报告目录 {output}")
            return 0
    except KeyboardInterrupt as error:
        state.update(status="interrupted", error=str(error) or "用户中止")
        return interrupt_code
    except Exception as error:
        state.update(status="failed", error=str(error))
        print(f"未通过：{error}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if output:
            state["ended"] = datetime.now().isoformat()
            write_json(output / "status.json", state)
