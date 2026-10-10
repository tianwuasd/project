"""zmic44 defaults and allocation-aware GPU selection."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME = "/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime"
DEFAULT_RESULTS = "/data_nas/zhangruichen/chd_ct_results"
SERVER_PROFILE = ROOT / "configs/servers/zmic44.json"


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
        timeout=15,
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
        timeout=15,
    )
    busy = {line.strip() for line in processes.stdout.splitlines()}
    for row in rows:
        row["busy"] = row["uuid"] in busy
    return rows


def select_gpu(rows, requested="auto", visible=None, minimum=6000):
    return select_gpus(rows, 1, requested, visible, minimum)[0]


def select_gpus(rows, count=1, requested="auto", visible=None, minimum=6000):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
        raise ValueError("显卡数量需为 1 至 5：当前最多五个独立训练阶段同时运行")
    allowed = None if visible is None else {s.strip() for s in visible.split(",") if s.strip()}
    choices = None if requested == "auto" else [s.strip() for s in requested.split(",")]
    if choices is not None and (len(choices) != count or any(not s for s in choices)):
        raise ValueError("GPU 编号/UUID 的个数必须与 --gpu-count 一致")
    eligible = []
    for row in rows:
        if allowed is not None and not ({row["index"], row["uuid"]} & allowed):
            continue
        if not row["busy"] and row["free"] >= minimum and row["util"] <= 10:
            eligible.append(row)
    if choices is None:
        selected = sorted(eligible, key=lambda row: row["free"], reverse=True)[:count]
    else:
        selected = []
        for choice in choices:
            matches = [r for r in eligible if choice in (r["index"], r["uuid"])]
            if len(matches) != 1:
                raise ValueError("指定/分配范围内没有满足条件的空闲 GPU：" + choice)
            selected.append(matches[0])
    if len(selected) != count:
        raise ValueError("指定/分配范围内没有足够空闲 GPU；不会越过 CUDA_VISIBLE_DEVICES 或自动退回 CPU")
    if len({r["uuid"] for r in selected}) != count:
        raise ValueError("不能重复选择同一张 GPU")
    return selected


def make_parser():
    from ..quickstart.commands import add_task_arguments

    parser = argparse.ArgumentParser(description="ImageCHD 主流程服务器入口")
    add_task_arguments(parser)
    parser.set_defaults(device="cuda", mode=None)
    parser.add_argument(
        "--reset-settings", action="store_true", help="忽略上次选择，重新采用服务器配置默认值"
    )
    parser.add_argument("--runtime")
    parser.add_argument("--results")
    parser.add_argument("--gpu", help="auto 或逗号分隔的物理编号/完整 UUID")
    parser.add_argument("--gpu-count", type=int, help="训练显卡数，默认沿用设置或 1（最多 5）")
    parser.add_argument("--python", help="已有 Python 路径；不安装依赖，由环境检查决定是否可用")
    parser.add_argument("--server-config", default=str(SERVER_PROFILE))
    parser.add_argument(
        "--tasks", help="多选步骤，逗号或空格分隔；例如 environment,preprocess,smoke,train,test"
    )
    parser.add_argument("--foreground", action="store_true", help="前台调试并等待完成；默认独立后台运行")
    parser.add_argument("--status", action="store_true", help="查看最近后台任务，不安装环境或查询GPU")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        from . import jobs

        if args.worker:
            return jobs.worker(Path(args.worker), args.lock_fd)
        profile = json.loads(Path(args.server_config).read_text(encoding="utf-8"))
        if args.status:
            return jobs.show_status(personal_path(args.runtime or profile["runtime"]))
        if sys.platform != "linux":
            raise ValueError("服务器入口用于 Linux；本机请使用 start.py 或独立 Python 脚本。")
        from .workflow import build_plan

        plan = build_plan(args, profile, sys.stdin.isatty() and not args.non_interactive)
        print("执行顺序：" + " → ".join(step["name"] for step in plan["steps"]))
        for step in plan["steps"]:
            print(step["name"] + "：" + json.dumps(step["arguments"], ensure_ascii=False))
        print("本次日志：" + plan["log"], flush=True)
        with jobs.acquire_lock(Path(plan["runtime"])) as lock:
            return jobs.submit(plan, lock, foreground=args.foreground)
    except (Exception, KeyboardInterrupt) as error:
        print("未通过：" + str(error), file=sys.stderr)
        return 1
