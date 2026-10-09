"""zmic44 defaults, one GPU per job, existing quickstart validation and smoke."""

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
    from ..quickstart.commands import add_task_arguments, choose_task

    parser = argparse.ArgumentParser(description="ImageCHD 主流程服务器入口")
    add_task_arguments(parser)
    parser.set_defaults(device="cuda")
    parser.add_argument("--runtime")
    parser.add_argument("--results")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--python", help="已有 Python 路径；不安装依赖，由环境检查决定是否可用")
    parser.add_argument("--server-config", default=str(SERVER_PROFILE))
    args = parser.parse_args(argv)
    try:
        args.task = choose_task(args, sys.stdin.isatty() and not args.non_interactive)
        profile = json.loads(Path(args.server_config).read_text(encoding="utf-8"))
        from .imagechd import run

        return run(args, profile)
    except (Exception, KeyboardInterrupt) as error:
        print("未通过：" + str(error), file=sys.stderr)
        return 1
