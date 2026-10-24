"""zmic44 defaults and allocation-aware GPU selection."""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from .prompts import PanelBack, UserExit, ask

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
    value = ask(f"{label} [{suggestion}]：")
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
    from .gpu_hold import owned_processes

    held_processes = owned_processes()
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
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid" + (",pid" if held_processes else ""),
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    busy = set()
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        if held_processes:
            gpu, pid = [s.strip() for s in line.split(",")]
            if pid.isdigit() and held_processes.get(int(pid)) == gpu:
                continue
            busy.add(gpu)
        else:
            busy.add(line.strip())
    for row in rows:
        row["busy"] = row["uuid"] in busy
    return rows


def show_gpus():
    print("\n当前显卡状态（此刻快照，空闲不代表已预约）：")
    try:
        rows = query_gpus()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = None if visible is None else {x.strip() for x in visible.split(",")}
        print("编号  型号                     空闲显存(MiB)  利用率  计算进程  可见范围")
        for row in rows:
            access = allowed is None or bool({row["index"], row["uuid"]} & allowed)
            print(
                f"{row['index']:>3}   {row['name']:<24} {row['free']:>8}       {row['util']:>3}%    "
                f"{'占用' if row['busy'] else '无'}       {'范围内' if access else '范围外'}"
            )
        print("只选择本次获准使用的显卡；后台执行前还会重新检查。\n")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print("无法取得显卡状态：" + str(error) + "；GPU 任务执行前必须检查通过。")


def select_gpu(rows, requested="auto", visible=None, minimum=6000):
    return select_gpus(rows, 1, requested, visible, minimum)[0]


def gpu_pool(rows, requested="auto", visible=None):
    """Resolve an allowed pool independently of current occupancy or target count."""
    allowed = None if visible is None else {s.strip() for s in visible.split(",") if s.strip()}
    candidates = [r for r in rows if allowed is None or {r["index"], r["uuid"]} & allowed]
    if requested != "auto":
        selected = []
        for choice in requested.split(","):
            matches = [r for r in candidates if choice.strip() in (r["index"], r["uuid"])]
            if len(matches) != 1:
                raise ValueError("GPU 不存在或不在获准可见范围内：" + choice)
            selected.append(matches[0])
        candidates = selected
    if not candidates:
        raise ValueError("获准可见范围内没有 GPU；请检查 GPU 编号和 CUDA_VISIBLE_DEVICES")
    if len({r["uuid"] for r in candidates}) != len(candidates):
        raise ValueError("不能重复选择同一张 GPU")
    return candidates


def idle_gpus(rows, pool, minimum=6000):
    allowed = set(pool)
    return sorted(
        [
            r
            for r in rows
            if r["uuid"] in allowed and not r["busy"] and r["free"] >= minimum and r["util"] <= 10
        ],
        key=lambda r: r["free"],
        reverse=True,
    )


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
    parser.add_argument("--name", help="本次新任务的显示名称；重名也使用独立目录，续跑保留原名称")
    parser.add_argument("--input-job", help="新建任务复用指定 workflow 的预处理结果；none 不复用并新建预处理")
    parser.add_argument("--rerun-job", help="选择已完成 workflow，在独立新任务中重新计算")
    parser.add_argument("--rerun-tasks", help="配合 --rerun-job 选择原任务步骤，逗号分隔；默认 all")
    parser.add_argument(
        "--preprocess-cache",
        choices=("reuse", "new"),
        help="预处理缓存：reuse 尝试沿用、失败自动新建（默认）；new 使用独立新目录",
    )
    parser.add_argument(
        "--query",
        choices=("status", "gpu", "models", "logs", "processes", "history", "results"),
        help="只读查询，不安装环境或启动任务",
    )
    parser.add_argument("--job-dir", help="查询指定历史 workflow 目录；默认最近任务")
    parser.add_argument(
        "--result-section",
        choices=("all", "segmentation", "decisions", "diagnosis-evaluation", "training", "files"),
        help="结果分区：默认 all；仅配合 --query results",
    )
    parser.add_argument(
        "--result-detail", choices=("summary", "detailed"), help="结果级别：默认 summary 简洁；detailed 详细"
    )
    parser.add_argument(
        "--step", default="auto", help="查询步骤名称，如 train/smoke/preflight；launcher 为总日志"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=("all", "server", "crop64", "crop128", "all64", "all128", "blood2d", "blood_lstm"),
        help="日志模型；server 为步骤调度日志",
    )
    parser.add_argument("--follow", action="store_true", help="持续刷新查询；Ctrl+C 只退出查看")
    parser.add_argument("--lines", type=int, default=60, help="各日志首次显示末尾多少行（1—1000）")
    parser.add_argument("--gpu", help="允许使用的候选卡：auto 或逗号分隔的物理编号/完整 UUID")
    parser.add_argument(
        "--gpu-count", type=int, help="训练最多并行使用的卡数（1—5）；不足时少卡运行，无空闲卡时等待"
    )
    parser.add_argument("--python", help="已有 Python 路径；不安装依赖，由环境检查决定是否可用")
    parser.add_argument("--server-config", default=str(SERVER_PROFILE))
    parser.add_argument(
        "--tasks", help="多选步骤，逗号或空格分隔；例如 environment,preprocess,smoke,train,test"
    )
    parser.add_argument("--foreground", action="store_true", help="前台调试并等待完成；默认独立后台运行")
    parser.add_argument("--status", action="store_true", help="查看最近后台任务，不安装环境或查询GPU")
    parser.add_argument("--resume-job", nargs="?", const="latest", help="恢复最近任务，或指定 workflow 目录")
    parser.add_argument(
        "--resume-check",
        choices=["full", "simple", "none"],
        help="续跑前检查：full 全面 / simple 简单（默认）/ none 跳过产物校验",
    )
    parser.add_argument("--append-tasks", help="配合 --resume-job 追加原计划没有的步骤，成功步骤不会重跑")
    publication = parser.add_mutually_exclusive_group()
    publication.add_argument(
        "--upload-job", nargs="?", const="latest", help="准备任务日志、配置、指标和权重上传包"
    )
    publication.add_argument("--upload-package", help="上传或重试已准备的包目录")
    parser.add_argument("--yes-upload", action="store_true", help="确认发布所选上传包至 GitHub Release")
    parser.add_argument("--stop", action="store_true", help="请求中断最近任务，保留已提交的缓存和检查点")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        from . import jobs

        if args.rerun_tasks and not args.rerun_job:
            raise ValueError("--rerun-tasks 需要配合 --rerun-job")
        if args.rerun_job and (args.resume_job or args.input_job or args.task or args.tasks):
            raise ValueError("重跑不能与续跑、新建输入或 --task/--tasks 混用")
        if args.input_job and args.resume_job:
            raise ValueError("续跑保留原输入任务，不能更换")
        if (args.input_job or args.rerun_job) and (
            args.query or args.status or args.stop or args.worker or args.upload_job or args.upload_package
        ):
            raise ValueError("输入任务/重跑参数不能与查询、停止、worker 或上传混用")
        if args.append_tasks and not args.resume_job:
            raise ValueError("--append-tasks 需要配合 --resume-job")
        if args.resume_check and not args.resume_job:
            raise ValueError("--resume-check 需要配合 --resume-job")
        publishing = args.upload_job or args.upload_package
        if args.yes_upload and not publishing:
            raise ValueError("--yes-upload 需要配合 --upload-job 或 --upload-package")
        if publishing and (
            args.task
            or args.tasks
            or args.resume_job
            or args.query
            or args.worker
            or args.stop
            or args.status
            or args.name
            or args.preprocess_cache
            or args.reset_settings
        ):
            raise ValueError("上传不能与启动、续跑、查询、停止或修改设置混用")
        if args.query and (
            args.task
            or args.tasks
            or args.resume_job
            or args.stop
            or args.worker
            or args.name
            or args.status
            or args.preprocess_cache
        ):
            raise ValueError("--query 是只读查询，不能同时指定启动、续跑、停止或 --status")
        if not 1 <= args.lines <= 1000:
            raise ValueError("--lines 需要在 1—1000 之间")
        if (args.result_section or args.result_detail) and args.query != "results":
            raise ValueError("--result-section/--result-detail 需要配合 --query results")
        if not args.query and (args.job_dir or args.follow or args.step != "auto" or args.model != "all"):
            raise ValueError("--job-dir/--follow/--step/--model 需要配合 --query")
        if args.worker:
            return jobs.worker(Path(args.worker), args.lock_fd)
        profile = json.loads(Path(args.server_config).read_text(encoding="utf-8"))
        if publishing:
            from .sharing import run_upload

            return run_upload(args, profile, sys.stdin.isatty() and not args.non_interactive)
        if args.query:
            from .inspection import query

            return query(args, profile)
        if args.stop:
            return jobs.stop_job(personal_path(args.runtime or profile["runtime"]))
        if args.status:
            return jobs.show_status(personal_path(args.runtime or profile["runtime"]))
        if sys.platform != "linux":
            raise ValueError("服务器入口用于 Linux；本机请使用 start.py 或独立 Python 脚本。")
        from .workflow import build_plan

        interactive = sys.stdin.isatty() and not args.non_interactive
        use_panel = interactive and not (args.task or args.tasks or args.resume_job or args.rerun_job)
        original = copy.deepcopy(args)
        while True:
            try:
                if use_panel:
                    from .panel import run

                    args = copy.deepcopy(original)
                    if not run(args, profile):
                        return 0
                    original.runtime = args.runtime
                if interactive:
                    print("输入 b 返回面板，q 退出程序；Ctrl+C 取消当前输入。")
                if args.rerun_job:
                    from .rerun import build_rerun_plan

                    plan = build_rerun_plan(args, profile, interactive)
                elif args.resume_job:
                    from .resume import build_resume_plan

                    plan = build_resume_plan(args, profile, interactive)
                else:
                    plan = build_plan(args, profile, interactive)
                break
            except (UserExit, EOFError):
                print("已退出，后台任务继续运行。")
                return 0
            except (PanelBack, KeyboardInterrupt):
                if not interactive:
                    raise
                print("已取消当前输入，返回面板。")
                use_panel = True
                original.task = original.tasks = original.resume_job = None
                original.append_tasks = None
                original.input_job = original.rerun_job = original.rerun_tasks = None
            except (ValueError, OSError) as error:
                if not use_panel:
                    raise
                print("未通过：" + str(error) + "；已返回面板。")
        print("执行顺序：" + " → ".join(step["name"] for step in plan["steps"]))
        for step in plan["steps"]:
            print(step["name"] + "：" + json.dumps(step["arguments"], ensure_ascii=False))
        print("本次日志：" + plan["log"], flush=True)
        with jobs.acquire_lock(Path(plan["runtime"])) as lock:
            return jobs.submit(plan, lock, foreground=args.foreground)
    except (UserExit, EOFError):
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print("未通过：" + str(error), file=sys.stderr)
        return 1
