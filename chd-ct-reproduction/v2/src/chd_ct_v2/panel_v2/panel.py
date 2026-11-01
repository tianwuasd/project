"""Chinese V2 terminal entry; all task operations are registry scoped."""

import argparse
import json
import sys
from pathlib import Path

from ..server.launcher import ROOT, SERVER_PROFILE, make_parser
from ..server.prompts import PanelBack, UserExit, ask
from . import catalog, service
from .identity import BRANCH, check, claim, locations


def parser():
    p = argparse.ArgumentParser(description="CT V2 独立服务器面板；科学流程 CT 0.24")
    p.add_argument("--server-config", default=str(SERVER_PROFILE))
    for key in ("runtime", "results", "prepared-root", "name", "input-job", "job"):
        p.add_argument("--" + key)
    p.add_argument("--stage", choices=list(service.STAGES))
    p.add_argument("--query", choices=("status", "gpu", "models", "logs", "processes", "history", "results"))
    p.add_argument(
        "--action",
        choices=(
            "resume",
            "rerun",
            "stop",
            "trash",
            "restore",
            "diagnostics",
            "network",
            "update",
            "push-code",
            "upload",
        ),
    )
    p.add_argument(
        "--result-section",
        choices=("all", "segmentation", "decisions", "diagnosis-evaluation", "training", "files"),
        default="all",
    )
    p.add_argument("--result-detail", choices=("summary", "detailed"), default="summary")
    p.add_argument("--step", default="auto")
    p.add_argument("--model", default="all")
    p.add_argument("--lines", type=int, default=60)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--yes", action="store_true", help="确认当前明确指定的提交/停止/回收/上传")
    p.add_argument("--preview", action="store_true", help="冻结并打印计划，不启动任务")
    for key in ("dataset", "config", "split-file", "spacing-excel", "diagnosis-source", "gpu"):
        p.add_argument("--" + key)
    p.add_argument("--gpu-count", type=int)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--limit", type=int)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return p


def select(values, stages=None, trash=False):
    rows = [
        r
        for r in catalog.list_jobs(values, include_trash=trash)
        if bool(r.get("trashed")) == trash and (not stages or r["stage"] in stages)
    ]
    if not rows:
        raise ValueError("本V2没有符合条件的登记任务")
    for i, row in enumerate(rows, 1):
        print(f"{i}  {row['name']} | {row['stage']} | {row['status']}\n   {row['id']}")
    answer = ask("选择编号（b 返回）：")
    if not answer.isdigit() or not 1 <= int(answer) <= len(rows):
        raise ValueError("请输入列表中的编号")
    return rows[int(answer) - 1]["id"]


def query(values, args, profile):
    check(values)
    if args.query == "history":
        for row in catalog.list_jobs(values, include_trash=True):
            print(
                f"{row['id']} | {row['name']} | {row['stage']} | {row['status']} | 回收={bool(row.get('trashed'))}"
            )
        return 0
    from ..server.inspection import query as show

    q = make_parser().parse_args([])
    for key in ("query", "step", "model", "lines", "follow", "result_section", "result_detail"):
        setattr(q, key, getattr(args, key))
    q.runtime, q.results = values["runtime"], values["results"]
    if args.query not in ("gpu", "processes"):
        record, _ = catalog.get(values, args.job)
        q.job_dir = record["directory"]
    return show(q, profile)


def confirm(args, message):
    return args.yes or (sys.stdin.isatty() and ask(message + " [y/N]：").lower() == "y")


def action(values, args, profile):
    from ..server import code_sync, jobs, network, updater, upload

    if args.action == "network":
        claim(values)
        network.configure_interactive(locations(values)["runtime"])
        return 0
    if args.action in ("update", "push-code"):
        claim(values)
        if args.action == "push-code":
            q = make_parser().parse_args([])
            q.runtime, q.yes_upload = values["runtime"], args.yes
            return code_sync.run(q, profile, sys.stdin.isatty())
        repo = updater.inspect_repository(ROOT)
        print(f"更新限定分支：{repo['branch']}；当前提交 {repo['head']}")
        if not confirm(args, "备份后更新整个CHD项目"):
            return 0
        with jobs.acquire_lock(locations(values)["runtime"]):
            updater.update(repo, locations(values)["runtime"])
        print("请退出后重新打开V2面板；冻结任务仍使用原代码。")
        return 0
    record = catalog.resolve_entry(values, args.job, allow_trash=args.action == "restore")
    key = record["id"]
    if args.action == "diagnostics":
        print("本地诊断：" + str(service.diagnostics(values, key)))
        return 0
    if not confirm(args, f"{args.action}：{record['name']} ({key})"):
        return 0
    if args.action == "resume":
        return service.resume(values, key)
    if args.action == "rerun":
        return service.rerun(values, profile, key)
    if args.action in ("trash", "restore"):
        catalog.recycle(values, key, restore=args.action == "restore")
        return 0
    if args.action == "stop":
        catalog.get(values, key)
        return jobs.stop_job(locations(values)["runtime"], Path(record["directory"]))
    if args.action == "upload":
        catalog.get(values, key)
        with jobs.acquire_lock(locations(values)["runtime"]), network.use(network.load(values["runtime"])):
            package = upload.prepare_upload(Path(record["directory"]), locations(values)["runtime"])
            upload.show_preview(package)
            if confirm(args, "将以上日志、指标与模型包上传独立产物分支"):
                print(upload.upload_package(package))
        return 0
    raise ValueError("未知操作")


def new_task(values, args, profile, interactive=False):
    options = {k: getattr(args, k) for k in service.RAW_OPTIONS if getattr(args, k, None) is not None}
    name = args.name or ("CT V2 " + args.stage)
    if interactive:
        name = ask(f"任务名称 [{name}]：") or name
        if args.stage in service.SOURCE:
            args.input_job = select(values, service.SOURCE[args.stage])
        if args.stage in ("prepare", "assessment"):
            options["dataset"] = ask(f"原始数据目录 [{profile['dataset']}]：") or profile["dataset"]
        if args.stage in ("train", "predict", "predict-all", "assessment"):
            options["gpu"] = ask("获准 GPU 编号/UUID [auto]：") or "auto"
            options["gpu_count"] = int(ask("最多同时使用几张 GPU [1]：") or "1")
        if args.stage == "diagnosis":
            options["diagnosis_source"] = ask("诊断表 XLSX/CSV：")
    print(f"阶段：{args.stage}；名称：{name}；来源：{args.input_job or '独立'}")
    print(f"结果根目录：{values['results']}；V2独立环境：{values['runtime']}")
    if args.preview:
        print(
            json.dumps(
                service.build(values, profile, args.stage, name, args.input_job, options),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not confirm(args, "创建并后台运行此V2任务"):
        return 0
    return service.submit(values, profile, args.stage, name, args.input_job, options)


def menu(values, profile):
    selected = None
    while True:
        print("\nCT V2 独立任务面板 | CT算法0.24 | 正式发布 " + BRANCH)
        print("1 预处理  2 训练  3 测试预测/评估  4 全病例预测  5 诊断  6 重算评估  a 独立小规模测评  e 环境")
        print("s 进度  g 显卡  m 模型  l 日志  p 我的进程  o 分区结果  h 历史/选择")
        print("r 恢复  k 独立重跑  t 停止  d 回收/恢复  z 本地诊断")
        print("c 网络  u 更新项目  w 上传项目代码  v 上传产物  0/q 退出（后台继续）")
        if selected:
            print("当前查看：" + selected)
        try:
            key = ask("请选择 [s]：").lower() or "s"
            if key in ("0", "q"):
                return 0
            args = parser().parse_args([])
            args.job = selected
            stages = {
                "1": "prepare",
                "2": "train",
                "3": "predict",
                "4": "predict-all",
                "5": "diagnosis",
                "6": "evaluate",
                "a": "assessment",
                "e": "environment",
            }
            queries = {
                "s": "status",
                "g": "gpu",
                "m": "models",
                "l": "logs",
                "p": "processes",
                "o": "results",
            }
            actions = {
                "r": "resume",
                "k": "rerun",
                "t": "stop",
                "z": "diagnostics",
                "c": "network",
                "u": "update",
                "w": "push-code",
                "v": "upload",
            }
            if key in stages:
                args.stage = stages[key]
                new_task(values, args, profile, True)
                selected = catalog.read(values).get("selected")
            elif key == "h":
                selected = select(values)
            elif key in queries:
                args.query = queries[key]
                if key == "o":
                    args.job = select(values)
                    sections = {
                        "1": "segmentation",
                        "2": "training",
                        "3": "decisions",
                        "4": "diagnosis-evaluation",
                        "5": "files",
                    }
                    args.result_section = sections.get(
                        ask("1 分割 2 训练 3 决策 4 诊断评估 5 文件 [1]：") or "1", "segmentation"
                    )
                    args.result_detail = "detailed" if ask("1 简洁 2 详细 [1]：") == "2" else "summary"
                if key in ("g", "l"):
                    args.follow = True
                query(values, args, profile)
            elif key == "d":
                restore = ask("1 回收 2 恢复 [1]：") == "2"
                args.action = "restore" if restore else "trash"
                args.job = select(values, trash=restore)
                action(values, args, profile)
            elif key in actions:
                args.action = actions[key]
                if key in ("r", "k", "t", "v", "z"):
                    args.job = select(values)
                action(values, args, profile)
                if key == "u":
                    return 0
            else:
                print("请输入菜单中的选项")
        except PanelBack:
            continue
        except (UserExit, EOFError):
            return 0
        except KeyboardInterrupt:
            print("\n已退出本次查看/操作；后台任务状态请用 s 核对。")
        except (ValueError, OSError, RuntimeError) as error:
            print("未执行/未完成：" + str(error))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.worker:
            from ..server.jobs import worker

            return worker(Path(args.worker), args.lock_fd)
        profile = json.loads(Path(args.server_config).read_text(encoding="utf-8"))
        values = {
            "runtime": args.runtime or profile["runtime"],
            "results": args.results or profile["results"],
            "prepared": args.prepared_root or profile["prepared"],
        }
        values = {k: str(v) for k, v in locations(values).items()}
        check(values)
        if sum(bool(x) for x in (args.stage, args.query, args.action)) > 1:
            raise ValueError("一次只能提交、查询或操作其中一种")
        if args.stage:
            return new_task(values, args, profile)
        if args.query:
            return query(values, args, profile)
        if args.action:
            return action(values, args, profile)
        if not sys.stdin.isatty():
            raise ValueError("非交互运行请指定 --stage、--query 或 --action；--help 查看参数")
        return menu(values, profile)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print("未完成：" + str(error), file=sys.stderr)
        return 1
