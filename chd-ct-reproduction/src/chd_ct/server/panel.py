"""Interactive server console: inspect safely, then explicitly start/resume/stop."""

import copy
import subprocess

from . import inspection, jobs
from .launcher import choose, personal_path
from .prompts import PanelBack, UserExit, ask


def pick_history(runtime, results, resume=False):
    records = inspection.show_history(runtime, results)
    if not records:
        return None
    value = ask("选择任务编号，0 返回：")
    if value in {"", "0"}:
        return None
    if not value.isdigit() or not 1 <= int(value) <= len(records):
        print("请输入列表内的任务编号。")
        return None
    job = records[int(value) - 1]
    if inspection.status(job).startswith("unavailable"):
        print("此任务记录损坏；请选择其他任务或核对原文件。")
        return None
    if resume and inspection.job_active(job):
        print("此任务正在运行；可以查询，不能重复续跑。")
        return None
    return job


def logs(job):
    steps = inspection.load(job / "job_state.json").get("steps", [])
    print("0 总流程日志；a 当前步骤（自动跟随）")
    for i, step in enumerate(steps, 1):
        print(f"{i} {step['name']} | {step['status']}")
    value = ask("日志步骤 [a]：").lower() or "a"
    if value == "0":
        step = "launcher"
    elif value == "a":
        step = "auto"
    elif value.isdigit() and 1 <= int(value) <= len(steps):
        step = steps[int(value) - 1]["name"]
    else:
        print("没有此步骤。")
        return
    model = "all"
    if step != "launcher" and (step == "auto" or step in inspection.MODEL_STEPS):
        print("模型：all 全部 / server 调度日志 / " + " / ".join(inspection.MODELS))
        model = ask("查看哪个模型 [all]：") or "all"
        if model not in (*inspection.MODELS, "all", "server"):
            print("没有此模型。")
            return
    follow = ask("持续刷新？y 持续 / n 看一次 [n]：").lower() == "y"
    if follow:
        print("按 Ctrl+C 返回面板，后台任务继续运行。")
    from .log_progress import show_progress

    inspection.tail_logs(
        lambda: inspection.log_paths(job, step, model),
        follow=follow,
        progress=lambda: show_progress(job, step, model),
    )


def results_menu(runtime, results):
    from . import results as viewer

    records = viewer.result_history(runtime, results)
    print("\n选择项目/任务查看结果（新到旧，旧版本已完成任务也可查看）：")
    for index, job in enumerate(records, 1):
        saved = viewer.read(job / "job_state.json")
        print(
            f"  {index}. {viewer.title(job)} | {viewer.state(saved.get('status'))}\n     {viewer.text(job)}"
        )
    print("d 手动输入历史 workflow 目录；0 返回面板。")
    current = inspection.latest(runtime)
    default = str(records.index(current) + 1) if current in records else "1" if records else "d"
    value = ask(f"项目/任务编号 [{default}]：").lower() or default
    if value == "0":
        return
    if value == "d":
        from pathlib import Path

        entered = ask("历史 workflow 完整目录：").strip().strip("\"'")
        if not entered:
            return
        job = Path(entered).expanduser().resolve()
    elif value.isdigit() and 1 <= int(value) <= len(records):
        job = records[int(value) - 1]
    else:
        print("请输入列表内的编号，或 d 输入目录。")
        return
    from .result_menu import browse_results

    browse_results(job)


def run(args, profile):
    runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", True))
    args.runtime = str(runtime)
    results = args.results or profile["results"]
    while True:
        current = inspection.latest(runtime)
        active = inspection.job_active(current)
        print("\n服务器任务面板")
        if current:
            print(
                f"最近任务：{inspection.name(current)}\n状态：{inspection.status(current)}\n目录：{current}"
            )
        else:
            print("尚无任务记录。")
        print("s 任务进度  g 显卡状态（1秒刷新）  m 模型 epoch/loss  l 详细日志  p 我的进程")
        print(
            "o 查看结果（选择项目/任务）  h 历史任务与日志  u 更新程序（多镜像）  0/q 退出面板（后台任务继续）"
        )
        print("v 上传日志、配置、指标和模型权重（GitHub Release）")
        print(
            "x 中断当前任务（保留检查点）"
            if active
            else "n 新建任务（可选输入任务）  r 续跑/追加  k 已完成任务重跑（独立新结果）"
        )
        print("普通输入页 b 返回面板；结果页 b/0 返回上一级；q 退出程序。")
        try:
            action = ask("请选择 [s]：").lower() or "s"
            if action == "0":
                return False
            if action == "g":
                inspection.watch_gpu()
            elif action == "p":
                inspection.show_processes()
            elif action in {"s", "m", "l"}:
                if current is None:
                    print("没有任务记录。")
                elif action == "s":
                    jobs.show_status(runtime, current)
                elif action == "m":
                    inspection.show_models(current)
                else:
                    logs(current)
            elif action == "h":
                selected = pick_history(runtime, results)
                if selected:
                    jobs.show_status(runtime, selected)
                    logs(selected)
            elif action == "o":
                results_menu(runtime, results)
            elif action == "v":
                from .sharing import run_upload

                selected = pick_history(runtime, results)
                if selected:
                    sharing_args = copy.deepcopy(args)
                    sharing_args.upload_job = str(selected)
                    sharing_args.upload_package = None
                    sharing_args.yes_upload = False
                    run_upload(sharing_args, profile, True)
            elif action == "x":
                if not active:
                    print("没有正在运行的任务。")
                    continue
                print(
                    f"将请求停止：{inspection.name(current)}\n{current}\n保留完整检查点；恢复时未完成的当前轮可能需要重跑。"
                )
                if ask("确认中断此任务？[y/N]：").lower() == "y":
                    if inspection.latest(runtime) != current:
                        raise ValueError("最近任务已改变，请重新选择停止目标。")
                    jobs.stop_job(runtime, directory=current)
            elif action == "u":
                from . import updater

                # Reject a dirty/wrong checkout before interrupting any job.
                repo = updater.inspect_repository()
                print(f"更新仓库：{repo['root']}\n当前版本：{repo['head'][:12]}")
                if active:
                    print(f"将先中断：{inspection.name(current)}\n{current}\n保留检查点，确认退出后更新。")
                    if ask("中断此任务并更新程序？[y/N]：").lower() != "y":
                        continue
                    if inspection.latest(runtime) != current:
                        raise ValueError("最近任务已改变，请重新选择。")
                    jobs.stop_job(runtime, directory=current)
                elif ask("备份当前代码并更新程序？[y/N]：").lower() != "y":
                    continue
                with updater.idle_lock(runtime, current):
                    try:
                        changed = updater.update(repo, runtime)
                    except (Exception, KeyboardInterrupt) as error:
                        print("更新未完成：" + (str(error) or "用户中断"))
                        print("本面板已退出。请检查更新记录后重新运行 bash start_server.sh。")
                        return False
                if changed:
                    print("更新完成。本面板已退出，请重新运行 bash start_server.sh 加载新版。")
                    return False
            elif action in {"n", "r", "k"}:
                if active:
                    print("此 runtime 已有任务在运行；请先查询或停止，不重复提交。")
                    continue
                if action in {"r", "k"}:
                    selected = pick_history(runtime, results, resume=True)
                    if selected is None:
                        continue
                    if action == "k":
                        args.rerun_job = str(selected)
                    else:
                        args.resume_job = str(selected)
                return True
            else:
                print("请输入菜单中的字母或 0。")
        except (KeyboardInterrupt, PanelBack):
            print("\n已返回面板，后台任务继续运行。输入 0 退出。")
        except (EOFError, UserExit):
            return False
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print("查询/操作未完成：" + str(error))
