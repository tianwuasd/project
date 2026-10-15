"""Interactive server console: inspect safely, then explicitly start/resume/stop."""

import subprocess

from . import inspection, jobs
from .launcher import choose, personal_path


def pick_history(runtime, results, resume=False):
    records = inspection.show_history(runtime, results)
    if not records:
        return None
    value = input("选择任务编号，0 返回：").strip()
    if value in {"", "0"}:
        return None
    if not value.isdigit() or not 1 <= int(value) <= len(records):
        print("请输入列表内的任务编号。")
        return None
    job = records[int(value) - 1]
    if resume and (
        inspection.job_active(job) or inspection.load(job / "job_state.json").get("status") == "passed"
    ):
        print("此任务正在运行或已全部完成；可以查询，不能重复续跑。")
        return None
    return job


def logs(job):
    steps = inspection.load(job / "job_state.json").get("steps", [])
    print("0 总流程日志；a 当前步骤（自动跟随）")
    for i, step in enumerate(steps, 1):
        print(f"{i} {step['name']} | {step['status']}")
    value = input("日志步骤 [a]：").strip().lower() or "a"
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
        model = input("查看哪个模型 [all]：").strip() or "all"
        if model not in (*inspection.MODELS, "all", "server"):
            print("没有此模型。")
            return
    follow = input("持续刷新？y 持续 / n 看一次 [n]：").strip().lower() == "y"
    if follow:
        print("按 Ctrl+C 返回面板，后台任务继续运行。")
    inspection.tail_logs(lambda: inspection.log_paths(job, step, model), follow=follow)


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
        print("s 任务进度  g 显卡状态  m 模型 epoch/loss  l 详细日志  p 我的进程")
        print("h 历史任务与日志  0 退出面板（后台任务继续）")
        print("x 停止当前任务" if active else "n 新建并命名任务  r 选择历史任务续跑")
        try:
            action = input("请选择 [s]：").strip().lower() or "s"
            if action == "0":
                return False
            if action == "g":
                inspection.show_gpu()
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
            elif action == "x":
                if not active:
                    print("没有正在运行的任务。")
                    continue
                print(
                    f"将请求停止：{inspection.name(current)}\n{current}\n保留完整检查点；恢复时未完成的当前轮可能需要重跑。"
                )
                if input("确认停止此任务？[y/N]：").strip().lower() == "y":
                    if inspection.latest(runtime) != current:
                        raise ValueError("最近任务已改变，请重新选择停止目标。")
                    jobs.stop_job(runtime, directory=current)
            elif action in {"n", "r"}:
                if active:
                    print("此 runtime 已有任务在运行；请先查询或停止，不重复提交。")
                    continue
                if action == "r":
                    selected = pick_history(runtime, results, resume=True)
                    if selected is None:
                        continue
                    args.resume_job = str(selected)
                return True
            else:
                print("请输入菜单中的字母或 0。")
        except KeyboardInterrupt:
            print("\n已返回面板，后台任务继续运行。输入 0 退出。")
        except EOFError:
            return False
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print("查询/操作未完成：" + str(error))
