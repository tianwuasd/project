"""Detached jobs with a runtime lock inherited across the parent/worker hand-off."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..recovery import archive_child, read_json
from .launcher import ROOT, make_parser, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].split()[0] in {"Z", "X"}:
            return None
        return {
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start": stat.rsplit(")", 1)[1].split()[19],
        }
    except (OSError, IndexError):
        return None


@contextmanager
def acquire_lock(runtime):
    import fcntl

    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / ".job.lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("此 runtime 已有任务运行；请用 --status 查看，不重复提交") from error
        yield lock
    finally:
        # Do not LOCK_UN: the detached worker shares this open file description.
        lock.close()


def initial_state(plan):
    return {
        "status": "queued",
        "name": plan.get("name") or Path(plan["directory"]).name,
        "submitted_at": now(),
        "directory": plan["directory"],
        "log": plan["log"],
        "steps": [
            dict(s["reuse"], status="reused")
            if s.get("reuse")
            else {
                "name": s["name"],
                "output": s["output"],
                "status": "pending",
                **({"cache_selection": s["cache_selection"]} if s.get("cache_selection") else {}),
            }
            for s in plan["steps"]
        ],
    }


def execute(plan):
    from . import gpu_hold, imagechd

    directory = Path(plan["directory"])
    state = initial_state(plan)
    state.update(status="running", pid=os.getpid(), identity=identity(os.getpid()), started_at=now())
    state_file = directory / "job_state.json"
    write_json(state_file, state)
    code = 1
    recent_gpus = set()
    handlers = {}
    hold_directory = None
    previous_hold = os.environ.pop(gpu_hold.ENV, None)

    def interrupt(sig, frame):
        raise KeyboardInterrupt(f"收到信号 {sig}")

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            handlers[sig] = signal.signal(sig, interrupt)
    try:
        hold_directory = gpu_hold.begin(plan)
        if hold_directory is not None:
            state["gpu_hold_directory"] = str(hold_directory)
        for index, step in enumerate(plan["steps"]):
            current = state["steps"][index]
            if step.get("reuse"):
                current.update(step["reuse"], status="reused")
                write_json(state_file, state)
                print("已保留完成步骤：" + step["name"], flush=True)
                continue
            if step.get("restart"):
                kept = archive_child(Path(step["output"]))
                if kept:
                    print("旧输出保留于：" + str(kept), flush=True)
            state["current_step"] = index
            current.update(status="running", started_at=now())
            write_json(state_file, state)
            args = make_parser().parse_args(["--non-interactive"])
            for key, value in step["arguments"].items():
                setattr(args, key, value)
            args.reset_settings = bool(plan.get("reset_settings")) and index == 0
            args.runtime = plan["runtime"]
            args.results = str(directory.parent)
            print(f"[{index + 1}/{len(plan['steps'])}] 开始 {step['name']} → {step['output']}", flush=True)
            code = imagechd.run(
                args,
                plan["profile"],
                output_dir=Path(step["output"]),
                lock_held=True,
                recent_gpus=recent_gpus,
            )
            from .cache_selection import fallback

            event = fallback(plan, index, code)
            if event:
                current["cache_selection"] = event
                write_json(state_file, state)
                args.prepared, args.resume = event["prepared"], False
                code = imagechd.run(
                    args,
                    plan["profile"],
                    output_dir=Path(step["output"]),
                    lock_held=True,
                    recent_gpus=recent_gpus,
                )
            if step["arguments"]["task"] == "preprocess" and code == 0:
                current["prepared"] = step["arguments"]["prepared"]
                print("预处理完成，后续使用缓存：" + current["prepared"], flush=True)
            if code == 0:
                report = Path(step["output"]) / "status.json"
                if report.is_file():
                    recent_gpus.update(
                        gpu["uuid"] for gpu in json.loads(report.read_text(encoding="utf-8")).get("gpus", [])
                    )
            if code == 0 and (Path(step["output"]) / "status.json").exists():
                from .resume import snapshot

                print("记录恢复校验信息：" + step["name"], flush=True)
                current["snapshot"] = snapshot(step)
            current.update(
                status="passed" if code == 0 else "interrupted" if code in {130, 143, 129} else "failed",
                exit_code=code,
                ended_at=now(),
            )
            write_json(state_file, state)
            if code:
                state["status"] = current["status"]
                print(f"流程停止：{step['name']} 退出码 {code}；后续步骤未运行。", flush=True)
                return code
        code = 0
        state["status"] = "passed"
        print("所选流程全部完成。", flush=True)
        return 0
    except KeyboardInterrupt as error:
        code = 130
        state.update(status="interrupted", error=str(error))
        return code
    except Exception as error:
        code = 1
        state.update(status="failed", error=str(error))
        print("流程失败：" + str(error), file=sys.stderr, flush=True)
        return code
    finally:
        cleanup_error = None
        try:
            gpu_hold.release(hold_directory)
        except BaseException as error:
            cleanup_error = error
            code = 1
            state.update(status="failed", error="GPU 保留清理失败：" + str(error))
        finally:
            os.environ.pop(gpu_hold.ENV, None)
            if previous_hold is not None:
                os.environ[gpu_hold.ENV] = previous_hold
        for step in state["steps"]:
            if step["status"] == "running":
                step.update(status=state["status"], exit_code=code, ended_at=now())
        state.update(exit_code=code, ended_at=now())
        write_json(state_file, state)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if cleanup_error is not None:
            raise cleanup_error


def submit(plan, lock, foreground=False):
    directory = Path(plan["directory"])
    resuming = bool(plan.get("resume_existing"))
    if resuming:
        from .resume import reject_live_commands

        reject_live_commands(plan)
        attempt = directory / "attempts" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        attempt.mkdir(parents=True, exist_ok=False)
        for name in ("plan.json", "job_state.json", "receipt.json"):
            if (directory / name).is_file():
                shutil.copy2(directory / name, attempt / name)
    else:
        directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", plan)
    state = initial_state(plan)
    write_json(directory / "job_state.json", state)
    runtime = Path(plan["runtime"])
    history_file = runtime / "job-history.json"
    records = read_json(history_file) if history_file.is_file() else []
    previous = runtime / "last-job.json"
    if previous.is_file():
        records.append(read_json(previous)["directory"])
    records.append(str(directory))
    write_json(history_file, list(dict.fromkeys(records)))
    write_json(runtime / "last-job.json", {"directory": str(directory)})
    command = [
        sys.executable,
        "-u",
        "-B",
        str(ROOT / "start_server.py"),
        "--worker",
        str(directory / "plan.json"),
        "--lock-fd",
        str(lock.fileno()),
    ]
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
    try:
        with Path(plan["log"]).open("a" if resuming else "w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
                cwd=ROOT,
                env=env,
            )
        write_json(directory / "receipt.json", {"pid": process.pid, "identity": identity(process.pid)})
    except Exception as error:
        state.update(status="failed", error=str(error), ended_at=now())
        write_json(directory / "job_state.json", state)
        raise
    print(f"已提交后台任务：{state['name']}，PID={process.pid}，尚不代表训练完成。", flush=True)
    print(
        f"日志：{plan['log']}\n状态：bash start_server.sh --status --runtime '{plan['runtime']}'", flush=True
    )
    print("可关闭 SSH；重新连接后查看状态或日志。", flush=True)
    if not foreground:
        return 0
    try:
        with Path(plan["log"]).open(encoding="utf-8", errors="replace") as reader:
            while process.poll() is None:
                print(reader.read(), end="", flush=True)
                time.sleep(0.2)
            print(reader.read(), end="", flush=True)
        return process.wait()
    except BaseException:
        from .bootstrap import stop_process_tree

        stop_process_tree(process)
        raise


def worker(plan_file, lock_fd):
    if sys.platform != "linux" or lock_fd is None:
        raise ValueError("后台 worker 只能由 Linux 启动器启动")
    import fcntl

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("不支持的任务计划版本")
    try:
        actual = os.fstat(lock_fd)
        expected = (Path(plan["runtime"]) / ".job.lock").stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("后台锁与本次 runtime 不匹配")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["CHD_RUNTIME_LOCK_FD"] = str(lock_fd)
        return execute(plan)
    finally:
        os.close(lock_fd)


def job_owner(directory, state=None):
    state = state if state is not None else read_json(directory / "job_state.json")
    receipt = directory / "receipt.json"
    candidates = [state, read_json(receipt) if receipt.is_file() else {}]
    for owner in candidates:
        if owner.get("pid"):
            current = identity(owner["pid"])
            if current and (not owner.get("identity") or current == owner["identity"]):
                return owner
    return next((owner for owner in candidates if owner.get("pid")), {})


def show_status(runtime, directory=None):
    pointer = runtime / "last-job.json"
    if directory is None and not pointer.is_file():
        print("尚未提交后台任务；先运行 bash start_server.sh。")
        return 0
    directory = (
        Path(directory)
        if directory is not None
        else Path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
    )
    state = json.loads((directory / "job_state.json").read_text(encoding="utf-8"))
    owner = job_owner(directory, state)
    status = state["status"]
    if status in {"queued", "running"} and owner.get("pid"):
        current = identity(owner["pid"])
        if current is None or (owner.get("identity") and current != owner["identity"]):
            status = "stale（原进程已退出或服务器已重启；保留最后记录，请查看日志）"
    details = {}
    if status == "running":
        for step in state["steps"]:
            if step["status"] != "running":
                continue
            report_path = Path(step["output"]) / "status.json"
            if not report_path.is_file():
                continue
            report = read_json(report_path)
            from .inspection import active_model_run, run_directory

            run = run_directory(Path(step["output"]))
            active_run = (
                active_model_run(run) if report.get("phase") in {"train", "preflight", "smoke"} else None
            )
            queue_path = active_run / "queue.json" if active_run else None
            queue = read_json(queue_path) if queue_path and queue_path.is_file() else {}
            if queue.get("sampled_at", 0) < report.get("phase_started_at", 0):
                queue = {}  # A resumed command has not published its new queue yet.
            if report.get("status") == "waiting_gpu":
                resources = report.get("gpu_resources", {})
                label = "waiting_gpu（等待显卡）"
                detail = f"可用 0/{resources.get('requested_gpus', '?')} 张卡，已等待 {resources.get('waited_seconds', 0)} 秒；{resources.get('reason', '')}"
            elif queue.get("status") in {"waiting_gpu", "running"}:
                label = "waiting_gpu（等待显卡）" if queue["status"] == "waiting_gpu" else "running"
                detail = f"正在使用 {queue.get('active_gpus', 0)}/{queue.get('gpu_limit', '?')} 张卡，待运行 {queue.get('pending_stages', '?')} 个阶段"
            else:
                continue
            if active_run is not None and active_run != run:
                detail = f"当前折：{active_run.name}；{detail}"
            elif active_run is not None and (run / "report.json").is_file():
                if read_json(run / "report.json").get("scheduling") == "cross_fold":
                    detail = "跨折共享队列；" + detail
            details[step["name"]] = (label, detail)
            if label.startswith("waiting_gpu"):
                status = label
    print(
        f"名称：{state.get('name') or directory.name}\n状态：{status}\n任务：{directory}\n日志：{state.get('log', directory / 'launcher.log')}"
    )
    if state.get("gpu_hold_directory"):
        from .gpu_hold import alive, records

        held = [r["gpu"] for r in records(Path(state["gpu_hold_directory"])) if alive(r)]
        if held:
            print(f"保留 {len(held)} 张 GPU 至整个所选流程结束：" + ", ".join(held))
    for step in state["steps"]:
        label = (
            "未知（主进程已退出，最后记录 running，子进程待核实）"
            if status.startswith("stale") and step["status"] == "running"
            else step["status"]
        )
        if step["name"] in details:
            label = details[step["name"]][0]
        print(f"  {step['name']}: {label} → {step['output']}")
        if step.get("cache_selection"):
            event = step["cache_selection"]
            print(f"    缓存自动新建：{event['reason']} → {event['prepared']}")
        elif step.get("prepared"):
            print("    预处理缓存：" + step["prepared"])
        if step["name"] in details:
            print("    " + details[step["name"]][1])
    if status.startswith("stale") or status in {"failed", "interrupted"}:
        print(f"恢复：bash start_server.sh --resume-job '{directory}'")
    if state.get("error"):
        print("错误：" + state["error"])
    return 0


def stop_job(runtime, directory=None):
    explicit_target = directory is not None
    pointer = runtime / "last-job.json"
    if directory is None and not pointer.is_file():
        raise ValueError("没有任务记录")
    directory = Path(directory) if directory is not None else Path(read_json(pointer)["directory"])
    if explicit_target:
        plan = read_json(directory / "plan.json")
        if Path(plan["runtime"]).resolve() != runtime.resolve():
            raise ValueError("停止目标与此 runtime 不符")
    owner = job_owner(directory)
    current = identity(owner["pid"]) if owner.get("pid") else None
    if current and owner.get("identity") == current:
        os.kill(owner["pid"], signal.SIGTERM)
        print("已请求停止；等待 --status 显示已退出，再恢复。完整检查点保留。")
        return 0
    from .resume import live_commands

    stopped = []
    for command in live_commands(read_json(directory / "plan.json")):
        try:
            if command["process_group"]:
                os.killpg(command["pid"], signal.SIGTERM)
            else:
                os.kill(command["pid"], signal.SIGTERM)
            stopped.append(command["pid"])
        except ProcessLookupError:
            pass
    print(
        "已请求停止存活子进程：" + str(stopped)
        if stopped
        else "原任务进程已退出；可以使用 --resume-job 恢复。"
    )
    return 0
