"""Detached jobs with a runtime lock inherited across the parent/worker hand-off."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

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
        "submitted_at": now(),
        "directory": plan["directory"],
        "log": plan["log"],
        "steps": [{"name": s["name"], "output": s["output"], "status": "pending"} for s in plan["steps"]],
    }


def execute(plan):
    from . import imagechd

    directory = Path(plan["directory"])
    state = initial_state(plan)
    state.update(status="running", pid=os.getpid(), identity=identity(os.getpid()), started_at=now())
    state_file = directory / "job_state.json"
    write_json(state_file, state)
    code = 1
    recent_gpus = set()
    handlers = {}

    def interrupt(sig, frame):
        raise KeyboardInterrupt(f"收到信号 {sig}")

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            handlers[sig] = signal.signal(sig, interrupt)
    try:
        for index, step in enumerate(plan["steps"]):
            current = state["steps"][index]
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
            if code == 0:
                report = Path(step["output"]) / "status.json"
                if report.is_file():
                    recent_gpus.update(
                        gpu["uuid"] for gpu in json.loads(report.read_text(encoding="utf-8")).get("gpus", [])
                    )
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
        for step in state["steps"]:
            if step["status"] == "running":
                step.update(status=state["status"], exit_code=code, ended_at=now())
        state.update(exit_code=code, ended_at=now())
        write_json(state_file, state)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def submit(plan, lock, foreground=False):
    directory = Path(plan["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", plan)
    state = initial_state(plan)
    write_json(directory / "job_state.json", state)
    runtime = Path(plan["runtime"])
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
        with Path(plan["log"]).open("w", encoding="utf-8") as stream:
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
    print(f"已提交后台任务 PID={process.pid}，尚不代表训练完成。", flush=True)
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
        return execute(plan)
    finally:
        os.close(lock_fd)


def show_status(runtime):
    pointer = runtime / "last-job.json"
    if not pointer.is_file():
        print("尚未提交后台任务；先运行 bash start_server.sh。")
        return 0
    directory = Path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
    state = json.loads((directory / "job_state.json").read_text(encoding="utf-8"))
    receipt = directory / "receipt.json"
    owner = (
        state
        if state.get("pid")
        else json.loads(receipt.read_text(encoding="utf-8"))
        if receipt.is_file()
        else {}
    )
    status = state["status"]
    if status in {"queued", "running"} and owner.get("pid"):
        current = identity(owner["pid"])
        if current is None or (owner.get("identity") and current != owner["identity"]):
            status = "stale（原进程已退出或服务器已重启；保留最后记录，请查看日志）"
    print(f"状态：{status}\n任务：{directory}\n日志：{state['log']}")
    for step in state["steps"]:
        print(f"  {step['name']}: {step['status']} → {step['output']}")
    if state.get("error"):
        print("错误：" + state["error"])
    return 0
