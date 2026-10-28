"""Workflow-scoped CUDA context holders; no dummy compute or blanket GPU claims.

Only recorded, identity-checked holder PIDs are ignored by this workflow's
occupancy checks. Other users' processes always remain busy. Holders follow the
workflow owner, not a short-lived stage, and exit if that owner disappears.
"""

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from ..recovery import output_lock, read_json, write_json

ENV = "CHD_GPU_HOLD_DIR"
_children = {}


def alive(record):
    from .jobs import identity

    return bool(record.get("identity")) and identity(record["pid"]) == record["identity"]


def begin(plan):
    """Called by the workflow owner; CPU-only workflows create no GPU state."""
    gpu_steps = [
        s["arguments"]
        for s in plan["steps"]
        if not s.get("reuse") and s["arguments"].get("device") == "cuda"
    ]
    if not gpu_steps:
        return None
    from .jobs import identity

    directory = Path(plan["directory"]) / "gpu-holds" / uuid4().hex
    directory.mkdir(parents=True)
    write_json(
        directory / "owner.json",
        dict(
            pid=os.getpid(),
            identity=identity(os.getpid()),
            limit=max(int(a.get("gpu_count") or 1) for a in gpu_steps),
            pool=[],
        ),
    )
    os.environ[ENV] = str(directory)
    return directory


def directory():
    value = os.environ.get(ENV)
    return Path(value) if value else None


def records(folder=None):
    folder = folder or directory()
    if folder is None:
        return []
    result = []
    for path in Path(folder).glob("*.holder.json"):
        if path.name.startswith("."):
            continue  # Atomic-write temporary files have not been published.
        try:
            result.append(read_json(path))
        except FileNotFoundError:
            pass  # A cancelled acquisition may remove its receipt during a query.
    return result


def owned_processes():
    folder = directory()
    if folder is None:
        return {}
    if not alive(read_json(folder / "owner.json")):
        raise ValueError("GPU 保留所属流程已退出，停止使用其保留记录")
    result = {}
    for record in records(folder):
        if not alive(record):
            raise ValueError("GPU 保留进程意外退出：" + record["gpu"] + "；请中断后续跑")
        result[record["pid"]] = record["gpu"]
    return result


def configure(pool):
    folder = directory()
    if folder is None:
        return
    with output_lock(folder):
        owner = read_json(folder / "owner.json")
        if not alive(owner):
            raise ValueError("GPU 保留所属流程已退出")
        if not set(owned_processes().values()).issubset(pool):
            raise ValueError("后续步骤的 GPU 范围排除了已保留显卡，请保持本流程的候选范围一致")
        owner["pool"] = list(pool)
        write_json(folder / "owner.json", owner)


def candidates(rows):
    """Prefer already held cards and never acquire above the workflow's cap."""
    folder = directory()
    if folder is None:
        return rows
    held = set(owned_processes().values())
    limit = read_json(folder / "owner.json")["limit"]
    if len(held) >= limit:
        return [r for r in rows if r["uuid"] in held]
    return sorted(rows, key=lambda r: r["uuid"] not in held)


def _command(python, folder, receipt):
    return [str(python), "-u", "-m", __name__, str(folder), str(receipt)]


def acquire(gpu, python=None, timeout=45):
    """Claim only an idle authorized candidate, immediately before useful work."""
    folder = directory()
    if folder is None:
        return True
    from .launcher import idle_gpus, query_gpus

    with output_lock(folder):
        owner = read_json(folder / "owner.json")
        held = set(owned_processes().values())
        if gpu not in owner["pool"]:
            raise ValueError("GPU 超出本流程获准候选范围：" + gpu)
        if gpu in held:
            return True
        if len(held) >= owner["limit"]:
            return False
        if not idle_gpus(query_gpus(), [gpu]):
            return False
        token = hashlib.sha256(gpu.encode()).hexdigest()[:16] + "_" + uuid4().hex[:8]
        receipt = folder / (token + ".holder.json")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CHD_HELD_GPU=gpu)
        process, cancelled = None, False
        handlers = {}

        def interrupt(sig, frame):
            nonlocal cancelled
            cancelled = True

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupt)
            with (folder / (token + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    _command(python or sys.executable, folder, receipt),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    **(
                        {"pass_fds": (int(env["CHD_RUNTIME_LOCK_FD"]),)}
                        if os.name == "posix" and env.get("CHD_RUNTIME_LOCK_FD")
                        else {}
                    ),
                )
                deadline = time.monotonic() + timeout
                while not receipt.is_file():
                    if cancelled:
                        raise KeyboardInterrupt("GPU 保留已取消")
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise ValueError("GPU 保留启动失败/超时，查看：" + str(log.name))
                    time.sleep(0.1)
                if cancelled:
                    raise KeyboardInterrupt("GPU 保留已取消")
                record = read_json(receipt)
                if not alive(record) or record["pid"] != process.pid or record["gpu"] != gpu:
                    raise ValueError("GPU 保留进程身份核验失败")
                _children[process.pid] = process
                print("已保留 GPU 至本流程结束：" + gpu, flush=True)
                return True
        except BaseException:
            if process is not None:
                process.kill()
                process.wait()
                _children.pop(process.pid, None)
            receipt.unlink(missing_ok=True)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if cancelled:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                if process is not None:
                    _children.pop(process.pid, None)
                receipt.unlink(missing_ok=True)
                raise KeyboardInterrupt("GPU 保留已取消")


def release(folder):
    """Release only our recorded holder identities, even after stage failure."""
    if folder is None:
        return
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        _release(folder)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _release(folder):
    all_records = records(folder)
    live = [r for r in all_records if alive(r)]
    for record in live:
        try:
            os.kill(record["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while any(alive(r) for r in live) and time.monotonic() < deadline:
        time.sleep(0.1)
    for record in live:
        if alive(record):
            os.kill(record["pid"], signal.SIGKILL)
    for record in all_records:
        child = _children.pop(record["pid"], None)
        if child is not None:
            child.wait(timeout=5)
    write_json(Path(folder) / "released.json", dict(released_at=time.time(), holders=live))
    print(f"已释放本流程保留的 {len(live)} 张 GPU。", flush=True)


def hold(folder, receipt):
    """Runs in the training Python environment; keeps one tiny CUDA tensor."""
    from .jobs import identity

    owner = read_json(Path(folder) / "owner.json")
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    if not alive(owner):
        return
    import torch

    torch.set_num_threads(1)
    context = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    if stopped[0] or not alive(owner):
        return
    write_json(
        Path(receipt),
        dict(
            pid=os.getpid(),
            identity=identity(os.getpid()),
            gpu=os.environ["CHD_HELD_GPU"],
            scope="workflow",
            started_at=time.time(),
        ),
    )
    while not stopped[0] and alive(owner):
        time.sleep(0.5)
    del context


if __name__ == "__main__":
    hold(sys.argv[1], sys.argv[2])
