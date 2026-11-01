"""Dependency-aware, one-process-per-GPU training, adapted from whole_heart/project1."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from ..server.launcher import write_json


def run_queue(
    stages,
    gpu_ids,
    output,
    command_for,
    env,
    before_start=None,
    on_complete=None,
    lock_fds=(),
    available_gpus=None,
    gpu_limit=None,
    poll_interval=5,
    *,
    dependencies=None,
    result_path_for=None,
    log_path_for=None,
    on_state=None,
):
    """Only the coordinator writes models.json; workers write stage-specific files."""
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("队列需要不重复的 GPU UUID")
    limit = len(gpu_ids) if gpu_limit is None else gpu_limit
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5:
        raise ValueError("队列并发上限需要为 1—5")
    output = Path(output)
    logs = output / "stage_logs"
    logs.mkdir(exist_ok=True)
    pending, completed, active, used = list(stages), {}, {}, set()
    if len(pending) != len(set(pending)):
        raise ValueError("训练队列阶段重复")
    dependencies = (
        {"blood_lstm": ["blood2d"]}
        if dependencies is None and "blood_lstm" in pending
        else dependencies or {}
    )
    if not set(dependencies).issubset(pending) or any(
        not set(required).issubset(pending) for required in dependencies.values()
    ):
        raise ValueError("训练队列依赖不在本次任务中")
    result_path_for = result_path_for or (lambda stage: output / (stage + ".result.json"))
    log_path_for = log_path_for or (lambda stage: logs / (stage + ".log"))
    records = {stage: {"status": "pending"} for stage in stages}
    state = {
        "status": "running",
        "gpu_uuids": list(gpu_ids),
        "gpu_limit": limit,
        "stages": records,
        "sampled_at": time.time(),
    }
    last_scan, last_notice, started = float("-inf"), float("-inf"), time.monotonic()
    last_summary = None
    handlers, stopped = {}, [False]

    def ready(stage):
        return all(required in completed for required in dependencies.get(stage, ()))

    def publish():
        state.update(
            active_gpus=len(active),
            pending_stages=len(pending),
            elapsed_seconds=int(time.monotonic() - started),
            sampled_at=time.time(),
        )
        if on_state:
            on_state(state)
        write_json(output / "queue.json", state)

    def stop_requested(sig, frame):
        # Do not raise between process creation and registration.
        stopped[0] = True

    watched = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        watched.append(signal.SIGHUP)
    for sig in watched:
        handlers[sig] = signal.signal(sig, stop_requested)
    try:
        while pending or active:
            if stopped[0]:
                raise KeyboardInterrupt("训练队列已中断")
            finished = False
            for gpu, (process, stream, stage) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[gpu]
                finished = True
                if code:
                    records[stage].update(status="failed", exit_code=code)
                    raise RuntimeError(f"阶段 {stage} 失败（{code}），查看 {log_path_for(stage)}")
                info = json.loads(Path(result_path_for(stage)).read_text(encoding="utf-8"))
                completed[stage] = info
                records[stage]["status"] = "complete"
                if on_complete:
                    on_complete(stage, info)
            moment = time.monotonic()
            scan = available_gpus is None or finished or moment - last_scan >= poll_interval
            candidates = []
            if scan and pending and len(active) < limit:
                candidates = list(available_gpus() if available_gpus else gpu_ids)
                if not set(candidates).issubset(gpu_ids) or len(set(candidates)) != len(candidates):
                    raise ValueError("可用 GPU 超出候选池或重复")
                last_scan = moment
            for gpu in candidates:
                if len(active) >= limit:
                    break
                if gpu in active:
                    continue
                selected = next((s for s in pending if ready(s)), None)
                if selected is None:
                    continue
                if before_start and before_start(gpu, gpu in used) is False:
                    # Another process acquired this card after the snapshot.
                    # Continue other cards; do not fail or block the whole queue.
                    continue
                if stopped[0]:
                    raise KeyboardInterrupt("训练队列已中断")
                log_path = Path(log_path_for(selected))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        list(map(str, command_for(selected))),
                        env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        # Keep workers in the coordinator's process group so server cleanup reaches all.
                        start_new_session=False,
                        **(
                            {
                                "pass_fds": tuple(
                                    set(lock_fds)
                                    | (
                                        {int(env["CHD_RUNTIME_LOCK_FD"])}
                                        if env.get("CHD_RUNTIME_LOCK_FD")
                                        else set()
                                    )
                                )
                            }
                            if os.name == "posix"
                            else {}
                        ),
                    )
                except BaseException:
                    stream.close()
                    raise
                active[gpu] = (process, stream, selected)
                pending.remove(selected)
                used.add(gpu)
                from ..server.jobs import identity

                records[selected].update(
                    status="running", gpu_uuid=gpu, pid=process.pid, identity=identity(process.pid)
                )
                publish()
                print(f"启动 {selected}，GPU={gpu}，日志：{log_path}", flush=True)
            if pending and not active and not any(ready(s) for s in pending):
                raise RuntimeError("训练队列依赖无法满足")
            state.update(
                status="waiting_gpu" if pending and not active else "running",
                active_gpus=len(active),
                pending_stages=len(pending),
                elapsed_seconds=int(time.monotonic() - started),
                sampled_at=time.time(),
            )
            summary = (state["status"], len(active), len(pending))
            if available_gpus and (summary != last_summary or time.monotonic() - last_notice >= 30):
                message = "等待 GPU" if state["status"] == "waiting_gpu" else "训练进行中"
                print(
                    f"{message}：正在使用 {len(active)}/{limit} 张卡，待运行 {len(pending)} 个阶段；空闲卡不足时继续少卡运行。",
                    flush=True,
                )
                last_notice, last_summary = time.monotonic(), summary
            publish()
            if active or pending:
                time.sleep(0.2)
        state["status"] = "complete"
        return completed
    except BaseException as error:
        state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        for process, stream, stage in active.values():
            try:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                stream.close()
                records[stage]["status"] = "interrupted"
        for stage in pending:
            records[stage]["status"] = "not_started"
        active.clear()
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        publish()


def check_gpu_handoff(gpu, used):
    from ..server.launcher import query_gpus, select_gpu

    # A previous worker's utilization sample can linger briefly after exit.
    for attempt in range(11 if used else 1):
        rows = query_gpus()
        target = next((r for r in rows if r["uuid"] == gpu), None)
        if target is None or target["busy"] or target["free"] < 6000:
            raise ValueError("GPU 不再空闲：" + gpu)
        if target["util"] <= 10:
            select_gpu(rows, gpu, os.environ.get("CUDA_VISIBLE_DEVICES"))
            return
        if not used or attempt == 10:
            raise ValueError("GPU 利用率尚未回落：" + gpu)
        time.sleep(1)
