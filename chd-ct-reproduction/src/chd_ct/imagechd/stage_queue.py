"""Dependency-aware, one-process-per-GPU training, adapted from whole_heart/project1."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from ..server.launcher import write_json


def run_queue(stages, gpu_ids, output, command_for, env, before_start=None, on_complete=None):
    """Only the coordinator writes models.json; workers write stage-specific files."""
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("队列需要不重复的 GPU UUID")
    output = Path(output)
    logs = output / "stage_logs"
    logs.mkdir()
    pending, completed, active, used = list(stages), {}, {}, set()
    records = {stage: {"status": "pending"} for stage in stages}
    state = {"status": "running", "gpu_uuids": list(gpu_ids), "stages": records}
    handlers, stopped = {}, [False]

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
            for gpu, (process, stream, stage) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[gpu]
                if code:
                    records[stage].update(status="failed", exit_code=code)
                    raise RuntimeError(f"阶段 {stage} 失败（{code}），查看 {logs / (stage + '.log')}")
                info = json.loads((output / (stage + ".result.json")).read_text(encoding="utf-8"))
                completed[stage] = info
                records[stage]["status"] = "complete"
                if on_complete:
                    on_complete(stage, info)
            for gpu in gpu_ids:
                if gpu in active:
                    continue
                ready = next((s for s in pending if s != "blood_lstm" or "blood2d" in completed), None)
                if ready is None:
                    continue
                if before_start:
                    before_start(gpu, gpu in used)
                if stopped[0]:
                    raise KeyboardInterrupt("训练队列已中断")
                stream = (logs / (ready + ".log")).open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        list(map(str, command_for(ready))),
                        env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        # Keep workers in the coordinator's process group so server cleanup reaches all.
                        start_new_session=False,
                    )
                except BaseException:
                    stream.close()
                    raise
                active[gpu] = (process, stream, ready)
                pending.remove(ready)
                used.add(gpu)
                records[ready].update(status="running", gpu_uuid=gpu, pid=process.pid)
                print(f"启动 {ready}，GPU={gpu}，日志：{logs / (ready + '.log')}", flush=True)
            write_json(output / "queue.json", state)
            if pending and not active:
                raise RuntimeError("训练队列依赖无法满足")
            if active:
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
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        write_json(output / "queue.json", state)


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
