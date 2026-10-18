"""Shared GPU authorization and handoff for single-fold and cross-fold queues."""

import os


def options(gpu_ids, gpu_limit, device, mode):
    if gpu_limit is not None and (
        isinstance(gpu_limit, bool)
        or not isinstance(gpu_limit, int)
        or not 1 <= gpu_limit <= 5
        or not gpu_ids
    ):
        raise ValueError("弹性队列需要候选 GPU 和 1—5 的并发上限")
    if not gpu_ids:
        return None
    if device != "cuda" or mode == "check":
        raise ValueError("多卡队列仅用于 CUDA 训练/短测/预检")
    from ..server import gpu_hold
    from ..server.launcher import gpu_pool, idle_gpus, query_gpus, select_gpus
    from .stage_queue import check_gpu_handoff

    if gpu_limit is not None:
        gpu_ids = [
            r["uuid"]
            for r in gpu_pool(query_gpus(), ",".join(gpu_ids), os.environ.get("CUDA_VISIBLE_DEVICES"))
        ]
    else:
        if len(gpu_ids) != len(set(gpu_ids)) or not 1 <= len(gpu_ids) <= 5:
            raise ValueError("需要 1—5 个不重复 GPU UUID")
        for gpu in gpu_ids:
            check_gpu_handoff(gpu, used=True)
        select_gpus(query_gpus(), len(gpu_ids), ",".join(gpu_ids), os.environ.get("CUDA_VISIBLE_DEVICES"))

    def available():
        devices = query_gpus()
        if not set(gpu_ids).intersection(r["uuid"] for r in devices):
            raise ValueError("候选 GPU 已不可见，请检查设备或分配范围")
        return [r["uuid"] for r in gpu_hold.candidates(idle_gpus(devices, gpu_ids))]

    def before_start(gpu, used):
        if gpu_limit is not None:
            if gpu not in available():
                return False
        else:
            check_gpu_handoff(gpu, used)
        return gpu_hold.acquire(gpu)

    return dict(
        gpu_ids=gpu_ids,
        before_start=before_start,
        available_gpus=available if gpu_limit is not None else None,
        gpu_limit=gpu_limit,
    )
