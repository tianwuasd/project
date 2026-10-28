"""Wait for an idle member of a frozen, allocation-constrained GPU pool."""

import time

from . import gpu_hold
from .launcher import idle_gpus


def wait_for_gpus(pool, limit, query, on_update, interval=5, claim=None):
    """Return up to limit cards as soon as ONE is idle; never wait to fill the cap."""
    started, announced = time.monotonic(), None
    while True:
        rows = query()  # Driver/query failures are errors, not a reason to wait forever.
        if not set(pool).intersection(r["uuid"] for r in rows):
            raise ValueError("候选 GPU 已不可见，请检查设备或分配范围")
        ready = gpu_hold.candidates(idle_gpus(rows, pool))[:limit]
        if claim:
            ready = [r for r in ready if claim(r["uuid"])]
        elapsed = int(time.monotonic() - started)
        detail = dict(
            requested_gpus=limit,
            available_gpus=len(ready),
            pool_uuids=list(pool),
            waited_seconds=elapsed,
            sampled_at=time.time(),
            reason=None if ready else "候选卡有计算进程、空闲显存不足 6000 MiB 或利用率高于 10%",
        )
        on_update("running" if ready else "waiting_gpu", detail)
        if ready:
            print(f"GPU 可用 {len(ready)}/{limit}（所选数量为上限），继续执行。", flush=True)
            return ready
        if announced is None or time.monotonic() - announced >= 30:
            print(
                f"等待 GPU：可用 0/{limit}，已等待 {elapsed} 秒；每 {interval} 秒检查，有一张即开始。",
                flush=True,
            )
            announced = time.monotonic()
        time.sleep(interval)
