from copy import deepcopy
import hashlib
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import unittest

import torch

from chd_segmentation.data.supply import BatchSupply
from chd_segmentation.training.build_gate import BuildGate
from chd_segmentation.training.resources import resource_lease


def fingerprint(batch):
    checksum = hashlib.sha256()
    for key in ("image", "label"):
        value = batch[key].numpy()
        checksum.update(str((value.dtype, value.shape)).encode())
        checksum.update(value.tobytes())
    return {"batch_index": batch["batch_index"], "case_ids": batch["case_ids"], "sha256": checksum.hexdigest()}


def verify_supply(config, prepared):
    # 调用方提供已获准的两例技术区域和独占工作区，本函数不启动优化器更新。
    if config["scope"] != "technical" or not 1 <= len(prepared["manifest"]["cases"]) <= 2:
        raise ValueError("供数检查仅允许一至两例真实技术区域")
    names = [row["case_id"] for row in prepared["manifest"]["cases"] if "train" in row["roles"]]
    if not names:
        raise ValueError("供数检查缺少真实训练病例")
    context = mp.get_context("spawn")
    results = []
    for capacity in (1, 2):
        current = deepcopy(config)
        current["run_dir"] = str(Path(config["run_dir"]) / f"supply-capacity-{capacity}")
        current["resources"].update(device="cpu", cpu_processes=1, workers=0, prefetch=capacity)
        began = perf_counter()
        with resource_lease(current, prepared, {}) as allocation:
            applied = allocation["supply"]
            if applied["prefetch"] != capacity:
                raise MemoryError("供数检查未获得所需的一批或两批实际额度")
            gate = BuildGate(context, applied["reader_slots"], applied["workspace_capacity_bytes"], applied["case_estimates"])
            slots = list(range(config["training"]["batch"]))
            with BatchSupply(prepared, current, slots, 0, applied["cache_bytes"][0], gate,
                             allocation["owner"], prefetch=capacity) as supply:
                supply.submit(names, 0, 0)
                first = supply.take_cpu()
                if first.get("cancelled"):
                    raise MemoryError("供数检查的第一批被资源保护取消")
                actual = [fingerprint(first)]
                first = supply.transfer(first, torch.device("cpu"))
                if capacity == 2:
                    supply.submit(names, 0, 1)
                    assert supply.inflight["batch_index"] == first["batch_index"]
                    assert len(supply.queue) == 1 and supply.queue[0]["batch_index"] != first["batch_index"]
                    with unittest.TestCase().assertRaises(BufferError):
                        supply.submit(names, 0, 2)
                    for future in supply.queue[0]["pending"]:
                        if future.result(timeout=60) is None:
                            raise MemoryError("下一顺序批预取被资源保护取消")
                    assert supply.inflight is not None
                else:
                    with unittest.TestCase().assertRaises(BufferError):
                        supply.submit(names, 0, 1)
                supply.mark_consumed()
                del first
                if capacity == 1:
                    supply.submit(names, 0, 1)
                second = supply.take_cpu()
                if second.get("cancelled"):
                    raise MemoryError("供数检查的第二批被资源保护取消")
                actual.append(fingerprint(second))
                supply.mark_consumed()
                assert not supply.queue and supply.inflight is None
                del second
        results.append({"capacity": capacity, "batches": actual, "seconds": perf_counter() - began,
                        "next_batch_ready_while_current_unconsumed": capacity == 2, "optimizer_updates": 0})
    assert results[0]["batches"] == results[1]["batches"]
    return {"state": "passed", "ordered_batches_equal": True, "runs": results,
            "cuda_transfer_tested": False, "optimizer_updates": 0}
