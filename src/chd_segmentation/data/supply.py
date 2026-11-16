from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
import multiprocessing as mp
import os
from time import perf_counter

import numpy as np
import torch

from .dataset import CaseDataset, sample_patch
from ..training.resources import register_member
from ..observability.events import event
from ..training.memory_guard import MemoryGuard

_dataset = None
_gate = None
_config = None
_rank = None


def initialize(prepared, config, resources, gate, owner, rank):
    global _dataset, _gate, _config, _rank
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    register_member(config, owner)
    _dataset, _gate = CaseDataset(prepared, resources, log_dir=config["run_dir"], logging=config.get("logging")), gate
    _config, _rank = config, rank
    event(config["run_dir"], "worker_ready", rank=rank, cache_budget_bytes=resources["cache_bytes"], threads=1)


def batch_index(settings, phase, index):
    steps = settings["steps_per_epoch"] if settings["epochs"] else settings["attempts"]
    validation = settings["validation_steps"] if settings["epochs"] else 1
    width = steps if phase == 0 else validation
    epoch, position = divmod(index, width)
    return epoch * (steps + validation) + (position if phase == 0 else steps + position)


def selection(request):
    names, settings, phase, attempt, slot = request
    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], phase, attempt, slot]))
    name = names[int(rng.integers(len(names)))]
    return name, rng, {"batch_index": batch_index(settings, phase, attempt),
        "phase": "train" if phase == 0 else "validation", "sample_index": attempt, "global_slot": slot, "case_id": name}


def make_patch(dataset, gate, config, rank, request):
    amount = int(np.prod(config["training"]["patch"])) * 16 * 3
    with MemoryGuard(config, "patch", owner=config.get("memory_owner")) as memory:
        with memory.admit(amount, label="sample_patch", release=dataset.clear) as allowance:
            if allowance is None:
                dataset.clear()
                return None
            return admitted_patch(dataset, gate, config, rank, request)


def admitted_patch(dataset, gate, config, rank, request):
    _, settings, phase, _, _ = request
    name, rng, context = selection(request)
    context["rank"] = rank
    began = perf_counter()
    hits = dataset.hits
    cached = name in dataset.cache
    with nullcontext({"case_estimate_bytes": 0}) if cached else gate.case(name) as allowance:
        event(config["run_dir"], "case_build_started", console=False, **context, cached=cached, budget=allowance)
        entry = dataset.get(name)
        pair = sample_patch(entry, settings["patch"], rng, settings["foreground_fraction"] if phase == 0 else 0.)
        unknown = entry["unknown_classes"]
        del entry
    elapsed = perf_counter() - began
    event(config["run_dir"], "case_build_completed", console=False, **context, seconds=elapsed,
          cache_hit=dataset.hits > hits, cached_bytes=dataset.cached_bytes)
    return pair, unknown, name, elapsed, dataset.hits > hits, context


def patch_task(request):
    return make_patch(_dataset, _gate, _config, _rank, request)


class BatchSupply:
    def __init__(self, prepared, config, slots, workers, cache_bytes, gate, owner, rank=0, cancel=None, prefetch=1):
        self.prepared, self.config, self.slots = prepared, config, slots
        self.workers, self.cache_bytes, self.gate = workers, cache_bytes, gate
        self.owner = owner
        self.rank = rank
        self.external_cancel = cancel
        if type(prefetch) is not int or prefetch not in (1, 2):
            raise ValueError("供数容量仅允许当前批或当前批加下一批")
        self.prefetch = prefetch
        self.pool = None
        self.local_pool = None
        self.local = None
        self.queue = deque()
        self.inflight = None
        self.transfer_state = None

    def __enter__(self):
        self.cancel = mp.get_context('spawn').Event()
        self.signals = (self.cancel,) if self.external_cancel is None else (self.cancel, self.external_cancel)
        self.config = {**self.config, "memory_owner": self.owner, 'memory_cancel': self.signals}
        resources = {**self.config["resources"], "cache_bytes": self.cache_bytes}
        if self.workers:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                initializer=initialize, initargs=(self.prepared, self.config, resources, self.gate, self.owner, self.rank))
        else:
            self.local = CaseDataset(self.prepared, resources, log_dir=self.config["run_dir"],
                                     logging=self.config.get("logging"))
            if self.prefetch == 2:
                self.local_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-supply")
        return self

    def submit(self, names, phase, attempt):
        occupied = len(self.queue) + int(self.inflight is not None)
        if self.queue or occupied >= self.prefetch:
            raise BufferError("供数容量已满，最多保留当前批和下一顺序批")
        requests = [(names, self.config["training"], phase, attempt, slot) for slot in self.slots]
        contexts = [selection(request)[2] for request in requests]
        index = contexts[0]["batch_index"]
        if self.pool:
            pending = [self.pool.submit(patch_task, request) for request in requests]
        elif self.local_pool:
            pending = [self.local_pool.submit(make_patch, self.local, self.gate, self.config, self.rank, request)
                       for request in requests]
        else:
            pending = requests
        self.queue.append({"batch_index": index, "pending": pending,
                           "futures": self.pool is not None or self.local_pool is not None})
        event(self.config["run_dir"], "batch_enqueued", console=False, rank=self.rank,
              batch_index=index, samples=contexts, queue_capacity=self.prefetch, occupied_batches=occupied + 1)

    def take_cpu(self):
        if not self.queue or self.inflight is not None:
            raise RuntimeError("供数队列为空或当前批尚未消费")
        queued = self.queue.popleft()
        if queued["futures"]:
            results = [future.result() for future in queued["pending"]]
        else:
            results = [make_patch(self.local, self.gate, self.config, self.rank, request) for request in queued["pending"]]
        if any(result is None for result in results):
            self.inflight = {"batch_index": queued["batch_index"], "case_ids": [], "slots": self.slots}
            return {"cancelled": True, "batch_index": self.inflight["batch_index"]}
        with MemoryGuard(self.config, "batch_stack", owner=self.owner) as memory:
            with memory.admit(sum(a.nbytes for r in results for a in r[0]) * 2, label="batch_stack") as allowed:
                if allowed is None:
                    self.inflight = {"batch_index": queued["batch_index"], "case_ids": [], "slots": self.slots}
                    return {"cancelled": True, "batch_index": queued["batch_index"]}
                tensors = [torch.from_numpy(np.stack(values)) for values in zip(*(r[0] for r in results))]
        if any(result[5]["batch_index"] != queued["batch_index"] for result in results):
            raise RuntimeError("供数结果与已派发的批次顺序不一致")
        self.inflight = {"batch_index": results[0][5]["batch_index"], "case_ids": [r[2] for r in results],
                         "slots": self.slots}
        event(self.config["run_dir"], "cpu_batch_dequeued", console=False, rank=self.rank, **self.inflight,
              queue_capacity=self.prefetch, occupied_batches=1 + len(self.queue),
              cpu_tensor_bytes=sum(t.numel() * t.element_size() for t in tensors))
        return {"image": tensors[0], "label": tensors[1], "unknown_classes": [r[1] for r in results],
                "case_ids": [r[2] for r in results], "supply_seconds": sum(r[3] for r in results),
                "cache_hits": sum(r[4] for r in results), "cache_misses": sum(not r[4] for r in results),
                "batch_index": self.inflight["batch_index"]}

    def transfer(self, batch, device):
        if self.inflight is None or batch["batch_index"] != self.inflight["batch_index"]:
            raise RuntimeError("传输的批次与当前供数槽位不一致")
        began = perf_counter()
        tensors = [batch["image"], batch["label"]]
        if device.type == "cuda":
            if self.transfer_state is not None:
                raise RuntimeError("上一批传输缓冲尚未释放")
            pinned = [value.pin_memory() for value in tensors]
            started = torch.cuda.Event(enable_timing=True)
            started.record(torch.cuda.current_stream(device))
            completed = torch.cuda.Event(enable_timing=True)
            self.transfer_state = {"batch_index": batch["batch_index"], "pinned": pinned,
                                   "started": started, "completed": completed, "host_started": began,
                                   "device": str(device)}
            # 当前计算流按序等待拷贝；源缓冲由批次和供数器共同持有至消费或异常收尾。
            batch["_host_buffers"], batch["_copy_event"] = pinned, completed
            try:
                transferred = [value.to(device, non_blocking=True) for value in pinned]
            finally:
                # 部分拷贝成功后发生错误时，异常收尾仍须等候已入队的拷贝。
                completed.record(torch.cuda.current_stream(device))
            tensors = transferred
        batch["image"], batch["label"] = tensors
        elapsed = perf_counter() - began
        event(self.config["run_dir"], "batch_transfer_enqueued" if device.type == "cuda" else "batch_transfer_completed",
              console=False, rank=self.rank, **self.inflight, device=str(device),
              seconds=elapsed if device.type != "cuda" else None, enqueue_seconds=elapsed,
              copy_completed=device.type != "cuda")
        return batch

    def take(self, device):
        return self.transfer(self.take_cpu(), device)

    def release_transfer(self):
        if self.transfer_state is None:
            return
        state = self.transfer_state
        if not state["completed"].query():
            state["completed"].synchronize()
        copied_seconds = state["started"].elapsed_time(state["completed"]) / 1000.
        event(self.config["run_dir"], "batch_transfer_completed", console=False, rank=self.rank,
              batch_index=state["batch_index"], device=state["device"], copy_completed=True,
              seconds=copied_seconds, device_copy_seconds=copied_seconds,
              host_buffer_lifetime_seconds=perf_counter() - state["host_started"])
        self.transfer_state = None

    def mark_consumed(self, discarded=False):
        if self.inflight is None:
            raise RuntimeError("供数批次发生重复消费或尚未出队")
        self.release_transfer()
        event(self.config["run_dir"], "batch_discarded" if discarded else "batch_consumed", console=False,
              rank=self.rank, **self.inflight, queue_capacity=self.prefetch, occupied_batches=len(self.queue))
        self.inflight = None

    def __exit__(self, *_):
        self.cancel.set()
        self.release_transfer()
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        if self.local_pool:
            self.local_pool.shutdown(wait=True, cancel_futures=True)
            self.local_pool = None
        if self.local:
            self.local.clear()
            self.local = None
        self.queue.clear()
        self.inflight = None
