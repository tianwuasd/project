from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque
from contextlib import contextmanager
import math
import threading
import sys
from time import perf_counter, sleep

from ...preprocessing.scheduling import CaseCPU
from ...training.host import host_capacity


class BlockExecution:
    def __init__(self, workspace):
        self.workspace = workspace
        self.config, self.guard = workspace.config, workspace.guard
        self.settings = self.config["preprocessing"]
        self.cpu = CaseCPU(self.config, self.guard.owner, workspace.case_id)
        cores = max(1, math.floor(host_capacity()["cores"] * self.config["resources"]["cpu_fraction"]))
        self.ceiling = min(cores, self.settings["max_case_threads"] or cores)
        self.pool = ThreadPoolExecutor(self.ceiling, thread_name_prefix="stunet-block")
        self.local = threading.local()
        self.wanted = min(2, self.ceiling)
        self.bytes = self.settings["block_bytes"]
        self.last_adjust, self.last_rate, self.last_threads = perf_counter(), None, 1
        self.phase, self.window_voxels, self.window_seconds = None, 0, 0.
        self.phase_states = {}
        self.cooldown_until = 0.
        self.completed, self.compute_seconds = 0, 0.
        self.peak_threads, self.peak_bytes = 0, 0
        self.applied_bytes = 0
        self.peak_job_bytes = 0
        self.adjustments = []
        self.active, self.max_active = 0, 0
        self.activity_lock = threading.Lock()
        self.parallel_states = []
        self.current_threads = 1

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.cpu.close()

    def _available(self):
        facts = self.guard.observed_boundary()
        if facts["stop_requested"]:
            raise MemoryError("当前病例处理已取消或按配置停止，完整病例保持可复用")
        budget = facts["budget"]
        reusable = max(0, budget["own_pending_bytes"] - budget["own_claim_pending_bytes"])
        return facts, max(0, budget["headroom_bytes"] + reusable)

    @contextmanager
    def _wave(self, jobs, label):
        amount = sum(job[1] for job in jobs)
        with self.guard.admit(amount, label=f"{self.workspace.case_id}:{label}",
                              wait=False) as ready:
            if ready is None:
                self.workspace.release_pages()
                yield False
            else:
                self.peak_bytes = max(self.peak_bytes, amount)
                yield True

    def _call(self, operation, job):
        self.local.allowance = job[1]
        with self.activity_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return operation(job[0])
        finally:
            self.local.allowance = 0
            with self.activity_lock:
                self.active -= 1

    def _select_phase(self, label):
        fields = ("wanted", "last_rate", "last_threads", "window_voxels", "window_seconds", "last_adjust", "cooldown_until")
        if self.phase != label:
            if self.phase is not None:
                self.phase_states[self.phase] = {field: getattr(self, field) for field in fields}
            self.phase, self.last_rate = label, None
            self.wanted, self.last_threads = min(2, self.ceiling), 1
            self.window_voxels, self.window_seconds = 0, 0.
            self.last_adjust, self.cooldown_until = perf_counter(), 0.
            for field, value in self.phase_states.get(label, {}).items():
                setattr(self, field, value)

    def _apply(self, jobs, operation, label, consume=None):
        began = perf_counter()
        self.cpu.shrink(len(jobs))
        self.current_threads = len(jobs)
        with self._wave(jobs, label) as admitted:
            if not admitted:
                return False
            futures = []
            try:
                for job in jobs:
                    futures.append(self.pool.submit(self._call, operation, job))
                results = [future.result() for future in futures]
                if consume is not None:
                    for job, result in zip(jobs, results):
                        consume(job[0], result)
                    del result
                    results.clear()
            finally:
                for future in futures:
                    future.cancel()
                wait(futures)
                if sys.exc_info()[0] is not None:
                    self.pool.shutdown(wait=True, cancel_futures=True)
                if futures:
                    del future
                futures.clear()
        seconds = perf_counter() - began
        self.compute_seconds += seconds
        self.completed += len(jobs)
        self.peak_threads = self.max_active
        self.applied_bytes = max(job[1] for job in jobs)
        self.peak_job_bytes = max(self.peak_job_bytes, self.applied_bytes)
        volume = sum(job[2] for job in jobs)
        self.window_voxels += volume
        self.window_seconds += seconds
        rate = self.window_voxels / max(self.window_seconds, 1e-9)
        now = perf_counter()
        if self.window_seconds >= self.settings["growth_seconds"]:
            previous = self.wanted
            if self.last_rate is not None and len(jobs) > self.last_threads and rate < self.last_rate * .9:
                self.wanted = max(1, self.last_threads)
                self.cooldown_until = now + self.settings["growth_seconds"] * 4
            elif now >= self.cooldown_until:
                self.last_threads, self.last_rate = len(jobs), rate
                self.wanted = min(self.ceiling, len(jobs) * 2)
            if self.settings["auto_blocks"]:
                self.bytes = min(self.settings["max_block_bytes"], self.bytes * 2)
            self.last_adjust = now
            self.window_voxels, self.window_seconds = 0, 0.
            self.adjustments.append({"threads_before": previous, "threads_requested": self.wanted,
                                     "block_bytes": self.bytes, "voxels_per_second": rate})
            self.adjustments = self.adjustments[-64:]
        return True

    def _capacity(self):
        facts, available = self._available()
        if facts["paused"]:
            self.workspace.release_pages()
            with self.guard.admit(0, label="block_pressure_wait", release=self.workspace.release_pages) as ready:
                if ready is None:
                    raise MemoryError("病例等待已取消或按配置停止")
            facts, available = self._available()
        threads = self.cpu.request(self.wanted)
        while threads < 1:
            self._available()
            sleep(.25)
            threads = self.cpu.request(self.wanted)
        state = {"case_slots": self.cpu.slots, "granted_threads": threads, "fair_cap": self.cpu.fair}
        if not self.parallel_states or self.parallel_states[-1] != state:
            self.parallel_states.append(state)
            self.parallel_states = self.parallel_states[-64:]
        minimum = min(4096, self.settings["block_bytes"])
        cap = self.bytes if not self.settings["auto_blocks"] else min(self.bytes, max(minimum, available // (threads * 2)))
        return threads, max(minimum, cap)

    def linear(self, shape, per_voxel, operation, label, extra=None, consume=None):
        self._select_phase(label)
        total = math.prod(shape)
        unit = max(1, self.settings["block_bytes"] // per_voxel)
        done, cursor = 0, 0
        while cursor < total:
            threads, cap = self._capacity()
            jobs = []
            for _ in range(threads):
                if cursor == total:
                    break
                length = min(total - cursor, max(1, cap // per_voxel))
                while True:
                    payload = (cursor, cursor + length)
                    amount = length * per_voxel + (0 if extra is None else extra(payload))
                    if amount <= cap or length == 1:
                        break
                    length = max(1, length // 2)
                jobs.append((payload, amount, length))
                cursor += length
            if not self._apply(jobs, operation, label, consume):
                cursor = jobs[0][0][0]
                continue
            done += sum(job[2] for job in jobs)
            self.workspace.report(label, min(math.ceil(done / unit), math.ceil(total / unit)), math.ceil(total / unit))

    def sections(self, shape, per_voxel, operation, label, full_axis=None):
        self._select_phase(label)
        pending = deque([tuple(slice(0, size) for size in shape)])
        unit = max(1, self.settings["block_bytes"] // per_voxel)
        total, done = math.prod(shape), 0
        while pending:
            threads, cap = self._capacity()
            jobs = []
            for _ in range(threads):
                if not pending:
                    break
                part = pending.popleft()
                sizes = [section.stop - section.start for section in part]
                while math.prod(sizes) * per_voxel > cap:
                    choices = [i for i, size in enumerate(sizes) if i != full_axis and size > 1]
                    if not choices:
                        break
                    axis = max(choices, key=lambda i: sizes[i])
                    middle = part[axis].start + (sizes[axis] + 1) // 2
                    left, right = list(part), list(part)
                    left[axis], right[axis] = slice(part[axis].start, middle), slice(middle, part[axis].stop)
                    pending.appendleft(tuple(right))
                    part = tuple(left)
                    sizes[axis] = middle - part[axis].start
                voxels = math.prod(sizes)
                jobs.append((part, voxels * per_voxel, voxels))
            if not self._apply(jobs, operation, label):
                pending.extendleft(job[0] for job in reversed(jobs))
                continue
            done += sum(job[2] for job in jobs)
            self.workspace.report(label, min(math.ceil(done / unit), math.ceil(total / unit)), math.ceil(total / unit))

    def metrics(self):
        return {"parallel_chunks": self.completed, "wave_seconds": self.compute_seconds,
                "peak_case_threads": self.peak_threads, "peak_wave_bytes": self.peak_bytes,
                "applied_case_threads": self.current_threads, "applied_block_bytes": self.applied_bytes,
                "requested_block_bytes": self.bytes, "peak_job_bytes": self.peak_job_bytes,
                "adjustments": self.adjustments, "parallel_states": self.parallel_states}
