from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

from filelock import FileLock
import psutil

from ..artifacts.store import read_json, write_json as durable_write_json
from ..observability.events import event
from .host import host_capacity

MEMORY_LIMIT_BYTES = 100_000_000_000
SAMPLE_SECONDS = 1.0


def write_json(path, value):
    # 活动进程登记仅要求进程间原子可见，断电后由进程出生身份重新核验。
    durable_write_json(path, value, durable=False)


def target_fraction(config):
    # 技术短测可显式使用更高比例；正式任务始终保留至少20%有效内存。
    ceiling = .95 if config.get("scope") == "technical" else .8
    return min(ceiling, config["resources"]["ram_fraction"])


def identity(process=None):
    process = psutil.Process() if process is None else process
    return {"pid": process.pid, "birth": process.create_time()}


def same_process(left, right):
    return left["pid"] == right["pid"] and left["birth"] == right["birth"]


def alive(record):
    try:
        return psutil.Process(record["pid"]).create_time() == record["birth"]
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def registered_members(record):
    members, missing = {}, []
    for member in (record, *record.get("members", [])):
        if (member["pid"], member["birth"]) in members:
            continue
        try:
            process = psutil.Process(member["pid"])
            if process.create_time() != member["birth"]:
                continue
            members[(member["pid"], member["birth"])] = identity(process)
            for child in process.children(recursive=True):
                value = identity(child)
                members[(value["pid"], value["birth"])] = value
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            missing.append({"pid": member["pid"], "field": "descendants", "reason": "access_denied"})
    return list(members.values()), missing


def process_memory(member):
    result = {**member, "rss_bytes": None, "pss_bytes": None, "uss_bytes": None,
              "accounted_bytes": None, "accounting_metric": None, "missing": []}
    try:
        process = psutil.Process(member["pid"])
        if process.create_time() != member["birth"]:
            return {**result, "state": "pid_reused"}
        try:
            information = process.memory_full_info()
        except psutil.AccessDenied:
            information = process.memory_info()
            result["missing"].append({"field": "full_memory_info", "reason": "access_denied"})
        for metric in ("rss", "pss", "uss"):
            value = getattr(information, metric, None)
            result[metric + "_bytes"] = value
            if value is None:
                result["missing"].append({"field": metric + "_bytes", "reason": "unavailable"})
        result["accounting_metric"] = "pss" if result["pss_bytes"] is not None else "rss"
        result["accounted_bytes"] = result[result["accounting_metric"] + "_bytes"]
        result["state"] = "measured"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        result["state"] = "exited"
    except psutil.AccessDenied:
        result["state"] = "access_denied"
        result["missing"].append({"field": "memory", "reason": "access_denied"})
    return result


def aggregate(processes, missing=()):
    living = [row for row in processes if row["state"] not in ("exited", "pid_reused")]
    result = {"process_count": len(living), "missing": list(missing),
              "accounting_metric": "sum_process_pss_else_rss"}
    for metric in ("rss_bytes", "pss_bytes", "uss_bytes", "accounted_bytes"):
        values = [row[metric] for row in living]
        result[metric] = sum(values) if all(value is not None for value in values) else None
        result[metric + "_measured_partial"] = sum(value for value in values if value is not None)
    result["complete"] = result["accounted_bytes"] is not None and not missing
    result["missing"].extend({"pid": row["pid"], **value} for row in living for value in row["missing"])
    return result


def fresh_snapshot(records, facts=None):
    facts = host_capacity() if facts is None else facts
    membership, missing, unique = [], [], {}
    for record in records:
        members, absent = registered_members(record)
        membership.append(members)
        missing.append(absent)
        unique.update({(member["pid"], member["birth"]): member for member in members})
    measured = {key: process_memory(value) for key, value in unique.items()}
    tasks = []
    for record, members, absent in zip(records, membership, missing):
        rows = [measured[(member["pid"], member["birth"])] for member in members]
        tasks.append({"owner": {"pid": record["pid"], "birth": record["birth"]},
                      "members": members, **aggregate(rows, absent)})
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
            "host": facts, "system": {"used_bytes": facts["host_total_bytes"] - facts["host_available_bytes"],
            "rss_bytes": None, "pss_bytes": None, "uss_bytes": None,
            "missing": "system_capacity_has_no_process_rss_pss_uss"}, "tasks": tasks,
            "project": aggregate(list(measured.values()), [value for values in missing for value in values]),
            "processes": list(measured.values())}


def system_headroom(facts, target, reserve):
    scopes = [{"scope": "physical", "total_bytes": facts["host_total_bytes"],
               "used_bytes": facts["host_total_bytes"] - facts["host_available_bytes"]}]
    for row in facts["cgroups"]:
        if row["limit_bytes"] is not None:
            scopes.append({"scope": row["path"], "total_bytes": row["limit_bytes"], "used_bytes": row["current_bytes"]})
    job = facts.get("windows_job")
    if job is not None:
        if job["limit_bytes"] is not None:
            scopes.append({"scope": "windows_job", "total_bytes": job["limit_bytes"], "used_bytes": job["current_bytes"]})
        if job["process_limit_bytes"] is not None:
            scopes.append({"scope": "windows_process", "total_bytes": job["process_limit_bytes"],
                           "used_bytes": job["process_current_bytes"]})
    for row in scopes:
        row["target_bytes"] = int(row["total_bytes"] * target)
        row["headroom_bytes"] = min(row["target_bytes"] - row["used_bytes"],
                                    row["total_bytes"] - row["used_bytes"] - reserve)
    return min(row["headroom_bytes"] for row in scopes), scopes


def pending_bytes(record, measurement, additional=0):
    state = record.get("memory_guard", {})
    claims = state.get("claims", [])
    reserved = sum(row["bytes"] for row in claims if row.get("reserve", False))
    claimed = sum(row["bytes"] for row in claims if not row.get("reserve", False))
    private = measurement["uss_bytes"]
    credited = 0 if private is None else private
    baseline = state.get("claim_baseline_uss_bytes", credited)
    materialized = min(claimed, max(0, credited - baseline)) if claims else 0
    # available 已经扣除实际内存，仅扣除尚未兑现的计划和增量承诺。
    planned = max(0, record["ram_bytes"] - credited)
    return max(planned, reserved + claimed - materialized + additional)


def memory_decision(config, records, snapshot, owner=None, additional=0):
    resources = config["resources"]
    target = target_fraction(config)
    configured_limit = min([resources.get("memory_limit_bytes", MEMORY_LIMIT_BYTES),
                            *(row.get("memory_limit_bytes", MEMORY_LIMIT_BYTES) for row in records)])
    limit = min(MEMORY_LIMIT_BYTES, configured_limit, int(snapshot["host"]["total_bytes"] * target))
    headroom, scopes = system_headroom(snapshot["host"], target, resources["reserve_ram_bytes"])
    pending, own, task_pending = 0, None, 0
    for record, measurement in zip(records, snapshot["tasks"]):
        is_own = owner is not None and same_process(record, owner)
        value = pending_bytes(record, measurement, additional if is_own else 0)
        pending += value
        if is_own:
            own, task_pending = measurement, value
    project = snapshot["project"]
    complete = project["complete"] and (owner is None or own is not None)
    used = project["accounted_bytes"]
    project_spare = None if used is None else limit - used - pending
    task_spare = None if own is None or own["accounted_bytes"] is None else limit - own["accounted_bytes"] - task_pending
    spare = min(headroom - pending, project_spare if project_spare is not None else -1,
                task_spare if owner is not None and task_spare is not None else headroom - pending)
    reason = "eligible" if complete and spare >= 0 else "memory_headroom"
    if not complete:
        reason = "memory_measurement_missing"
    return {"admitted": complete and spare >= 0, "reason": reason, "headroom_bytes": spare,
            "system_headroom_bytes": headroom, "pending_bytes": pending,
            "target_fraction": target, "memory_limit_bytes": limit,
            "configured_memory_limit_bytes": configured_limit, "own": own,
            "own_pending_bytes": task_pending, "project_remaining_bytes": project_spare,
            "scopes": scopes}


def bounded_measurement(value):
    result = {name: item for name, item in value.items() if name not in ("members", "missing")}
    result["missing"] = value.get("missing", [])[:256]
    result["missing_truncated"] = max(0, len(value.get("missing", [])) - 256)
    return result


def bounded_snapshot(snapshot):
    return {**snapshot, "processes": snapshot["processes"][:4096],
            "processes_truncated": max(0, len(snapshot["processes"]) - 4096),
            "tasks": [bounded_measurement(value) for value in snapshot["tasks"][:256]],
            "tasks_truncated": max(0, len(snapshot["tasks"]) - 256),
            "project": bounded_measurement(snapshot["project"]),
            "budget": {**snapshot["budget"], "own": bounded_measurement(snapshot["budget"]["own"])}}


class MemoryGuard:
    def __init__(self, config, stage, owner=None, on_pressure=None):
        self.config, self.stage, self.on_pressure = config, stage, on_pressure
        self.member = identity()
        self.owner = self.member if owner is None else owner
        self.root = Path(config["work_root"]) / "resources"
        self.registry = self.root / "active.json"
        self.lock = self.root / ".resource.lock"
        self.snapshot_path = Path(config["run_dir"]) / "memory.json"
        resources = config["resources"]
        if resources.get("memory_sample_seconds", SAMPLE_SECONDS) != SAMPLE_SECONDS:
            raise ValueError("实际内存采样间隔必须为 1 秒")
        self.pressure_seconds = resources.get("memory_pressure_seconds", resources["adapt_seconds"] * resources["pressure_retries"])
        self.resume_seconds = resources.get("memory_resume_seconds", resources["growth_seconds"])
        self.snapshot_seconds = resources.get("memory_snapshot_seconds", 15)
        self.resume_fraction = resources.get("memory_resume_fraction", 0.02)
        if not all(math.isfinite(value) and value > 0 for value in
                   (self.pressure_seconds, self.resume_seconds, self.snapshot_seconds)):
            raise ValueError("内存保护等待时长必须为有限正数")
        if not 0 < self.resume_fraction < 0.8:
            raise ValueError("内存恢复迟滞比例必须在 (0, 0.8) 内")
        self.stopping = threading.Event()
        self.entered = False
        self.created_record = False
        self.last_notified = None

    def _rows(self):
        records = read_json(self.registry) if self.registry.exists() else []
        records = [row for row in records if alive(row) or any(alive(member) for member in row.get("members", []))]
        for row in records:
            row["members"] = [member for member in row.get("members", []) if alive(member)]
            state = row.get("memory_guard")
            if state is not None:
                previous = state.get("claims", [])
                state["claims"] = [claim for claim in previous if alive(claim)]
                state["waiting"] = [request for request in state.get("waiting", []) if alive(request)]
                if not state["claims"] or len(previous) != len(state["claims"]):
                    state.pop("claim_baseline_uss_bytes", None)
        return records

    def _own(self, records):
        matches = [row for row in records if same_process(row, self.owner)]
        if len(matches) != 1:
            raise RuntimeError("内存保护的项目任务登记已失效")
        return matches[0]

    def _sample_locked(self, records, additional=0, demand=False):
        row = self._own(records)
        snapshot = fresh_snapshot(records)
        state = row.setdefault("memory_guard", {})
        if not alive(row):
            state.setdefault("stop_requested_at", snapshot["time_utc"])
            state.update(paused=True, stop_reason="owner_exited")
        for record, measurement in zip(records, snapshot["tasks"]):
            record["members"] = [value for value in measurement["members"] if not same_process(value, record)]
        waiting = state.get("waiting", [])
        pressure_demand = max([additional, *(value["bytes"] for value in waiting)])
        decision = memory_decision(self.config, records, snapshot, self.owner, additional)
        pressure_decision = memory_decision(self.config, records, snapshot, self.owner, pressure_demand)
        if state.get("stop_reason") == "owner_exited":
            decision.update(admitted=False, reason="owner_exited")
            pressure_decision.update(admitted=False, reason="owner_exited")
        now = snapshot["monotonic"]
        interval = None if "last_sample_at" not in state else now - state["last_sample_at"]
        state["last_sample_at"] = now
        actual = decision["own"]["accounted_bytes"]
        limit = decision["memory_limit_bytes"]
        overloaded = not decision["admitted"]
        if overloaded or (demand or waiting) and not pressure_decision["admitted"]:
            state.setdefault("pressure_since", now)
            state["paused"] = True
            state.pop("healthy_since", None)
        elif state.get("paused"):
            margin = int(snapshot["host"]["total_bytes"] * self.resume_fraction)
            recovery_space = min(decision["system_headroom_bytes"], limit - actual,
                                 limit - snapshot["project"]["accounted_bytes"])
            if pressure_decision["admitted"] and recovery_space >= margin:
                state.setdefault("healthy_since", now)
                if now - state["healthy_since"] >= self.resume_seconds and "stop_requested_at" not in state:
                    state["paused"] = False
                    state.pop("pressure_since", None)
                    state.pop("healthy_since", None)
            else:
                state.pop("healthy_since", None)
        if state.get("paused") and now - state.setdefault("pressure_since", now) >= self.pressure_seconds:
            state.setdefault("stop_requested_at", snapshot["time_utc"])
        peaks = state.setdefault("peaks", {})
        for name, value in (("host_used_bytes", snapshot["host"]["host_total_bytes"] - snapshot["host"]["host_available_bytes"]),
                            ("task_accounted_bytes", actual), ("project_accounted_bytes", snapshot["project"]["accounted_bytes"]),
                            ("task_rss_bytes", decision["own"]["rss_bytes"]), ("task_pss_bytes", decision["own"]["pss_bytes"]),
                            ("task_uss_bytes", decision["own"]["uss_bytes"]),
                            ("project_rss_bytes", snapshot["project"]["rss_bytes"]),
                            ("project_pss_bytes", snapshot["project"]["pss_bytes"]),
                            ("project_uss_bytes", snapshot["project"]["uss_bytes"]),
                            ("pending_bytes", decision["pending_bytes"])):
            if value is not None:
                peaks[name] = max(peaks.get(name, 0), value)
        snapshot.update(stage=self.stage, owner=self.owner, budget=decision, paused=state.get("paused", False),
                        stop_requested="stop_requested_at" in state, stop_reason=state.get("stop_reason"), pressure_since=state.get("pressure_since"),
                        peaks=dict(peaks), sample_seconds=SAMPLE_SECONDS, actual_interval_seconds=interval)
        if now - state.get("last_snapshot_at", -math.inf) >= self.snapshot_seconds or snapshot["stop_requested"]:
            write_json(self.snapshot_path, bounded_snapshot(snapshot))
            state["last_snapshot_at"] = now
        return snapshot

    def _ensure_active(self):
        if not self.entered:
            raise RuntimeError("请在 MemoryGuard context 内检查内存")
        if self.future.done():
            self.future.result()

    def snapshot(self):
        self._ensure_active()
        with FileLock(str(self.lock)):
            records = self._rows()
            snapshot = self._sample_locked(records)
            write_json(self.registry, records)
        return snapshot

    def _monitor(self):
        next_sample = time.monotonic() + SAMPLE_SECONDS
        while not self.stopping.wait(max(0, next_sample - time.monotonic())):
            with FileLock(str(self.lock)):
                records = self._rows()
                self._sample_locked(records)
                write_json(self.registry, records)
            next_sample += SAMPLE_SECONDS
            now = time.monotonic()
            if next_sample < now:
                next_sample += (math.floor((now - next_sample) / SAMPLE_SECONDS) + 1) * SAMPLE_SECONDS

    def _notify(self, snapshot):
        signature = (snapshot["paused"], snapshot["stop_requested"])
        if signature != self.last_notified and signature != (False, False):
            bounded = bounded_snapshot(snapshot)
            event(self.config["run_dir"], "memory_pressure", stage=self.stage,
                  paused=signature[0], stop_requested=signature[1], budget=bounded["budget"],
                  task=bounded["budget"]["own"], project=bounded["project"], peaks=snapshot["peaks"])
            if self.on_pressure is not None:
                self.on_pressure(snapshot)
        self.last_notified = signature

    def boundary(self):
        snapshot = self.snapshot()
        self._notify(snapshot)
        return snapshot

    def checkpoint(self):
        return self.boundary()

    @property
    def stop_requested(self):
        return self.boundary()["stop_requested"]

    @contextmanager
    def admit(self, amount, label="allocation", wait=True, reserve=False):
        if type(amount) is not int or amount < 0:
            raise ValueError("内存准入量必须为非负整数 bytes")
        if type(reserve) is not bool:
            raise TypeError("reserve 必须为布尔值")
        self._ensure_active()
        token = uuid4().hex
        acquired = False
        while True:
            with FileLock(str(self.lock)):
                records = self._rows()
                state = self._own(records).setdefault("memory_guard", {})
                requests = state.setdefault("waiting", [])
                if not any(value["token"] == token for value in requests):
                    requests.append({**self.member, "token": token, "bytes": amount})
                snapshot = self._sample_locked(records, amount, demand=True)
                if snapshot["budget"]["admitted"] and not snapshot["paused"] and not snapshot["stop_requested"]:
                    state = self._own(records).setdefault("memory_guard", {})
                    claims = state.setdefault("claims", [])
                    private = snapshot["budget"]["own"]["uss_bytes"] or 0
                    previous = sum(value["bytes"] for value in claims if not value.get("reserve", False))
                    credited = min(previous, max(0, private - state.get("claim_baseline_uss_bytes", private)))
                    state["claim_baseline_uss_bytes"] = private - credited
                    claims.append({**self.member, "token": token, "bytes": amount, "label": label[:256], "reserve": reserve})
                    acquired = True
                if acquired or not wait or snapshot["stop_requested"]:
                    state["waiting"] = [value for value in requests if value["token"] != token]
                write_json(self.registry, records)
            self._notify(snapshot)
            if acquired or not wait or snapshot["stop_requested"]:
                break
            if self.stopping.wait(SAMPLE_SECONDS):
                raise RuntimeError("内存保护已退出，无法继续等待新分配")
            self._ensure_active()
        try:
            yield snapshot if acquired else None
        finally:
            if acquired:
                with FileLock(str(self.lock)):
                    records = self._rows()
                    state = self._own(records)["memory_guard"]
                    claims = state["claims"]
                    if sum(claim["token"] == token for claim in claims) != 1:
                        raise RuntimeError("内存准入额度发生重复释放或丢失")
                    measurement = next(value for value in fresh_snapshot(records)["tasks"]
                                       if same_process(value["owner"], self.owner))
                    private = measurement["uss_bytes"] or 0
                    credited = min(sum(value["bytes"] for value in claims if not value.get("reserve", False)),
                                   max(0, private - state.get("claim_baseline_uss_bytes", private)))
                    state["claims"] = [claim for claim in claims if claim["token"] != token]
                    if not state["claims"]:
                        state.pop("claim_baseline_uss_bytes", None)
                    else:
                        state["claim_baseline_uss_bytes"] = private - max(0, credited - (0 if reserve else amount))
                    write_json(self.registry, records)

    def __enter__(self):
        if self.entered:
            raise RuntimeError("MemoryGuard context 不允许重复进入")
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock)):
            records = self._rows()
            matches = [row for row in records if same_process(row, self.owner)]
            if not matches:
                if not same_process(self.owner, self.member):
                    raise RuntimeError("worker 的父任务尚未登记，禁止独立扩大预算")
                records.append({**self.owner, "members": [], "ram_bytes": 0, "cpu_slots": 0, "uuids": [],
                                "memory_limit_bytes": self.config["resources"].get("memory_limit_bytes", MEMORY_LIMIT_BYTES)})
                self.created_record = True
            row = self._own(records)
            if not same_process(self.member, self.owner) and not any(same_process(self.member, value) for value in row.get("members", [])):
                row.setdefault("members", []).append(self.member)
            self._sample_locked(records)
            write_json(self.registry, records)
        self.stopping.clear()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-monitor")
        self.future = self.executor.submit(self._monitor)
        self.entered = True
        return self

    def __exit__(self, exception_type, exception, trace):
        self.stopping.set()
        self.executor.shutdown(wait=True)
        self.future.result()
        with FileLock(str(self.lock)):
            records = self._rows()
            row = self._own(records)
            if any(same_process(claim, self.member) for claim in row.get("memory_guard", {}).get("claims", [])):
                raise RuntimeError("退出内存保护时仍有未释放的准入额度")
            if self.created_record:
                records.remove(row)
            write_json(self.registry, records)
        self.entered = False
        return False
