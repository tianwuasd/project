import math
from pathlib import Path

from filelock import FileLock

from ..artifacts.store import write_json
from .host import host_capacity
from .memory import alive, identity, same_process
from .registry import active_peers


def update_case_slots(config, allocation, count):
    root = Path(config["work_root"]) / "resources"
    with FileLock(str(root / ".resource.lock")):
        path = root / "active.json"
        rows = active_peers(path)
        record = next(row for row in rows if same_process(row, allocation["owner"]))
        record["preprocess_slots"] = max(1, count)
        plan = allocation["supply"]
        record["preprocess_inline"] = plan["workers"][0] == 0
        fixed = plan["breakdown"]["processes_bytes"]
        case = plan["case_estimate_bytes"]
        record["ram_bytes"] = fixed + min(plan["workspace_capacity_bytes"], max(1, count) * case)
        _update_cpu(record)
        write_json(path, rows, durable=False)


def _update_cpu(record):
    claims = [row for row in record.get("case_cpu_claims", []) if alive(row)]
    record["case_cpu_claims"] = claims
    coordinator = 0 if record.get("preprocess_inline") else 1
    record["cpu_slots"] = coordinator + sum(row["threads"] for row in claims) + max(0, record.get("preprocess_slots", 1) - len(claims))


class CaseCPU:
    def __init__(self, config, owner, case_id):
        self.config, self.owner, self.case_id = config, owner, case_id
        self.member = identity()
        self.root = Path(config["work_root"]) / "resources"
        self.slots, self.fair = 1, 1
        from ..preprocessing.offline import _WORKER_HANDLE
        self.handle = config.get('resource_handle', _WORKER_HANDLE)

    def request(self, wanted):
        if self.handle is not None:
            self.slots = self.handle.state.get('case_slots', 1)
            self.fair = max(1, self.handle.state.get('cpu_slots', 1)//self.slots)
            cap = self.config['preprocessing']['max_case_threads']
            return min(wanted, self.fair, cap or self.fair)
        with FileLock(str(self.root / ".resource.lock")):
            path = self.root / "active.json"
            rows = active_peers(path)
            own = next(row for row in rows if same_process(row, self.owner))
            if not alive(own):
                raise RuntimeError("预处理协调进程已退出")
            if same_process(self.owner, self.member):
                own["preprocess_inline"] = True
            _update_cpu(own)
            claims = own["case_cpu_claims"]
            others = [row for row in claims if not (same_process(row, self.member) and row["case_id"] == self.case_id)]
            coordinator = 0 if own.get("preprocess_inline") else 1
            cores = max(0, max(1, math.floor(host_capacity()["cores"] * self.config["resources"]["cpu_fraction"])) -
                        sum(row["cpu_slots"] for row in rows if row is not own) - coordinator)
            slots = own.get("preprocess_slots", 1)
            fair = max(0, cores // slots)
            self.slots, self.fair = slots, fair
            available = max(0, cores - sum(row["threads"] for row in others) - max(0, slots - len(others) - 1))
            cap = self.config["preprocessing"]["max_case_threads"]
            granted = min(wanted, fair, available, cap or cores)
            own["case_cpu_claims"] = others + [{**self.member, "case_id": self.case_id, "threads": granted}]
            _update_cpu(own)
            write_json(path, rows, durable=False)
            return granted

    def shrink(self, granted):
        if self.handle is not None:
            return
        with FileLock(str(self.root / ".resource.lock")):
            path = self.root / "active.json"
            rows = active_peers(path)
            own = next(row for row in rows if same_process(row, self.owner))
            claim = next(row for row in own["case_cpu_claims"]
                         if same_process(row, self.member) and row["case_id"] == self.case_id)
            if not 0 <= granted <= claim["threads"]:
                raise ValueError("CPU额度只能在已获准范围内缩减")
            claim["threads"] = granted
            _update_cpu(own)
            write_json(path, rows, durable=False)

    def close(self):
        if self.handle is not None:
            return
        with FileLock(str(self.root / ".resource.lock")):
            path = self.root / "active.json"
            rows = active_peers(path)
            own = next(row for row in rows if same_process(row, self.owner))
            own["case_cpu_claims"] = [row for row in own.get("case_cpu_claims", [])
                if not (same_process(row, self.member) and row["case_id"] == self.case_id)]
            _update_cpu(own)
            write_json(path, rows, durable=False)
