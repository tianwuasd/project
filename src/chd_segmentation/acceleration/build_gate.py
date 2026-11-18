from contextlib import contextmanager


class BuildGate:
    def __init__(self, context, limit, capacity_bytes=None, estimates=()):
        self.condition = context.Condition()
        self.limit = context.Value("i", limit)
        self.active = context.Value("i", 0)
        self.capacity = context.Value("q", capacity_bytes or 0)
        self.used = context.Value("q", 0)
        self.estimates = {row["case_id"]: row["workspace_bytes"] for row in estimates}

    def set_limit(self, value, capacity_bytes=None):
        if value < 1:
            raise ValueError("病例构建并发至少为 1")
        with self.condition:
            if self.active.value:
                raise RuntimeError("调整病例构建并发前必须停止供数")
            self.limit.value = value
            if capacity_bytes is not None:
                if capacity_bytes < 1:
                    raise ValueError("病例构建内存预算必须为正数")
                self.capacity.value = capacity_bytes
            self.condition.notify_all()

    def acquire(self, amount):
        with self.condition:
            if amount and amount > self.capacity.value:
                raise MemoryError("单病例工作空间超出当前构建预算")
            self.condition.wait_for(lambda: self.active.value < self.limit.value and
                (not amount or self.used.value + amount <= self.capacity.value))
            self.active.value += 1
            self.used.value += amount
            return {"active_cases": self.active.value, "active_workspace_bytes": self.used.value,
                    "reader_limit": self.limit.value, "workspace_limit_bytes": self.capacity.value,
                    "case_estimate_bytes": amount}

    def release(self, amount):
        with self.condition:
            self.active.value -= 1
            self.used.value -= amount
            if self.active.value < 0 or self.used.value < 0:
                raise RuntimeError("病例构建额度发生重复释放")
            self.condition.notify_all()

    @contextmanager
    def case(self, name):
        amount = self.estimates[name]
        snapshot = self.acquire(amount)
        try:
            yield snapshot
        finally:
            self.release(amount)

    def __enter__(self):
        return self.acquire(0)

    def __exit__(self, *_):
        self.release(0)
