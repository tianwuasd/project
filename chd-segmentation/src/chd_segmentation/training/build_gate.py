class BuildGate:
    def __init__(self, context, limit):
        self.condition = context.Condition()
        self.limit = context.Value("i", limit)
        self.active = context.Value("i", 0)

    def set_limit(self, value):
        if value < 1:
            raise ValueError("病例构建并发至少为 1")
        with self.condition:
            if self.active.value:
                raise RuntimeError("调整病例构建并发前必须停止供数")
            self.limit.value = value
            self.condition.notify_all()

    def __enter__(self):
        with self.condition:
            self.condition.wait_for(lambda: self.active.value < self.limit.value)
            self.active.value += 1

    def __exit__(self, *_):
        with self.condition:
            self.active.value -= 1
            self.condition.notify_all()
