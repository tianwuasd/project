import os

import psutil


class HostCoordinator:
    PHASES = {name: number for number, name in enumerate(("startup_ready", "model_ready", "data_ready", "transfer_ready",
        "consumed", "pool_closed", "pool_applied", "checkpoint_saved", "summary_saved", "memory_boundary", "memory_admitted",
        "group_memory_ready", "group_memory_released"))}

    def __init__(self, context, world, cancel=None):
        self.world = world
        self.cancel = cancel
        self.condition = context.Condition()
        self.generation = context.Value("q", 0)
        self.phase = context.Value("i", -1)
        self.batch = context.Value("q", -1)
        self.arrived = context.Array("b", world)
        self.pids = context.Array("q", world)
        self.births = context.Array("d", world)

    def register(self, rank):
        with self.condition:
            if self.pids[rank]:
                raise RuntimeError("训练进程重复登记主机协调身份")
            self.pids[rank] = os.getpid()
            self.births[rank] = psutil.Process().create_time()

    def wait(self, rank, batch, phase, update=None):
        code = self.PHASES[phase]
        with self.condition:
            generation = self.generation.value
            if not any(self.arrived):
                self.batch.value, self.phase.value = batch, code
            if self.batch.value != batch or self.phase.value != code or self.arrived[rank]:
                raise RuntimeError("各 rank 的批次或主机协调阶段不一致")
            self.arrived[rank] = 1
            if all(self.arrived):
                self.generation.value += 1
                for index in range(self.world):
                    self.arrived[index] = 0
                self.condition.notify_all()
            else:
                while generation == self.generation.value:
                    if self.cancel is not None and self.cancel.is_set():
                        raise InterruptedError('主机协调等待已取消')
                    if update is not None:
                        update(ready_ranks=[i for i, value in enumerate(self.arrived) if value],
                               world_size=self.world)
                    for index, pid in enumerate(self.pids):
                        if not pid and phase == "startup_ready":
                            continue
                        if not pid or not psutil.pid_exists(pid) or psutil.Process(pid).create_time() != self.births[index]:
                            raise RuntimeError("主机等待期间训练进程已经退出或身份改变")
                    self.condition.wait(.5)
        if update is not None:
            update(ready_ranks=list(range(self.world)), world_size=self.world)
