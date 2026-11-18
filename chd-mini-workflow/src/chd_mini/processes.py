from dataclasses import dataclass

from joblib.externals.loky.backend.utils import kill_process_tree
import psutil


@dataclass
class ProcessHandle:
    process: psutil.Process

    @property
    def pid(self):
        return self.process.pid

    def join(self):
        _, alive = psutil.wait_procs([self.process], timeout=5)
        if alive:
            raise RuntimeError("本任务进程尚未退出")


def stop_owned(owned):
    table = {p.pid: p for p in psutil.process_iter(["create_time"], ad_value=None)}
    live = [p for pid, p in table.items() if pid in owned and p.info["create_time"] == owned[pid]]
    for proc in sorted(live, key=lambda p: p.info["create_time"], reverse=True):
        # loky负责进程自然退出与终止操作之间的竞争，出生身份由本工具核验。
        kill_process_tree(ProcessHandle(proc), use_psutil=True)
    _, alive = psutil.wait_procs(live, timeout=5)
    if alive:
        raise RuntimeError("本任务进程未退出，保留文件等待安全清理")
