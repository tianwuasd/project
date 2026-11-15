import argparse
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import sys
from time import sleep
import unittest

import psutil


def orphan_worker(config, owner, connection):
    from chd_segmentation.training.engine import require_owner
    from chd_segmentation.training.memory_guard import MemoryGuard, identity
    with MemoryGuard(config, "orphan_worker", owner=owner) as guard:
        connection.send({"owner": owner, "worker": identity()})
        if connection.recv() != "owner_stopped":
            raise ValueError("真实进程生命周期测试收到未知指令")
        facts = guard.boundary()
        assert facts["paused"] and facts["stop_requested"] and facts["stop_reason"] == "owner_exited"
        with guard.admit(1, wait=False) as ticket:
            assert ticket is None
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "协调进程"):
            require_owner({"owner": owner})
        from chd_segmentation.models.supply import CpuReady
        ready = CpuReady(mp.get_context('spawn'), 2, config['run_dir'], owner=owner)
        with unittest.TestCase().assertRaisesRegex(RuntimeError, '协调进程'):
            ready.wait(0, 'owner_already_exited')
    connection.send({"owner_exit_stopped_admission": True, "host_wait_refuses_missing_owner": True})


def supervisor(config, connection):
    from chd_segmentation.training.memory_guard import MemoryGuard, identity
    with MemoryGuard(config, "supervisor"):
        child = mp.get_context("spawn").Process(target=orphan_worker, args=(config, identity(), connection))
        child.start()
        sleep(120)
        raise TimeoutError("真实 owner 生命周期测试未按时结束")


def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from smoke import SmokeWorkspace
    from chd_segmentation.config import DEFAULTS
    from chd_segmentation.training.memory_guard import MemoryGuard, alive

    with SmokeWorkspace(args.work_root, args.report) as workspace:
        config = deepcopy(DEFAULTS)
        config.update(scope="technical", work_root=str(workspace.root / "course"),
                      run_dir=str(workspace.root / "course" / "reserve"))
        config["resources"].update(ram_fraction=.95, reserve_ram_bytes=1024 ** 3)
        reserved = 4 * 1024 ** 2
        with MemoryGuard(config, "checkpoint_reserve") as guard:
            with guard.admit(reserved, label="checkpoint_reserve", reserve=True) as ticket:
                assert ticket is not None
                payload = bytearray(b"\x01") * (8 * 1024 ** 2)
                facts = guard.boundary()
                assert facts["budget"]["pending_bytes"] >= reserved
                workspace.results.append({"checkpoint_reserved_bytes": reserved,
                    "actual_pending_bytes_after_resident_growth": facts["budget"]["pending_bytes"],
                    "actual_uss_bytes": facts["budget"]["own"]["uss_bytes"], "optimizer_updates": 0})
                del payload
        config["run_dir"] = str(workspace.root / "course" / "orphan")
        context = mp.get_context("spawn")
        incoming, outgoing = context.Pipe()
        process = context.Process(target=supervisor, args=(config, outgoing))
        worker = None
        try:
            process.start()
            outgoing.close()
            if not incoming.poll(60):
                raise TimeoutError("真实 worker 未及时完成登记")
            record = incoming.recv()
            worker = record["worker"]
            process.terminate()
            process.join(15)
            assert not process.is_alive()
            incoming.send("owner_stopped")
            if not incoming.poll(30):
                raise TimeoutError("owner 结束后 worker 未停止新增准入")
            workspace.results.append(incoming.recv())
            if alive(worker):
                psutil.Process(worker["pid"]).wait(timeout=15)
            assert not alive(worker)
        finally:
            incoming.close()
            if process.is_alive():
                process.terminate()
                process.join(15)
            if worker is not None and alive(worker):
                child = psutil.Process(worker["pid"])
                child.terminate()
                child.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--report", required=True)
    run(parser.parse_args())
