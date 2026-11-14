import _thread
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import threading
from time import monotonic
import traceback
from unittest import TestCase
from uuid import uuid4

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.cli import wait_seconds
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.training.resources import active_peers, choose_gpus, gpu_decision, inventory, resource_lease


def verify(root, report):
    root, report = Path(root).resolve(), Path(report).resolve()
    if root == report or root.is_relative_to(report) or report.is_relative_to(root):
        raise ValueError("测试工作区与报告必须独立")
    root.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    owner, started, check = uuid4().hex, monotonic(), TestCase()
    write_json(root / "owner.json", {"nonce": owner})
    result = {"state": "failed", "scope": "本机真实显卡状态与独立资源登记；不加载模型、不执行 GPU 计算",
              "optimizer_updates": 0, "scenarios": []}
    try:
        sample = inventory()
        if not sample:
            raise ValueError("本机没有可见的 NVIDIA 显卡，无法执行真实调度短测")
        result["observed_gpus"] = sample
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", manifest=str(root / "unused-manifest.json"),
                      pretrained=str(root / "unused.model"), work_root=str(root), run_dir=str(root / "logs"))
        config["training"]["batch"] = 1
        config["resources"].update(device="cuda", gpu_count=1)
        path = root / "config.json"
        write_json(path, config)
        config = load_config(path)
        check.assertIsNone(config["resources"]["wait_seconds"])
        check.assertIsNone(load_config(path, {"resources": {"wait_seconds": wait_seconds("forever")}})["resources"]["wait_seconds"])
        check.assertEqual(load_config(path, {"resources": {"wait_seconds": wait_seconds("2.5")}})["resources"]["wait_seconds"], 2.5)
        for invalid in (-1, float("inf"), float("nan")):
            with check.assertRaises(ValueError):
                load_config(path, {"resources": {"wait_seconds": invalid}})
        history = [(monotonic(), sample)]
        decision = gpu_decision(config, {}, history, {}, [])
        idle = [g for g in sample if not g["busy"] and g["free"] >= max(
            config["resources"]["vram_reserve_bytes"], g["total"] * config["resources"]["vram_reserve_fraction"])]
        if idle:
            check.assertEqual(decision["gpus"][0]["uuid"], max(idle, key=lambda g: g["free"])["uuid"])
            manual = deepcopy(config)
            manual["resources"]["gpu_ids"] = [idle[0]["uuid"]]
            check.assertEqual(choose_gpus(manual, {}, history, {}, [])[0]["uuid"], idle[0]["uuid"])
        else:
            check.assertIsNone(decision["gpus"])
        # 登记当前真实进程对这些真实设备的预留；只作用于独立测试目录。
        process = psutil.Process()
        reserved = {"pid": process.pid, "birth": process.create_time(), "uuids": [g["uuid"] for g in sample],
                    "cpu_slots": 0, "ram_bytes": 0}
        registry = root / "resources" / "active.json"
        write_json(registry, [reserved])
        check.assertIsNone(choose_gpus(config, {}, history, {}, active_peers(registry)))
        finite = deepcopy(config)
        finite["resources"]["wait_seconds"] = 0
        mark = monotonic()
        with check.assertRaises(TimeoutError):
            with resource_lease(finite, {}, {}):
                raise AssertionError("已有真实进程预留时不应获得设备")
        result["scenarios"].append({"name": "finite_wait", "seconds": monotonic() - mark, "result": "passed"})
        unlimited = deepcopy(config)
        unlimited["resources"].update(wait_seconds=None, poll_seconds=.05)
        timer = threading.Timer(.8, _thread.interrupt_main)
        mark = monotonic()
        timer.start()
        try:
            with check.assertRaises(KeyboardInterrupt):
                with resource_lease(unlimited, {}, {}):
                    raise AssertionError("等待期间不应越过真实设备预留")
        finally:
            timer.cancel()
            timer.join()
        result["scenarios"].append({"name": "unlimited_wait_and_cancel", "seconds": monotonic() - mark,
                                    "result": "passed"})
        check.assertEqual(read_json(registry), [reserved])
        write_json(registry, [])
        check.assertEqual(active_peers(registry), [])
        check.assertEqual(choose_gpus(config, {}, history, {}, []), decision["gpus"])
        logs = root / "logs" / "events.jsonl"
        events = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
        waiting = [row for row in events if row["event"] == "waiting_resources"]
        check.assertTrue(any(row["waiting_forever"] for row in waiting))
        check.assertTrue(all(row["candidates"] for row in waiting))
        check.assertTrue(all("已预留" in row["candidates"][0]["ranks"]["0"]["reason"] for row in waiting))
        shutil.copyfile(logs, report / "events.jsonl")
        result.update(state="passed", checks=["actual_idle_selection", "manual_uuid", "live_process_reservation",
                      "finite_timeout", "unlimited_wait", "manual_cancel", "release", "candidate_diagnostics",
                      "null_and_cli_forever", "invalid_wait_values"],
                      unverified=["多张物理显卡的混合分配", "真实繁忙卡共享容量", "CUDA/NCCL 训练"])
    finally:
        exception_type, exception, trace = sys.exc_info()
        if exception is not None:
            result["error"] = str(exception)
            result["traceback"] = "".join(traceback.format_exception(exception_type, exception, trace))
        if read_json(root / "owner.json") != {"nonce": owner} or root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
            raise ValueError("测试目录归属或链接检查失败，停止清理")
        files = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in root.rglob("*") if p.is_file()]
        result.update(seconds=monotonic() - started, artifacts=files, artifact_count=len(files),
                      artifact_bytes=sum(p["bytes"] for p in files), rss_end_bytes=psutil.Process().memory_info().rss,
                      peak_rss_bytes=None, peak_vram_bytes=None)
        result["scenarios_per_second"] = len(result["scenarios"]) / result["seconds"]
        write_json(report / "acceptance.json", result)
        shutil.rmtree(root)
        result["cleanup"] = {"deleted_files": len(files), "released_bytes": result["artifact_bytes"], "root_exists": root.exists()}
        write_json(report / "acceptance.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实显卡状态下的持续等待与取消短测，不执行 GPU 计算")
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    verify(args.work_root, args.report)
