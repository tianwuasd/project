import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.config import load_config
from chd_segmentation.preprocessing.prepare import load_prepared
from chd_segmentation.training.engine import code_identity, train


def observe(stop):
    process, peak = psutil.Process(), 0
    while not stop.wait(.2):
        rows = {p.info["pid"]: p.info for p in psutil.process_iter(["pid", "ppid", "memory_info"])}
        family, pending = {process.pid}, [process.pid]
        while pending:
            parent = pending.pop()
            children = {pid for pid, row in rows.items() if row["ppid"] == parent} - family
            family.update(children)
            pending.extend(children)
        peak = max(peak, sum(rows[pid]["memory_info"].rss for pid in family
                             if pid in rows and rows[pid]["memory_info"] is not None))
    return peak


def run(args):
    root = Path(args.work_root).resolve(strict=True)
    owner, request = read_json(root / "owner.json"), read_json(root / "training-ready.json")
    current = code_identity()
    if owner["nonce"] != args.owner_nonce or request["nonce"] != args.owner_nonce or request["code_identity"] != current:
        raise ValueError("训练工作区归属或源码身份不同")
    if (root / "training-finished.json").exists():
        raise ValueError("本次训练验收已经执行")
    config_path, prepared_path = Path(request["config"]).resolve(strict=True), Path(request["prepared"]).resolve(strict=True)
    if not config_path.is_relative_to(root) or not prepared_path.is_relative_to(root):
        raise ValueError("训练验收输入越出获准工作区")
    config, prepared = load_config(config_path), load_prepared(prepared_path)
    if (config["scope"] != "technical" or not 1 <= len(prepared["manifest"]["cases"]) <= 2 or
            config["training"]["epochs"] * config["training"]["steps_per_epoch"] != 2 or
            config["resources"]["device"] != "cpu" or config["resources"]["cpu_processes"] != 2):
        raise ValueError("训练验收仅允许两个CPU进程、最多两例和两次更新")
    stop = threading.Event()
    observer = ThreadPoolExecutor(max_workers=1)
    measurement = observer.submit(observe, stop)
    began = perf_counter()
    try:
        first = train(config, prepared_path, stop_after=1)
        second = train(config, prepared_path, resume=True)
        assert first["updates"] == 1 and second["updates"] == 2 and second["restored_exact"]
    finally:
        stop.set()
        observer.shutdown(wait=True)
        peak = measurement.result()
    result = {"state": "passed", "nonce": args.owner_nonce, "code_identity": current,
              "training_seconds": perf_counter() - began, "first": first, "resume": second,
              "training_peak_process_group_rss_bytes": peak, "peak_vram_bytes": None}
    # 完整训练组已经关闭；工作区由持有同一归属标记的验收父进程统一清理。
    write_json(root / "training-finished.json", result)
    print({"state": result["state"], "updates": second["updates"], "restored_exact": second["restored_exact"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("work-root", "owner-nonce", "report"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
