import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter, sleep

import psutil


def wait_memory(required, fraction=.95):
    last = 0.
    while True:
        facts = psutil.virtual_memory()
        spare = int(facts.available - facts.total * (1 - fraction))
        if spare >= required:
            return
        if perf_counter() - last >= 15:
            print(json.dumps({"stage": "waiting_local_memory", "headroom_bytes": spare, "fraction": fraction,
                              "required_additional_bytes": required}), flush=True)
            last = perf_counter()
        sleep(1)


def retain_events(source, target, limit=512 * 1024):
    rows, size, execution = deque(), 0, []
    with source.open(encoding="utf-8") as incoming:
        for line in incoming:
            row = json.loads(line)
            if row["event"] == "preprocess_execution":
                execution.append(row)
            encoded = line.encode("utf-8")
            if len(encoded) > limit:
                raise ValueError("单个验收日志事件超过保留上限")
            rows.append(encoded)
            size += len(encoded)
            while size > limit:
                size -= len(rows.popleft())
    with target.open("wb") as outgoing:
        outgoing.writelines(rows)
    return execution


def cleanup_existing(args):
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.training.memory_guard import alive
    root = Path(args.work_root).resolve(strict=True)
    owner = read_json(root / "owner.json")
    if owner["nonce"] != args.owner_nonce or alive(owner):
        raise ValueError("清理要求本次归属标记完全一致且原进程已经退出")
    if any(alive(row) or any(alive(member) for member in row.get("members", []))
           for row in read_json(root / "course" / "resources" / "active.json")):
        raise ValueError("清理时仍有登记的测试进程存活")
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("测试工作区存在符号链接，拒绝清理")
    report = Path(args.report).resolve() / ("interrupted-" + owner["nonce"][:8])
    report.mkdir(parents=True, exist_ok=False)
    retain_events(root / "course" / "ImageCHD-STUNet" / "events.jsonl", report / "events.jsonl", 256 * 1024)
    files = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size} for p in root.rglob("*") if p.is_file()]
    receipt = {"state": "interrupted_before_training", "optimizer_updates": 0, "owner": owner,
               "work_root": str(root), "files": files, "bytes": sum(row["bytes"] for row in files),
               "reason": args.interruption_reason}
    write_json(report / "cleanup.json", receipt)
    shutil.rmtree(root)
    receipt["cleanup"] = {"deleted_files": len(files), "released_bytes": receipt["bytes"], "work_root_exists": root.exists()}
    write_json(report / "cleanup.json", receipt)
    print(json.dumps(receipt["cleanup"]))


def run(args):
    wait_memory(2 * 1024 ** 3, args.local_ram_fraction)
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from smoke import SmokeWorkspace
    from chd_segmentation.artifacts.store import digest, read_json, write_json
    from chd_segmentation.config import DEFAULTS, load_config
    from chd_segmentation.models.stunet.preprocess import StunetTransform
    from chd_segmentation.models.stunet.streaming import stream_case
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.preprocessing.offline_store import begin_case, case_directory, case_key, recipe, save_case, owned_remove
    from chd_segmentation.preprocessing.prepare import prepare
    from chd_segmentation.training.engine import train
    from chd_segmentation.training.engine import code_identity
    from chd_segmentation.training.memory_guard import MemoryGuard
    from chd_segmentation.training.host import host_capacity
    from verify_training_supply_real import verify_supply

    with SmokeWorkspace(args.work_root, args.report) as workspace:
        root = workspace.root
        rows = [deepcopy(r) for r in read_json(args.manifest)["cases"] if r["case_id"] in ("ct_1003", "ct_1141")]
        if len(rows) != 2:
            raise ValueError("必须且只使用两个既定真实病例")
        for row in rows:
            row["roles"] = ["train"] if row["case_id"] == "ct_1003" else ["validation"]
            for kind in ("image", "label"):
                row[kind] = str((Path(args.data_root) / row[kind]).resolve(strict=True))
        write_json(root / "manifest.json", {"dataset": "imagechd", "scope": "technical", "data_root": args.data_root, "cases": rows})
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", scope="technical", manifest=str(root / "manifest.json"), pretrained=args.pretrained,
                      work_root=str(root / "course"), run_dir=str(root / "course" / "ImageCHD-STUNet"), result_root=str(root / "result"))
        config["preprocessing"].update(root=str(root / "data"), block_bytes=2 * 1024 ** 2)
        config["training"].update(patch=[32] * 3, batch=4, epochs=2, steps_per_epoch=1, validation_steps=1, save_every_attempts=1)
        config["planning"].update(target_spacing_xyz=[.25, .25, .5], samples_per_case=1000, max_samples=2000)
        config["technical"].update(roi_xyz=[[240, 272], [240, 272], [87, 119]])
        config["resources"].update(cpu_processes=2, workers=0, max_workers=2, readers=1, max_readers=1,
                                   reserve_ram_bytes=args.reserve_mib * 1024 ** 2, cache_bytes=1024 ** 2,
                                   ram_fraction=args.local_ram_fraction)
        if args.adaptive:
            config["technical"].update(roi_xyz=[[240, 304], [240, 304], [87, 151]])
            config["preprocessing"].update(block_bytes=256 * 1024, max_block_bytes=8 * 1024 ** 2,
                                           max_case_threads=4, growth_seconds=.05)
        write_json(root / "config.json", config)
        config = load_config(root / "config.json")
        info = {"cases": [r["case_id"] for r in rows], "roi_xyz": config["technical"]["roi_xyz"],
                "numeric_comparison": [], "optimizer_update_limit": 2, "gpu_training": False,
                "local_technical_ram_fraction": args.local_ram_fraction, "formal_ram_fraction": .8,
                "adaptive": args.adaptive, "execution": []}
        workspace.results.append(info)
        began = perf_counter()
        source_path = root / "course" / "source.json"
        source = prepare(config, source_path)
        info["plan_seconds"] = perf_counter() - began
        for separate in (False, True):
            prepared = deepcopy(source)
            cfg = prepared["plan"]["configurations"]["3d_fullres"]
            for kind in ("data", "seg"):
                cfg[f"resampling_fn_{kind}_kwargs"]["force_separate_z"] = separate
            identity = digest(recipe(prepared, config))
            with MemoryGuard(config, "numerical_check") as guard:
                for row in prepared["manifest"]["cases"]:
                    key = case_key(row, identity)
                    directory = root / "comparison" / str(separate) / key[:16]
                    pending = begin_case(directory, key, identity)
                    with guard.admit(32 * 1024 ** 2, label="reference_32cube") as ready:
                        if ready is None:
                            raise MemoryError("真实数值比较等待内存余量")
                        before = StunetTransform(prepared, config["resources"]["reserve_ram_bytes"])(row)
                    mark = perf_counter()
                    after, blocks = stream_case(prepared, row, config, pending, guard)
                    try:
                        info["execution"].append({"case_id": row["case_id"], "separate_z": separate,
                                                  **blocks.execution.metrics()})
                        delta = float(np.max(np.abs(before["image"] - after["image"])))
                        np.testing.assert_array_equal(before["image"], after["image"])
                        np.testing.assert_array_equal(before["label"], after["label"])
                        for name in ("source_shape_xyz", "source_affine", "working_affine", "spacing_xyz", "roi_xyz"):
                            np.testing.assert_equal(before[name], after[name])
                        for name in ("shape_before_cropping", "shape_after_cropping_and_before_resampling", "bbox_used_for_cropping"):
                            np.testing.assert_equal(before["properties"][name], after["properties"][name])
                        for label, locations in before["properties"]["class_locations"].items():
                            np.testing.assert_array_equal(locations, after["properties"]["class_locations"][label])
                        info["numeric_comparison"].append({"case_id": row["case_id"], "separate_z": separate,
                            "max_abs_difference": delta, "image_bitwise_equal": bool(np.array_equal(before["image"], after["image"])),
                            "label_equal": True, "geometry_equal": True, "locations_equal": True, "seconds": perf_counter() - mark})
                        save_case(directory, key, identity, after, blocks)
                    finally:
                        blocks.release_handles()
                    del before, after
                    owned_remove(directory)
                    write_json(workspace.report / "streaming.json", info)
        began = perf_counter()
        ready_path = root / "course" / "ready.json"
        preprocess_config = deepcopy(config)
        if args.adaptive:
            preprocess_config["resources"].update(workers=2, readers=2, max_readers=2,
                cpu_fraction=min(.8, 5 / host_capacity()["cores"]))
        prepared = preprocess(preprocess_config, source_path, ready_path)
        info["preprocess_seconds"] = perf_counter() - began
        before = {str(p): p.stat().st_mtime_ns for p in Path(prepared["storage"]["root"]).rglob("*") if p.is_file() and p.name != "catalog.json"}
        repeated = preprocess(config, source_path, ready_path)
        assert repeated["identity"] == prepared["identity"]
        assert all(Path(p).stat().st_mtime_ns == value for p, value in before.items())
        progress = read_json(Path(config["run_dir"]) / "progress-preprocess.json")
        assert progress["done"] == progress["total"] == progress["reused"] == 2 and progress["new"] == 0
        info["reused_without_array_writes"] = True
        if args.adaptive:
            info["ordered_prefetch"] = verify_supply(config, prepared)
            assert any(row["peak_case_threads"] >= 2 for row in info["execution"])
            assert any(row["peak_job_bytes"] > config["preprocessing"]["block_bytes"] for row in info["execution"])
        if not args.numeric_only:
            wait_memory(4 * 1024 ** 3, args.local_ram_fraction)
            if args.external_training:
                request = {"state": "ready", "nonce": workspace.nonce, "code_identity": code_identity(),
                           "config": str(root / "config.json"), "prepared": str(ready_path)}
                write_json(root / "training-ready.json", request)
                print(json.dumps({"stage": "waiting_external_training", **request}), flush=True)
                while not (root / "training-finished.json").exists():
                    sleep(1)
                result = read_json(root / "training-finished.json")
                if result["nonce"] != workspace.nonce or result["code_identity"] != code_identity():
                    raise ValueError("外部训练回执与本次工作区或代码身份不同")
                info.update(result)
            else:
                mark = perf_counter()
                first = train(config, ready_path, stop_after=1)
                second = train(config, ready_path, resume=True)
                assert first["updates"] == 1 and second["updates"] == 2 and second["restored_exact"]
                info.update(training_seconds=perf_counter() - mark, first=first, resume=second)
        log_path = Path(config["run_dir"]) / "events.jsonl"
        info["offline_execution"] = retain_events(log_path, workspace.report / "events.jsonl")
        write_json(workspace.report / "streaming.json", info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "data-root", "pretrained", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--local-ram-fraction", type=float, default=.95)
    parser.add_argument("--reserve-mib", type=int, default=1024)
    parser.add_argument("--cleanup-existing", action="store_true")
    parser.add_argument("--owner-nonce")
    parser.add_argument("--external-training", action="store_true")
    parser.add_argument("--interruption-reason", default="调整独立本地技术配置后重新运行；正式任务配置保持原值")
    args = parser.parse_args()
    if not 0 < args.local_ram_fraction <= 1 or args.reserve_mib < 0:
        raise ValueError("本地技术配置内存参数无效")
    if args.cleanup_existing:
        cleanup_existing(args)
    else:
        run(args)
