import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest import TestCase

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import digest, read_json, write_json
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.data.dataset import CaseDataset, sample_patch
from chd_segmentation.prediction.infer import predict
from chd_segmentation.prediction.metrics import evaluate
from chd_segmentation.preprocessing.offline import data_info, preprocess
from chd_segmentation.preprocessing.offline_store import case_directory, recipe, storage_lease, validate_storage
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.engine import train


def run(args):
    check = TestCase()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        root = workspace.root
        rows = [deepcopy(r) for r in read_json(args.manifest)["cases"] if r["case_id"] in ("ct_1003", "ct_1141")]
        check.assertEqual(len(rows), 2)
        for row in rows:
            row["roles"] = ["train"] if row["case_id"] == "ct_1003" else ["validation"]
            for kind in ("image", "label"):
                row[kind] = str((Path(args.data_root) / row[kind]).resolve(strict=True))
        write_json(root / "manifest.json", {"dataset": "imagechd", "scope": "technical",
                   "data_root": str(Path(args.data_root).resolve()), "cases": rows})
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", scope="technical", manifest=str(root / "manifest.json"),
                      pretrained=args.pretrained, work_root=str(root / "course"), run_dir=str(root / "course" / "training"),
                      result_root=str(root / "result"))
        config["preprocessing"]["root"] = str(root / "data" / "preprocessed")
        config["training"].update(patch=[32] * 3, batch=4, epochs=2, steps_per_epoch=1, validation_steps=1, save_every_attempts=1)
        config["planning"].update(target_spacing_xyz=[.30469, .30469, .75], samples_per_case=1000, max_samples=2000)
        config["technical"].update(roi_xyz=[[240, 272], [240, 272], [87, 119]])
        config["resources"].update(cpu_processes=2, workers="auto", max_workers=2, max_readers=2,
                                   reserve_ram_bytes=1024 ** 3, cache_bytes=16 * 1024 ** 2)
        write_json(root / "config.json", config)
        config = load_config(root / "config.json")
        source_path, ready_path = root / "course" / "source.json", root / "course" / "ready.json"
        record = {"cases": [r["case_id"] for r in rows], "roi_xyz": config["technical"]["roi_xyz"],
                  "scope": "两个真实病例；32³；两个 CPU/Gloo 进程；合计两次更新", "gpu_training": False}
        workspace.results.append(record)
        mark = perf_counter()
        source = prepare(config, source_path)
        check.assertEqual(source["plan"]["fit_ids"], ["ct_1003"])
        record["plan_seconds"] = perf_counter() - mark
        with check.assertRaises(ValueError):
            validate_storage(source)
        mark = perf_counter()
        prepared = preprocess(config, source_path, ready_path)
        record["preprocess_seconds"] = perf_counter() - mark
        version = Path(prepared["storage"]["root"])
        stamps = {str(p): p.stat().st_mtime_ns for p in version.rglob("*") if p.is_file() and p.name != "catalog.json"}
        mark = perf_counter()
        check.assertEqual(preprocess(config, source_path, ready_path), prepared)
        check.assertTrue(all(Path(p).stat().st_mtime_ns == value for p, value in stamps.items()))
        record["reuse_seconds"] = perf_counter() - mark
        # 删除本轮自有的一份数组，实际验证仅补缺失病例，保留同配方身份。
        missing = case_directory(version, prepared["storage"]["cases"]["ct_1003"]["key"]) / "image.npy"
        missing.unlink()
        with check.assertRaises(ValueError):
            validate_storage(prepared)
        mark = perf_counter()
        rebuilt = preprocess(config, source_path, ready_path)
        check.assertEqual(rebuilt["identity"], prepared["identity"])
        record["repair_seconds"] = perf_counter() - mark
        different = deepcopy(config)
        different["preprocessing"]["statistics_source"] = "q-stage-a"
        with check.assertRaises(ValueError):
            validate_storage(prepared, config=different)
        check.assertNotEqual(digest(recipe(source, config)), digest(recipe(source, different)))
        changed = deepcopy(config)
        changed["training"]["epochs"] = 200
        changed["resources"]["gpu_count"] = 4
        check.assertEqual(recipe(source, config), recipe(source, changed))
        dataset = CaseDataset(prepared, config["resources"])
        entry = dataset.get("ct_1003")
        check.assertIsInstance(entry["image"], np.memmap)
        check.assertFalse(entry["image"].flags.writeable)
        a = sample_patch(entry, [32] * 3, np.random.default_rng(5), .33)
        b = sample_patch(dataset.get("ct_1003"), [32] * 3, np.random.default_rng(5), .33)
        check.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        del entry, a, b
        dataset.clear()
        with storage_lease(prepared, config):
            check.assertTrue(data_info(ready_path)["cleanup_blocked"])
        check.assertFalse(data_info(ready_path)["cleanup_blocked"])
        record["offline_bytes"] = data_info(ready_path)["bytes"]
        if args.preprocess_only:
            event_path = Path(config["run_dir"]) / "events.jsonl"
            shutil.copyfile(event_path, workspace.report / "events.jsonl")
            record.update(optimizer_updates=0, coverage="离线全局资源登记、真实预处理、复用、缺失修补、引用与 mmap")
            write_json(workspace.report / "offline-preprocess-only.json", record)
            return
        events_before = len((Path(config["run_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines())
        mark = perf_counter()
        first = train(config, ready_path, stop_after=0 if args.zero_update else 1)
        second = train(config, ready_path, resume=True, stop_after=0 if args.zero_update else None)
        check.assertEqual((first["updates"], second["updates"]), (0, 0) if args.zero_update else (1, 2))
        check.assertTrue(second["restored_exact"])
        record.update(training_seconds=perf_counter() - mark, first=first, resume=second)
        mark = perf_counter()
        record["prediction"] = predict(config, root / "result" / "predictions")
        record["prediction_seconds"] = perf_counter() - mark
        check.assertEqual(predict(config, root / "result" / "predictions")["reused"], 2)
        mark = perf_counter()
        record["evaluation"] = evaluate(root / "result" / "predictions", prepared, root / "result" / "metrics.json", config)
        record["evaluation_seconds"] = perf_counter() - mark
        event_path = Path(config["run_dir"]) / "events.jsonl"
        events = [json.loads(s) for s in event_path.read_text(encoding="utf-8").splitlines()]
        check.assertFalse(any(e.get("stage") in ("crop_normalize_resample", "read_decompress") for e in events[events_before:]))
        completed = [e for e in events if e["event"] == "preprocess_case_completed"]
        check.assertEqual(len(completed), 3)
        record["preprocess_cases_total_including_repair"] = 3
        record["training_and_prediction_resample_calls"] = 0
        record["updates_per_second"] = (0 if args.zero_update else 2) / record["training_seconds"]
        record["optimizer_updates_in_this_run"] = 0 if args.zero_update else 2
        record["cases_per_second"] = 2 / record["preprocess_seconds"]
        record["references"] = data_info(ready_path)["references"]
        # 仿真工具只读导入；虚拟 GPU 仅用于生产选卡规则及服务器路径映射。
        sys.path.insert(0, args.simulator)
        from simulator import decision, resolve_path
        from chd_segmentation.training.resources import capacity_key, gpu_decision
        profile = read_json(Path(args.simulator) / "profiles" / "zmic44.json")
        state = {"mappings": {"/data4/zhangruichen/data": str(root / "data")}, "jobs": [],
                 "cpu_slots": 256, "ram_bytes": 251 * 2 ** 30,
                 "gpus": [{"index": i, "uuid": f"SIM-GPU-{i:02d}", "total": 24 * 2 ** 30,
                           "free": 24 * 2 ** 30, "busy": i < 6, "owned": False, "utilization": 0} for i in range(8)]}
        chosen = decision(state, {"gpus": 2, "vram_per_rank_bytes": 2 ** 30, "cpu_slots": 4, "ram_bytes": 4 * 2 ** 30},
                          (DEFAULTS, gpu_decision, capacity_key))
        check.assertTrue(chosen["admitted"])
        check.assertEqual([g["index"] for g in chosen["gpus"]], [6, 7])
        check.assertEqual(resolve_path("/data4/zhangruichen/data/preprocessed", state), str(root / "data" / "preprocessed"))
        record["simulation"] = {"source": args.simulator, "profile": profile, "decision": chosen,
                                "actual_compute": "CPU/Gloo", "virtual_gpu_compute": False}
        # 第二个技术折单独拟合另一个主训练病例，只核对计划与身份，不训练。
        swapped = deepcopy(rows)
        for row in swapped:
            row["roles"] = ["train"] if row["case_id"] == "ct_1141" else ["validation"]
        write_json(root / "fold1-manifest.json", {"dataset": "imagechd", "scope": "technical",
                   "data_root": args.data_root, "cases": swapped})
        fold = deepcopy(config)
        fold.update(fold=1, manifest=str(root / "fold1-manifest.json"), plan=str(root / "course" / "fold1-plan.json"))
        next_source = prepare(fold, root / "course" / "fold1-source.json")
        check.assertEqual(next_source["plan"]["fit_ids"], ["ct_1141"])
        check.assertNotEqual(digest(recipe(source, config)), digest(recipe(next_source, fold)))
        record["fold_statistics_isolated"] = True
        record["unverified"] = ["Linux cgroup 与真实双卡 NCCL", "128³ CUDA 训练吞吐", "全量预处理容量及速度"]
        shutil.copyfile(event_path, workspace.report / "events.jsonl")
        write_json(workspace.report / "offline.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两例真实病例、两次更新的离线预处理完整短测")
    for name in ("manifest", "data-root", "pretrained", "work-root", "report", "simulator"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--zero-update", action="store_true")
    run(parser.parse_args())
