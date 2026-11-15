import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest import TestCase

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.data.supply import BatchSupply, selection
from chd_segmentation.models.stunet.adapter import StunetAdapter
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.build_gate import BuildGate
from chd_segmentation.training.checkpoint import load_checkpoint
from chd_segmentation.training.engine import train
from chd_segmentation.training.resources import resource_lease


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify_events(events, settings, names, check):
    consumed = {}
    built = Counter()
    batches = Counter()
    contexts = defaultdict(set)
    expected = set(range(4))
    for row in events:
        kind, rank, index = row["event"], row.get("rank"), row.get("batch_index")
        if kind == "batch_enqueued":
            check.assertEqual(row["queue_capacity"], 1)
            check.assertEqual(row["occupied_batches"], 1)
            check.assertEqual(index, consumed.get(rank, -1) + 1)
            batches[(rank, index)] += 1
            for context in row["samples"]:
                phase = 0 if context["phase"] == "train" else 1
                predicted, _, _ = selection((names, settings, phase, context["sample_index"], context["global_slot"]))
                check.assertEqual(context["case_id"], predicted)
        elif kind == "case_build_started":
            check.assertEqual(index, consumed.get(rank, -1) + 1)
            allowance = row["budget"]
            if "active_workspace_bytes" in allowance:
                check.assertLessEqual(allowance["active_workspace_bytes"], allowance["workspace_limit_bytes"])
                check.assertLessEqual(allowance["active_cases"], allowance["reader_limit"])
        elif kind == "case_build_completed":
            built[(index, row["global_slot"])] += 1
            contexts[index].add(row["global_slot"])
        elif kind == "batch_consumed":
            check.assertEqual(index, consumed.get(rank, -1) + 1)
            consumed[rank] = index
            check.assertEqual(row["occupied_batches"], 0)
    check.assertEqual(set(contexts), expected)
    for index in expected:
        check.assertEqual(contexts[index], {0, 1, 2, 3})
        for rank in (0, 1):
            check.assertEqual(batches[(rank, index)], 1)
        for slot in range(4):
            check.assertEqual(built[(index, slot)], 1)
    check.assertEqual(consumed, {0: 3, 1: 3})
    waits = [row["elapsed_seconds"] for row in events if row["event"] == "stage_completed" and row.get("stage") == "batch_supply"]
    return {"global_batches": len(expected), "samples": sum(built.values()), "queue_capacity": 1,
            "maximum_supply_seconds": max(waits), "no_missing_or_duplicate_samples": True,
            "requested_samples_match_global_slot_rng": True}


def verify_transfer(config, prepared, workspace, check):
    config = deepcopy(config)
    config["run_dir"] = str(workspace.root / "course" / "transfer")
    config["resources"].update(device="cuda", gpu_count=1, min_gpus=None, workers=0, share=False, wait_seconds=0)
    if torch.cuda.is_initialized():
        raise RuntimeError("传输验证需要尚未初始化 CUDA 的协调进程")
    spec = asdict(StunetAdapter.spec(config, prepared["plan"]))
    with resource_lease(config, prepared, spec) as allocation:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = allocation["gpus"][0]["uuid"]
        try:
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
            torch.cuda.reset_peak_memory_stats(device)
            gate = BuildGate(mp.get_context("spawn"), allocation["supply"]["reader_slots"],
                             allocation["supply"]["workspace_capacity_bytes"], allocation["supply"]["case_estimates"])
            with BatchSupply(prepared, config, [0, 1], 0, allocation["supply"]["cache_bytes"][0], gate,
                             allocation["owner"]) as supply:
                supply.submit(["ct_1003"], 0, 0)
                with check.assertRaises(BufferError):
                    supply.submit(["ct_1003"], 0, 1)
                batch = supply.take_cpu()
                original = [batch["image"].clone(), batch["label"].clone()]
                with check.assertRaises(BufferError):
                    supply.submit(["ct_1003"], 0, 1)
                batch = supply.transfer(batch, device)
                check.assertTrue(torch.cuda.current_stream(device).query())
                check.assertTrue(torch.equal(batch["image"].cpu(), original[0]))
                check.assertTrue(torch.equal(batch["label"].cpu(), original[1]))
                torch.cuda.synchronize(device)
                del batch
                supply.mark_consumed()
                supply.submit(["ct_1003"], 0, 1)
                check.assertEqual(len(supply.queue), 1)
            gc.collect()
            result = {"gpu": allocation["gpus"][0], "copy_completed_before_return": True,
                      "buffer_released_after_device_synchronization": True,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                      "remaining_allocated_bytes": torch.cuda.memory_allocated(device), "optimizer_updates": 0}
            check.assertEqual(result["remaining_allocated_bytes"], 0)
            torch.cuda.empty_cache()
            shutil.copyfile(Path(config["run_dir"]) / "events.jsonl", workspace.report / "transfer-events.jsonl")
            return result
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def run(args):
    check = TestCase()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        manifest = read_json(args.manifest)
        rows = [deepcopy(row) for row in manifest["cases"] if row["case_id"] in ("ct_1003", "ct_1141")]
        check.assertEqual(len(rows), 2)
        for row in rows:
            row["roles"] = ["train", "validation"]
            for kind in ("image", "label"):
                row[kind] = str((Path(args.data_root) / row[kind]).resolve(strict=True))
        write_json(workspace.root / "manifest.json", {"dataset": "imagechd", "scope": "technical",
                   "data_root": str(Path(args.data_root).resolve()), "cases": rows})
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", scope="technical", manifest=str(workspace.root / "manifest.json"),
                      pretrained=str(Path(args.pretrained).resolve()), work_root=str(workspace.root / "course"),
                      run_dir=str(workspace.root / "course" / "training"), result_root=str(workspace.root / "result"))
        config["training"].update(patch=[32, 32, 32], batch=4, epochs=2, steps_per_epoch=1,
                                  validation_steps=1, save_every_attempts=1)
        config["planning"].update(target_spacing_xyz=[.30469, .30469, .75])
        config["resources"].update(cpu_processes=2, workers="auto", max_workers=4,
                                   cache_bytes=64 * 1024 ** 2, reserve_ram_bytes=1024 ** 3,
                                   collective_timeout_seconds=5, adapt_seconds=30)
        config["technical"].update(roi_xyz=[[240, 272], [240, 272], [87, 119]], allow_role_overlap=True)
        write_json(workspace.root / "config.json", config)
        config = load_config(workspace.root / "config.json")
        record = {"cases": [row["case_id"] for row in rows], "scope": "32³ 技术区域；两个 CPU/Gloo 进程",
                  "optimizer_update_limit": 2, "collective_timeout_seconds": 5}
        workspace.results.append(record)
        prepared_path = workspace.root / "course" / "prepared.json"
        mark = perf_counter()
        prepared = prepare(config, prepared_path)
        record["prepare_seconds"] = perf_counter() - mark
        mark = perf_counter()
        first = train(config, prepared_path, stop_after=1)
        check.assertEqual(first["updates"], 1)
        check.assertEqual(first["epoch"], 1)
        second = train(config, prepared_path, resume=True)
        check.assertEqual(second["updates"], 2)
        check.assertEqual(second["epoch"], 2)
        check.assertTrue(second["restored_exact"])
        state = load_checkpoint(config["run_dir"], second["identity"])
        check.assertEqual(state["sampler_rng"]["next"], 2)
        del state
        record.update(training_seconds=perf_counter() - mark, first=first, resume=second)
        record["updates_per_second"] = 2 / record["training_seconds"]
        events_path = Path(config["run_dir"]) / "events.jsonl"
        events = read_events(events_path)
        record["ordering"] = verify_events(events, config["training"], [row["case_id"] for row in prepared["manifest"]["cases"]], check)
        record["supply_wait_exceeded_collective_timeout"] = record["ordering"]["maximum_supply_seconds"] > 5
        check.assertTrue(record["supply_wait_exceeded_collective_timeout"])
        check.assertEqual(len(list(Path(config["run_dir"]).glob("state-*.pt"))), 1)
        shutil.copyfile(events_path, workspace.report / "events.jsonl")
        if args.cuda_transfer:
            record["cuda_transfer"] = verify_transfer(config, prepared, workspace, check)
        record["unverified"] = ["真实多卡 CUDA/NCCL", "服务器完整数据尺寸", "持续训练吞吐"]
        write_json(workspace.report / "ordered-supply.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两例真实病例、两次更新的按序供数及恢复验证")
    for name in ("manifest", "data-root", "pretrained", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cuda-transfer", action="store_true")
    run(parser.parse_args())
