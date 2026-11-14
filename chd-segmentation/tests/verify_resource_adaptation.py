import argparse
from copy import deepcopy
import json
import multiprocessing as mp
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
from chd_segmentation.data.supply import BatchSupply
from chd_segmentation.prediction.infer import predict
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.budget import cpu_plan
from chd_segmentation.training.build_gate import BuildGate
from chd_segmentation.training.checkpoint import load_checkpoint
from chd_segmentation.training.engine import rebalance_supply, train
from chd_segmentation.training.resources import host_capacity, resource_lease


def run(manifest_path, data_root, pretrained, root, report):
    check = TestCase()
    with SmokeWorkspace(root, report) as workspace:
        manifest = read_json(manifest_path)
        row = deepcopy(next(row for row in manifest["cases"] if row["case_id"] == "ct_1003"))
        row["roles"] = ["train", "validation"]
        for kind in ("image", "label"):
            row[kind] = str((Path(data_root) / row[kind]).resolve(strict=True))
        write_json(workspace.root / "manifest.json", {"dataset": "imagechd", "scope": "technical",
                   "data_root": str(Path(data_root).resolve()), "cases": [row]})
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", scope="technical", manifest=str(workspace.root / "manifest.json"),
                      pretrained=str(Path(pretrained).resolve()), work_root=str(workspace.root / "course"),
                      run_dir=str(workspace.root / "course" / "training"), result_root=str(workspace.root / "result"))
        config["training"].update(patch=[32, 32, 32], batch=4, epochs=2, steps_per_epoch=1,
                                  validation_steps=1, save_every_attempts=1, foreground_fraction=1.)
        config["technical"].update(roi_xyz=[[240, 272], [240, 272], [87, 119]], allow_role_overlap=True)
        config["resources"].update(workers=1, max_workers=1, readers=1, reserve_ram_bytes=1024 ** 3,
                                   cache_bytes=32 * 1024 ** 2, adapt_seconds=.01, growth_seconds=.01)
        write_json(workspace.root / "config.json", config)
        config = load_config(workspace.root / "config.json")
        record = {"case_id": row["case_id"], "roi_xyz": config["technical"]["roi_xyz"],
                  "optimizer_update_limit": 2, "global_batch": 4, "host": host_capacity()}
        workspace.results.append(record)
        mark = perf_counter()
        prepared_path = workspace.root / "course" / "prepared.json"
        prepared = prepare(config, prepared_path)
        record["prepare_seconds"] = perf_counter() - mark
        plan = cpu_plan(config, prepared, 2)
        check.assertTrue(plan["admitted"])
        check.assertLessEqual(plan["ram_bytes"], plan["budget_bytes"])
        check.assertEqual(plan["breakdown"]["cache_instances"], 2)
        record["two_rank_budget"] = plan
        constrained = deepcopy(config)
        constrained["resources"].update(reserve_ram_bytes=host_capacity()["available_bytes"] + 1024 ** 3,
                                         wait_seconds=0)
        check.assertFalse(cpu_plan(constrained, prepared, 2)["admitted"])
        with check.assertRaises(TimeoutError):
            with resource_lease(constrained, prepared, {}):
                raise AssertionError("不足的实际 RAM 预算不能准入")
        mark = perf_counter()
        with resource_lease(config, prepared, {}) as allocation:
            context = mp.get_context("spawn")
            gate = BuildGate(context, allocation["supply"]["reader_slots"])
            job = {**allocation, "config": config, "prepared": prepared}
            with BatchSupply(prepared, config, [0], 1, allocation["supply"]["cache_bytes"][0], gate,
                             allocation["owner"]) as supply:
                supply.submit([row["case_id"]], 0, 0)
                reference = supply.take(torch.device("cpu"))
                supply.submit([row["case_id"]], 0, 1)
                config["resources"]["workers"] = 0
                check.assertEqual(rebalance_supply(job, supply, gate, torch.device("cpu"), 0, context.Event()), "applied")
                check.assertEqual(supply.workers, 0)
                check.assertFalse(supply.queue)
                supply.submit([row["case_id"]], 0, 0)
                replay = supply.take(torch.device("cpu"))
                check.assertTrue(torch.equal(reference["image"], replay["image"]))
                check.assertTrue(torch.equal(reference["label"], replay["label"]))
                config["resources"]["workers"] = 1
                check.assertEqual(rebalance_supply(job, supply, gate, torch.device("cpu"), 0, context.Event()), "applied")
                check.assertEqual(supply.workers, 1)
                supply.submit([row["case_id"]], 0, 0)
                replay = supply.take(torch.device("cpu"))
                check.assertTrue(torch.equal(reference["image"], replay["image"]))
        record["supply_resize_seconds"] = perf_counter() - mark
        record["supply_resize"] = ["worker1_to0", "worker0_to1", "unconsumed_prefetch_replayed_exactly"]
        mark = perf_counter()
        first = train(config, prepared_path, stop_after=0)
        check.assertEqual(first["updates"], 0)
        second = train(config, prepared_path, resume=True, stop_after=1)
        check.assertTrue(second["restored_exact"])
        check.assertEqual(second["updates"], 1)
        check.assertEqual(second["epoch"], 1)
        config["resources"].update(cpu_processes=2, workers=0)
        third = train(config, prepared_path, resume=True)
        check.assertEqual(third["updates"], 2)
        check.assertEqual(third["epoch"], 2)
        check.assertTrue(third["restored_training_state"])
        check.assertFalse(third["restored_exact"])
        check.assertEqual(third["batch_layout"], [2, 2])
        state = load_checkpoint(config["run_dir"], third["identity"])
        check.assertEqual(len(state["rank_rng"]), 2)
        check.assertEqual(state["sampler_rng"]["next"], 2)
        check.assertEqual(len(list(Path(config["run_dir"]).glob("state-*.pt"))), 1)
        del state
        record.update(training_seconds=perf_counter() - mark, first=first, same_rank_resume=second,
                      changed_rank_resume=third)
        record["updates_per_second"] = 2 / record["training_seconds"]
        config["resources"].update(cpu_processes=1, workers=0)
        record["prediction"] = predict(config, workspace.root / "result" / "prediction")
        record["prediction_reuse"] = predict(config, workspace.root / "result" / "prediction")
        check.assertEqual(record["prediction_reuse"]["reused"], 1)
        events = [json.loads(line) for line in (Path(config["run_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        check.assertTrue(any(row["event"] == "rank_count_changed" for row in events))
        check.assertTrue(any(row["event"] == "case_geometry" for row in events))
        record["unverified"] = ["Linux 实际 cgroup 限额", "多张物理 GPU/NCCL 重组", "繁忙显卡共享", "全量吞吐"]
        shutil.copyfile(Path(config["run_dir"]) / "events.jsonl", workspace.report / "events.jsonl")
        write_json(workspace.report / "resource-adaptation.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一个真实 ImageCHD 病例、两次更新的资源与恢复短测")
    for name in ("manifest", "data-root", "pretrained", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.manifest, args.data_root, args.pretrained, args.work_root, args.report)
