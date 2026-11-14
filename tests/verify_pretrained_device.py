import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest import TestCase

import torch
from torch.multiprocessing.spawn import ProcessRaisedException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.models.stunet.adapter import StunetAdapter, WEIGHT_SHA256
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.engine import train
from chd_segmentation.training.resources import inventory


def run(args):
    check = TestCase()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        workspace.device = "cpu_and_cuda_weights_only"
        rows = inventory()
        eligible = [row for row in rows if not row["busy"] and row["free"] > 1024 ** 3]
        if not eligible:
            raise RuntimeError("本地没有可用于权重校验的空闲显卡")
        gpu = max(eligible, key=lambda row: row["free"])
        # 测试从未初始化 CUDA 的独立进程启动；遵守当前可见设备范围。
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        tokens = [token.strip() for token in visible.split(",")] if visible is not None else None
        ordinal = (next(i for i, token in enumerate(tokens) if token in (str(gpu["index"]), gpu["uuid"]))
                   if tokens is not None else gpu["index"])
        device = torch.device("cuda", ordinal)
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        observed = str(torch.cuda.get_device_properties(device).uuid)
        observed = observed if observed.startswith("GPU-") else "GPU-" + observed
        check.assertEqual(observed.lower(), gpu["uuid"].lower())
        record = {"scope": "单病例技术区域；CUDA 只加载与校验权重，无训练或推断",
                  "optimizer_updates": 0, "gpu": gpu, "torch": torch.__version__,
                  "official_weight_sha256": WEIGHT_SHA256}
        workspace.results.append(record)
        spec = {"num_classes": 8, "patch": [128, 128, 128],
                "strides": [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels": [[3, 3, 3]] * 6}
        mark = perf_counter()
        cpu = StunetAdapter.build(spec)
        record["cpu_receipt"] = StunetAdapter.load_pretrained(cpu, args.pretrained)
        record["cpu_load_seconds"] = perf_counter() - mark
        gpu_model = StunetAdapter.build(spec).to(device)
        heads = {name: value.detach().cpu().clone() for name, value in gpu_model.state_dict().items()
                 if name.startswith("seg_outputs.")}
        name = next(name for name in cpu.state_dict() if not name.startswith("seg_outputs."))
        with check.assertRaises(RuntimeError):
            torch.equal(gpu_model.state_dict()[name], cpu.state_dict()[name])
        record["original_cross_device_comparison_reproduced"] = True
        mark = perf_counter()
        record["cuda_receipt"] = StunetAdapter.load_pretrained(gpu_model, args.pretrained)
        torch.cuda.synchronize(device)
        record["cuda_load_verify_seconds"] = perf_counter() - mark
        own = gpu_model.state_dict()
        names = [name for name in own if not name.startswith("seg_outputs.")]
        check.assertTrue(all(torch.equal(own[name].detach().cpu(), cpu.state_dict()[name]) for name in names))
        check.assertTrue(all(torch.equal(own[name].detach().cpu(), value) for name, value in heads.items()))
        record.update(verified_backbone_tensors=len(names), untouched_head_tensors=len(heads),
                      peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device))
        del own, cpu, gpu_model, heads
        gc.collect()
        torch.cuda.empty_cache()
        record["remaining_allocated_vram_bytes"] = torch.cuda.memory_allocated(device)
        check.assertEqual(record["remaining_allocated_vram_bytes"], 0)

        manifest = read_json(args.manifest)
        row = deepcopy(next(row for row in manifest["cases"] if row["case_id"] == "ct_1003"))
        row["roles"] = ["train", "validation"]
        for kind in ("image", "label"):
            row[kind] = str((Path(args.data_root) / row[kind]).resolve(strict=True))
        write_json(workspace.root / "manifest.json", {"dataset": "imagechd", "scope": "technical",
                   "data_root": str(Path(args.data_root).resolve()), "cases": [row]})
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", scope="technical", manifest=str(workspace.root / "manifest.json"),
                      pretrained=str(Path(args.pretrained).resolve()), work_root=str(workspace.root / "course"),
                      run_dir=str(workspace.root / "course" / "training"), result_root=str(workspace.root / "result"))
        config["training"].update(patch=[32, 32, 32], batch=4, attempts=1)
        config["technical"].update(roi_xyz=[[240, 272], [240, 272], [87, 119]], allow_role_overlap=True)
        config["resources"].update(workers=0, max_workers=1, readers=1, reserve_ram_bytes=1024 ** 3)
        write_json(workspace.root / "config.json", config)
        config = load_config(workspace.root / "config.json")
        record.update(case_id=row["case_id"], roi_xyz=config["technical"]["roi_xyz"])
        prepared_path = workspace.root / "course" / "prepared.json"
        mark = perf_counter()
        prepare(config, prepared_path)
        record["prepare_seconds"] = perf_counter() - mark
        broken = deepcopy(config)
        # 使用真实目录作为错误权重输入，触发真实读取失败，留下真实初始化记录。
        broken["pretrained"] = str(Path(args.pretrained).resolve().parent)
        with check.assertRaises(ProcessRaisedException) as failure:
            train(broken, prepared_path, stop_after=0)
        record["actual_initialization_failure"] = str(failure.exception)
        run_dir = Path(config["run_dir"])
        check.assertTrue((run_dir / "run.json").is_file())
        check.assertFalse((run_dir / "latest.json").exists())
        mark = perf_counter()
        restarted = train(config, prepared_path, stop_after=0)
        record["restart_seconds"] = perf_counter() - mark
        check.assertEqual(restarted["updates"], 0)
        check.assertEqual(restarted["attempts"], 0)
        record["restarted"] = restarted
        with check.assertRaises(FileExistsError):
            train(config, prepared_path, stop_after=0)
        record["existing_checkpoint_protected"] = True
        record["resume"] = train(config, prepared_path, resume=True, stop_after=0)
        check.assertTrue(record["resume"]["restored_exact"])
        check.assertEqual(record["resume"]["updates"], 0)
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        check.assertTrue(any(row["event"] == "initialization_restarted" for row in events))
        shutil.copyfile(run_dir / "events.jsonl", workspace.report / "events.jsonl")
        write_json(workspace.report / "pretrained-device.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实 CPU/CUDA 预训练校验和零更新初始化重试")
    for name in ("manifest", "data-root", "pretrained", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
