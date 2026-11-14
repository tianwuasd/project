import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import shutil
import sys

import nibabel as nib
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import digest, file_digest, read_json, write_json
from chd_segmentation.config import load_config
from chd_segmentation.datasets.splits import apply_split, select_test
from chd_segmentation.datasets.transfer import bind_inputs, manifest_identity
from chd_segmentation.models.stunet.adapter import StunetAdapter, WEIGHT_SHA256
from chd_segmentation.prediction.testing import run_test
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.engine import checkpoint, code_identity


def run(args):
    source = Path(args.bundle).resolve(strict=True)
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        packet = workspace.root / "inputs"
        packet.mkdir()
        descriptor = read_json(source / "bundle.json")
        descriptor["datasets"] = {"chd1282": descriptor["datasets"]["chd1282"]}
        descriptor["experiments"] = [e for e in descriptor["experiments"] if e["config"]["dataset"] == "chd1282"]
        descriptor["immutable_files"] = {k: v for k, v in descriptor["immutable_files"].items() if k.startswith("chd1282_")}
        for name in descriptor["immutable_files"]:
            shutil.copyfile(source / name, packet / name)
        manifest = read_json(source / "chd1282_manifest.json")
        manifest["scope"] = "technical"
        write_json(packet / "chd1282_manifest.json", manifest)
        descriptor["datasets"]["chd1282"]["identity"] = manifest_identity(manifest)
        experiment = descriptor["experiments"][0]
        config = experiment["config"]
        config.update(scope="technical", plan=args.plan)
        config["planning"] = {"mode": "existing"}
        config["training"].update(patch=[32, 32, 32], precision="fp32")
        config["resources"].update(device="cpu", cpu_processes=1, gpu_count=1, workers=0,
                                   reserve_ram_bytes=1024 ** 3, cache_bytes=0)
        descriptor["identity"] = digest({k: v for k, v in descriptor.items() if k != "identity"})
        write_json(packet / "bundle.json", descriptor)
        bound = bind_inputs(packet, workspace.root, {"chd1282": args.data_root}, args.weights)
        again = bind_inputs(packet, workspace.root, {"chd1282": args.data_root}, args.weights)
        assert not again["changed_files"]
        config_path = packet / experiment["filename"]
        config = load_config(config_path)
        raw = read_json(packet / "chd1282_manifest.json")
        selected = apply_split(raw, config)
        assert Counter(row["source_role"] for row in selected["cases"]) == {
            "train": 590, "development": 128, "source_test": 127, "target_primary": 298, "target_secondary": 31}
        internal = select_test(selected, "internal")["cases"][0]
        external = select_test(selected, "external", "target_primary")["cases"][0]
        chosen = [internal, external]
        sizes = [nib.load(Path(raw["data_root"]) / row["image"]).shape for row in chosen]
        shape = [min(a) for a in zip(*sizes)]
        roi = [[int(n // 2 - 16), int(n // 2 + 16)] for n in shape]
        config["technical"].update(roi_xyz=roi, allow_role_overlap=False)
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        config = load_config(config_path)
        prepared = prepare(config, Path(config["work_root"]) / "prepared.json")
        spec = asdict(StunetAdapter.spec(config, prepared["plan"]))
        torch.set_num_threads(2)
        torch.manual_seed(config["training"]["seed"])
        model = StunetAdapter.build(spec)
        receipt = StunetAdapter.load_pretrained(model, args.weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.99, nesterov=True, weight_decay=3e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scientific = {"prepared_identity": prepared["identity"], "model": "stunet", "spec": spec,
                      "training": config["training"], "pretrained": WEIGHT_SHA256, "code_identity": code_identity(),
                      "versions": {n: version(n) for n in ("torch", "monai", "nnunetv2", "numpy", "scipy")}}
        identity = digest(scientific)
        run_dir = Path(config["run_dir"])
        run_dir.mkdir(parents=True)
        write_json(run_dir / "run.json", {"identity": identity, "scientific": scientific, "prepared": prepared, "config": config})
        progress = {"attempts": 0, "updates": 0, "skipped": 0, "epoch": 0, "phase": "train", "val_step": 0,
                    "val_sum": 0., "validation_loss": None}
        checkpoint(run_dir, model, optimizer, scaler, progress,
                   {"world": 1, "identity": identity, "spec": spec, "config": config}, receipt, torch.device("cpu"), 0)
        del model, optimizer, scaler
        record = {"dataset": "chd1282", "optimizer_updates": 0, "model": "official backbone with initialized segmentation head",
                  "roi_xyz": roi, "bound_inputs": bound, "frozen_role_counts": dict(Counter(r["source_role"] for r in selected["cases"])),
                  "secondary_external_metadata_cases": len(select_test(selected, "external", "target_secondary")["cases"]), "tests": []}
        workspace.results.append(record)
        for kind, row in zip(("internal", "external"), chosen):
            result = run_test(config, kind, case_ids=[row["case_id"]])
            reports = [Path(r["report"]) for r in result["groups"].values()]
            stamps = [p.stat().st_mtime_ns for p in reports]
            repeated = run_test(config, kind, case_ids=[row["case_id"]])
            assert result["new_predictions"] == 1 and repeated["reused_predictions"] == 1
            assert all(r["reused"] for r in repeated["groups"].values())
            assert [p.stat().st_mtime_ns for p in reports] == stamps
            actual_image, actual_label = [Path(raw["data_root"]) / row[k] for k in ("image", "label")]
            assert file_digest(actual_image) == row["sha256"]["image"] and file_digest(actual_label) == row["sha256"]["label"]
            record["tests"].append({"case_id": row["case_id"], "source_identity": row["sha256"],
                                    "first": result, "repeat": repeated, "metrics_reused_without_write": True})
        assert record["tests"][0]["first"]["checkpoint_sha256"] == record["tests"][1]["first"]["checkpoint_sha256"]
        record["source_unchanged"] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两例真实病例内部/外部测试验收；初始化模型，零训练更新")
    for name in ("bundle", "data-root", "weights", "plan", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
