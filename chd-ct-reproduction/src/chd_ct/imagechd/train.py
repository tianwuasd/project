"""Six-stage training from reusable caches; no raw-image preprocessing."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..paths import config_path
from .augment import augment_batch
from .common import LABELS, MODEL_FORMAT, NORMALIZATION, load_case, read_prepared, write_json
from .config import STAGES, UNAVAILABLE, build_model, load_config, weights_for
from .dataset import StageDataset
from .losses import masked_loss


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_rows(data):
    if data.get("for_prediction"):
        raise ValueError("无标签预测缓存不能用于训练")
    patients = {}
    for row in data["cases"]:
        patient, split = row.get("patient_id"), row.get("split")
        if not patient or split not in {"train", "val", "test"}:
            raise ValueError("缺少有效患者划分")
        if patients.setdefault(patient, split) != split:
            raise ValueError("同一患者跨数据划分")
    rows = {split: [r for r in data["cases"] if r["split"] == split] for split in ("train", "val")}
    if not all(rows.values()):
        raise ValueError("train/val 必须非空")
    return rows


def recurrent_features(encoder, sequence):
    with torch.no_grad():
        return torch.stack([encoder.features(sequence[:, i]) for i in range(sequence.shape[1])], dim=1)


def train_stage(directory, rows, stage, config, output, mode, manifest_hash):
    spec = config["stages"][stage]
    model = build_model(stage, spec).to(config["device"])
    loaders = {
        s: DataLoader(
            StageDataset(directory, rs, stage, config),
            batch_size=spec["batch"],
            shuffle=s == "train",
            num_workers=0,
        )
        for s, rs in rows.items()
    }
    encoder, encoder_record = None, None
    if stage == "blood_lstm":
        encoder_record = torch.load(output / "blood2d.pt", map_location="cpu", weights_only=True)
        encoder = build_model("blood2d", config["stages"]["blood2d"])
        encoder.load_state_dict(encoder_record["state_dict"])
        encoder.to(config["device"]).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    history, best = [], float("inf")
    maximum = 1 if mode in {"smoke", "preflight"} else None
    epochs = 1 if maximum else spec["epochs"]
    for epoch in range(epochs):
        fraction = epoch / epochs
        lr = config["learning_rate"] * (1 if fraction < 0.5 else 0.1 if fraction < 0.75 else 0.01)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses, samples = {}, {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            for step, (image, target) in enumerate(loader):
                if maximum is not None and step >= maximum:
                    break
                image, target = image.to(config["device"]), target.to(config["device"])
                if split == "train" and config.get("augment"):
                    image, target = augment_batch(image, target)
                with torch.set_grad_enabled(split == "train"):
                    features = recurrent_features(encoder, image) if encoder is not None else image
                    loss = masked_loss(model(features), target, weights_for(stage))
                    if not torch.isfinite(loss):
                        raise RuntimeError("损失非有限值")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach()) * len(image)
                count += len(image)
            if not count:
                raise ValueError("阶段没有可用样本")
            losses[split], samples[split] = total / count, count
        history.append({"epoch": epoch + 1, "learning_rate": lr, "loss": losses, "samples_seen": samples})
        print(json.dumps({"stage": stage, **history[-1]}), flush=True)
        write_json(output / (stage + ".history.json"), history)
        if losses["val"] < best:
            best = losses["val"]
            record = {
                "format": MODEL_FORMAT,
                "stage": stage,
                "labels": list(LABELS),
                "normalization": NORMALIZATION,
                "config": config,
                "spec": spec,
                "mode": mode,
                "prepared_sha256": manifest_hash,
                "epoch": epoch + 1,
                "val_loss": best,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            if encoder_record is not None:
                record.update(
                    encoder_state_dict=encoder_record["state_dict"],
                    encoder_spec=config["stages"]["blood2d"],
                    encoder_sha256=file_hash(output / "blood2d.pt"),
                )
            temporary = output / (stage + ".tmp.pt")
            torch.save(record, temporary)
            temporary.replace(output / (stage + ".pt"))
    return {"file": stage + ".pt", "sha256": file_hash(output / (stage + ".pt")), "best_val_loss": best}


def train(prepared, output, config_path, mode="smoke", device="cpu"):
    directory, data = read_prepared(prepared)
    config = load_config(config_path)
    rows = training_rows(data)
    if config.get("profile") == "paper-scale" and device == "cpu" and mode not in {"check", "smoke"}:
        raise ValueError("paper-scale 需要 GPU 预检；CPU 仅支持 check/smoke")
    if mode not in {"check", "smoke", "preflight", "train"}:
        raise ValueError("未知训练模式")
    if mode == "train" and (data.get("limit") is not None or config.get("profile") == "smoke"):
        raise ValueError("limit 缓存/smoke 配置不能冒充正式训练")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "task": "chd_multistage",
        "mode": mode,
        "train_cases": len(rows["train"]),
        "val_cases": len(rows["val"]),
        "test_cases_used": 0,
        "prepared": str(directory),
        "cache_resolution": data["resolution"],
        "unavailable": UNAVAILABLE,
    }
    collection = {
        "status": "running",
        "format": MODEL_FORMAT,
        "labels": list(LABELS),
        "normalization": NORMALIZATION,
        "mode": mode,
        "prepared_sha256": file_hash(directory / "dataset.json"),
        "stages": {},
        "unavailable": UNAVAILABLE,
    }
    write_json(output / "report.json", report)
    try:
        if mode == "check":
            for selected in rows.values():
                for row in selected:
                    load_case(directory, row)
            report["status"] = "checked"
            return report
        config = copy.deepcopy(config)
        config["device"] = device
        if mode == "smoke":
            config["profile"], config["augment"] = "smoke", False
            for stage, spec in config["stages"].items():
                spec.update(size=16, base=2, batch=1, epochs=1)
                if stage != "blood_lstm":
                    spec["levels"] = 3
        elif mode == "preflight":
            for spec in config["stages"].values():
                spec["epochs"] = 1
        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        collection["config"] = config
        write_json(output / "effective-config.json", config)
        write_json(output / "models.json", collection)
        for stage in STAGES:
            collection["stages"][stage] = train_stage(
                directory, rows, stage, config, output, mode, collection["prepared_sha256"]
            )
            write_json(output / "models.json", collection)
        collection["status"] = "complete"
        write_json(output / "models.json", collection)
        report.update(status="passed", models=str(output / "models.json"), stages=list(STAGES))
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        collection.update(status="failed", error=str(error))
        write_json(output / "models.json", collection)
        raise
    finally:
        write_json(output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHD 六阶段训练：只读取预处理缓存")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(config_path("chd.yaml")))
    parser.add_argument("--mode", choices=["check", "smoke", "preflight", "train"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    train(args.prepared, args.output, args.config, args.mode, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
