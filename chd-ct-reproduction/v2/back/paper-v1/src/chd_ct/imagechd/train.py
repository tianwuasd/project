"""Train only from prepared caches; never scans or preprocesses raw NIfTI."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..models import UNet
from .common import FORMAT, IGNORE, LABELS, NORMALIZATION, load_config, read_prepared, write_json


def masked_loss(logits, target):
    valid = target != IGNORE
    if not valid.any():
        raise ValueError("没有可监督的体素")
    weights = logits.new_tensor([1] + [2] * 7)
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, 8).movedim(-1, 1).to(logits) * valid[:, None]
    probabilities = logits.softmax(1) * valid[:, None]
    axes = (0, 2, 3, 4)
    dice = (2 * (probabilities * onehot).sum(axes) + 1e-6) / (
        probabilities.sum(axes) + onehot.sum(axes) + 1e-6
    )
    return (
        1
        - (dice * weights).sum() / weights.sum()
        + F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE)
    )


class PreparedDataset(Dataset):
    def __init__(self, directory, rows, size):
        self.directory, self.rows, self.size = directory, rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.directory / self.rows[index]["cache"], allow_pickle=False) as record:
            image, target = record["image"], record["target"]
        if image.shape != (self.size,) * 3 or target.shape != image.shape:
            raise ValueError("缓存尺寸不同，请重新预处理")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("缓存影像必须已归一化至 [0,1]")
        if not np.isin(target, [*range(8), IGNORE]).all():
            raise ValueError("缓存标签超出七结构协议")
        return torch.from_numpy(image.copy())[None].float(), torch.from_numpy(target.copy()).long()


def training_rows(data):
    if data["for_prediction"]:
        raise ValueError("无标签预测缓存不能用于训练")
    patients = {}
    for row in data["cases"]:
        split, patient = row.get("split"), row.get("patient_id")
        if split not in {"train", "val", "test"} or not patient:
            raise ValueError("缺少有效的患者划分")
        if patients.setdefault(patient, split) != split:
            raise ValueError("同一患者跨数据划分")
    result = {s: [r for r in data["cases"] if r["split"] == s] for s in ("train", "val")}
    if not all(result.values()):
        raise ValueError("训练需要非空 train/val")
    return result


def train(prepared, output, config_path, mode="smoke", device="cpu"):
    directory, data = read_prepared(prepared)
    config = load_config(config_path)
    if config["size"] != data["size"]:
        raise ValueError("模型输入尺寸与预处理缓存不同；修改配置或另建预处理版本")
    rows = training_rows(data)
    if mode == "train" and data.get("limit") is not None:
        raise ValueError("limit 预处理结果仅用于短测；正式训练请预处理完整数据集")
    if mode not in {"check", "smoke", "preflight", "train"}:
        raise ValueError("未知训练模式")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "task": "imagechd7",
        "mode": mode,
        "prepared": str(directory),
        "device": device,
        "train_cases": len(rows["train"]),
        "val_cases": len(rows["val"]),
        "test_cases_used": 0,
        "metrics_space": "resized voxel grid, not physical/native space",
    }
    write_json(output / "report.json", report)
    try:
        datasets = {s: PreparedDataset(directory, rs, data["size"]) for s, rs in rows.items()}
        if mode == "check":
            for dataset in datasets.values():
                for index in range(len(dataset)):
                    dataset[index]
            report["status"] = "checked"
            return report
        effective = copy.deepcopy(config)
        if mode in {"smoke", "preflight"}:
            effective["epochs"] = 1
        if mode == "smoke":
            effective.update(base=2, levels=3, batch=1)
        write_json(output / "effective-config.json", effective)
        torch.manual_seed(effective["seed"])
        np.random.seed(effective["seed"])
        torch.set_num_threads(effective["threads"])
        model = UNet(3, 8, effective["base"], effective["levels"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=effective["learning_rate"])
        loaders = {
            s: DataLoader(ds, batch_size=effective["batch"], shuffle=s == "train", num_workers=0)
            for s, ds in datasets.items()
        }
        best, history = float("inf"), []
        maximum = 1 if mode in {"smoke", "preflight"} else None
        for epoch in range(effective["epochs"]):
            losses = {}
            samples = {}
            for split, loader in loaders.items():
                model.train(split == "train")
                total, count = 0.0, 0
                for step, (image, target) in enumerate(loader):
                    if maximum is not None and step >= maximum:
                        break
                    image, target = image.to(device), target.to(device)
                    with torch.set_grad_enabled(split == "train"):
                        loss = masked_loss(model(image), target)
                        if not torch.isfinite(loss):
                            raise RuntimeError("损失非有限值")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                    total += float(loss.detach()) * len(image)
                    count += len(image)
                losses[split], samples[split] = total / count, count
            history.append({"epoch": epoch + 1, "loss": losses, "cases_seen": samples})
            print(json.dumps(history[-1]), flush=True)
            write_json(output / "history.json", history)
            if losses["val"] < best:
                best = losses["val"]
                checkpoint = {
                    "format": FORMAT,
                    "labels": list(LABELS),
                    "normalization": NORMALIZATION,
                    "config": effective,
                    "mode": mode,
                    "epoch": epoch + 1,
                    "val_loss": best,
                    "prepared_sha256": hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest(),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                torch.save(checkpoint, output / "best.pt.tmp")
                (output / "best.pt.tmp").replace(output / "best.pt")
        report.update(
            status="passed", best_val_loss=best, epochs=len(history), checkpoint=str(output / "best.pt")
        )
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅训练 ImageCHD 七结构模型；输入必须已预处理")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True, help="新的模型目录")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[3] / "configs/imagechd7.yaml")
    )
    parser.add_argument("--mode", choices=["check", "smoke", "preflight", "train"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    train(args.prepared, args.output, args.config, args.mode, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
