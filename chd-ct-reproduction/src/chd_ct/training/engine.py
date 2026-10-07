import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import build_model, weights_for
from ..data import read_manifest
from .augment import augment_batch
from .dataset import StageDataset
from .losses import dice_ce_loss


def load_checkpoint(path, device="cpu", expected_stage=None):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("format_version") != 1 or (expected_stage and record.get("stage") != expected_stage):
        raise ValueError(f"Incompatible checkpoint stage/format: {path}")
    model = build_model(record["stage"], record["spec"])
    model.load_state_dict(record["state_dict"], strict=True)
    return model.to(device).eval(), record


def recurrent_features(encoder, x):
    # Evaluate one time step at a time: a paper-sized 2D U-Net is too large for a batch of seven.
    with torch.no_grad():
        return torch.stack([encoder.features(x[:, time]) for time in range(x.shape[1])], dim=1)


def train_stage(config, manifest, stage, output, device="cpu", max_steps=None):
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    spec = config["stages"][stage]
    if config["profile"] == "paper" and device == "cpu":
        raise ValueError("paper profile requires a GPU; use smoke for CPU verification")
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(config.get("threads", 2))
    rows = read_manifest(manifest)
    manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    selected = {split: [r for r in rows if r["split"] == split] for split in ("train", "val")}
    if not all(selected.values()):
        raise ValueError("Training requires nonempty patient-separated train and val splits")
    loaders = {
        s: DataLoader(
            StageDataset(rs, stage, config), batch_size=spec["batch"], shuffle=(s == "train"), num_workers=0
        )
        for s, rs in selected.items()
    }
    model = build_model(stage, spec).to(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoder, encoder_record = None, None
    if stage == "blood_lstm":
        encoder, encoder_record = load_checkpoint(output / "blood2d.pt", device, "blood2d")
        if encoder_record["spec"] != config["stages"]["blood2d"]:
            raise ValueError("Frozen blood2d encoder configuration differs from current config")
        if encoder_record["manifest_sha256"] != manifest_hash:
            raise ValueError("Frozen encoder belongs to a different manifest/fold; retrain blood2d")
        encoder.requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    history, best = [], float("inf")
    for epoch in range(spec["epochs"]):
        fraction = epoch / spec["epochs"]
        lr = config["learning_rate"] * (1 if fraction < 0.5 else 0.1 if fraction < 0.75 else 0.01)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            for step, (x, y) in enumerate(loader):
                if max_steps is not None and step >= max_steps:
                    break
                x, y = x.to(device), y.to(device)
                if split == "train" and config.get("augment", False):
                    x, y = augment_batch(x, y)
                with torch.set_grad_enabled(split == "train"):
                    inputs = recurrent_features(encoder, x) if encoder is not None else x
                    loss = dice_ce_loss(model(inputs), y, weights_for(stage))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss; refusing to save checkpoint")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach()) * y.shape[0]
                count += y.shape[0]
            losses[split] = total / count
        history.append({"epoch": epoch + 1, "learning_rate": lr, **losses})
        print(json.dumps({"stage": stage, **history[-1]}), flush=True)
        if losses["val"] < best:
            best = losses["val"]
            record = {
                "format_version": 1,
                "stage": stage,
                "spec": spec,
                "config": config,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "seed": seed,
                "epoch": epoch + 1,
                "validation_loss": best,
                "max_steps": max_steps,
                "manifest_sha256": manifest_hash,
                "train_cases": [r["case_id"] for r in selected["train"]],
                "validation_cases": [r["case_id"] for r in selected["val"]],
            }
            if encoder_record is not None:
                # Bundle the exact frozen feature encoder, preventing mismatched inference features.
                record["encoder_state_dict"] = encoder_record["state_dict"]
                record["encoder_spec"] = encoder_record["spec"]
            temporary = output / f"{stage}.tmp.pt"
            torch.save(record, temporary)
            temporary.replace(output / f"{stage}.pt")
    (output / f"{stage}.history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output / f"{stage}.pt"
