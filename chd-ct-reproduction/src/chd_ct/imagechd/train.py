"""Six-stage training from reusable caches; no raw-image preprocessing."""

import argparse
import copy
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.grid import GRID_ADAPTER
from ..models.unet import ARCHITECTURE
from ..paths import config_path
from ..recovery import atomic_path, file_hash, output_lock, read_json
from .augment import augment_batch
from .common import LABELS, MODEL_FORMAT, NORMALIZATION, load_case, read_prepared, write_json
from .config import STAGES, UNAVAILABLE, build_model, load_config, weights_for
from .dataset import StageDataset
from .losses import masked_loss


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


def train_stage(directory, rows, stage, config, output, mode, manifest_hash, resume=False):
    torch.set_num_threads(config["threads"])
    stage_seed = config["seed"] + STAGES.index(stage)
    torch.manual_seed(stage_seed)
    np.random.seed(stage_seed)
    random.seed(stage_seed)
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
    last_file = output / (stage + ".last.pt")
    contract = dict(
        recovery_version=1,
        stage=stage,
        config=config,
        mode=mode,
        prepared_sha256=manifest_hash,
        architecture=ARCHITECTURE,
        encoder_sha256=file_hash(output / "blood2d.pt") if encoder is not None else None,
    )
    start, best_bytes = 0, None
    if resume and last_file.is_file():
        last = torch.load(last_file, map_location="cpu", weights_only=True)
        if last.get("contract") != contract:
            raise ValueError("训练检查点的配置、数据或编码器来源不同：" + stage)
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        start, history, best = last["epoch"], last["history"], last["best"]
        best_bytes = last["best_bytes"]
        if not 0 <= start <= epochs or len(history) != start:
            raise ValueError("训练检查点轮次不完整：" + stage)
        torch.set_rng_state(last["torch_rng"])
        rng = last["numpy_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        random.setstate(last["python_rng"])
        if config["device"] == "cuda":
            torch.cuda.set_rng_state(last["cuda_rng"])
        # The last checkpoint is the transaction: restore exports even if a crash
        # occurred between publishing last.pt and best.pt/history.json.
        with atomic_path(output / (stage + ".pt")) as temporary:
            temporary.write_bytes(best_bytes)
        write_json(output / (stage + ".history.json"), history)
        print(f"{stage}：从已完成第 {start}/{epochs} 轮接续", flush=True)
    elif resume and (output / (stage + ".pt")).exists():
        raise ValueError("旧版训练权重不含优化器状态，不能精确续训；请另建训练任务：" + stage)
    for epoch in range(start, epochs):
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
        if losses["val"] < best:
            best = losses["val"]
            record = {
                "format": MODEL_FORMAT,
                "architecture": ARCHITECTURE,
                "grid_adapter": GRID_ADAPTER,
                "stage": stage,
                "labels": list(LABELS),
                "normalization": NORMALIZATION,
                "config": config,
                "spec": spec,
                "mode": mode,
                "prepared_sha256": manifest_hash,
                "epoch": epoch + 1,
                "val_loss": best,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            if encoder_record is not None:
                record.update(
                    encoder_state_dict=encoder_record["state_dict"],
                    encoder_spec=config["stages"]["blood2d"],
                    encoder_sha256=file_hash(output / "blood2d.pt"),
                )
            buffer = io.BytesIO()
            torch.save(record, buffer)
            best_bytes = buffer.getvalue()
        rng = np.random.get_state()
        last = dict(
            contract=contract,
            epoch=epoch + 1,
            history=history,
            best=best,
            best_bytes=best_bytes,
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            numpy_rng=(rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]),
            python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state() if config["device"] == "cuda" else None,
        )
        with atomic_path(last_file) as temporary:
            torch.save(last, temporary)
        with atomic_path(output / (stage + ".pt")) as temporary:
            temporary.write_bytes(best_bytes)
        write_json(output / (stage + ".history.json"), history)
    return {"file": stage + ".pt", "sha256": file_hash(output / (stage + ".pt")), "best_val_loss": best}


def train(
    prepared, output, config_path, mode="smoke", device="cpu", gpu_ids=None, resume=False, gpu_limit=None
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output) as lock_fd:
        return train_locked(prepared, output, config_path, mode, device, gpu_ids, resume, lock_fd, gpu_limit)


def train_locked(prepared, output, config_path, mode, device, gpu_ids, resume, lock_fd, gpu_limit=None):
    if gpu_limit is not None and (
        isinstance(gpu_limit, bool)
        or not isinstance(gpu_limit, int)
        or not 1 <= gpu_limit <= 5
        or not gpu_ids
    ):
        raise ValueError("弹性队列需要候选 GPU 和 1—5 的并发上限")
    if gpu_ids:
        if device != "cuda" or mode == "check":
            raise ValueError("多卡队列仅用于 CUDA 训练/短测/预检")
        from ..server.launcher import gpu_pool, idle_gpus, query_gpus, select_gpus
        from .stage_queue import check_gpu_handoff

        if gpu_limit is not None:
            gpu_ids = [
                r["uuid"]
                for r in gpu_pool(query_gpus(), ",".join(gpu_ids), os.environ.get("CUDA_VISIBLE_DEVICES"))
            ]
        else:
            if len(gpu_ids) != len(set(gpu_ids)) or not 1 <= len(gpu_ids) <= 5:
                raise ValueError("需要 1—5 个不重复 GPU UUID")
            # Legacy direct calls keep their explicit, fixed-card semantics.
            for gpu in gpu_ids:
                check_gpu_handoff(gpu, used=True)
            select_gpus(query_gpus(), len(gpu_ids), ",".join(gpu_ids), os.environ.get("CUDA_VISIBLE_DEVICES"))
    directory, data = read_prepared(prepared)
    config = load_config(config_path)
    rows = training_rows(data)
    if config.get("profile") == "author-unet" and device == "cpu" and mode not in {"check", "smoke"}:
        raise ValueError("author-unet 需要 GPU 预检；CPU 仅支持 check/smoke")
    if mode not in {"check", "smoke", "preflight", "train"}:
        raise ValueError("未知训练模式")
    if mode == "train" and (data.get("limit") is not None or config.get("profile") == "smoke"):
        raise ValueError("limit 缓存/smoke 配置不能冒充正式训练")
    output = Path(output).expanduser().resolve()
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
        "architecture": ARCHITECTURE,
        "grid_adapter": GRID_ADAPTER,
        "labels": list(LABELS),
        "normalization": NORMALIZATION,
        "mode": mode,
        "prepared_sha256": file_hash(directory / "dataset.json"),
        "stages": {},
        "unavailable": UNAVAILABLE,
    }
    validated = False
    try:
        if mode == "check":
            if (output / "models.json").exists():
                raise ValueError("缓存检查请使用独立输出目录")
            validated = True
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
        contract = dict(
            config=config,
            mode=mode,
            prepared_sha256=collection["prepared_sha256"],
            architecture=ARCHITECTURE,
            recovery_version=1,
            cache_hashes={r["case_id"]: file_hash(directory / r["cache"]) for r in data["cases"]},
        )
        contract_path = output / "training-contract.json"
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError("训练数据或配置已改变，不能续训")
        if resume and not contract_path.exists() and (output / "models.json").exists():
            raise ValueError("旧版训练未记录优化器，不能精确续训；请另建训练目录")
        validated = True
        write_json(output / "report.json", report)
        write_json(contract_path, contract)
        write_json(output / "effective-config.json", config)
        write_json(output / "models.json", collection)
        if gpu_ids:
            from .stage_queue import check_gpu_handoff, run_queue

            def command_for(stage):
                return [
                    sys.executable,
                    "-m",
                    "chd_ct.imagechd.stage_worker",
                    "--prepared",
                    directory,
                    "--output",
                    output,
                    "--stage",
                    stage,
                    "--mode",
                    mode,
                    *(["--resume"] if resume else []),
                    *(["--check-environment"] if gpu_limit is not None else []),
                ]

            def on_complete(stage, info):
                collection["stages"][stage] = info
                write_json(output / "models.json", collection)

            def available():
                devices = query_gpus()
                if not set(gpu_ids).intersection(r["uuid"] for r in devices):
                    raise ValueError("候选 GPU 已不可见，请检查设备或分配范围")
                return [r["uuid"] for r in idle_gpus(devices, gpu_ids)]

            run_queue(
                STAGES,
                gpu_ids,
                output,
                command_for,
                os.environ.copy(),
                (lambda gpu, used: gpu in available()) if gpu_limit is not None else check_gpu_handoff,
                on_complete,
                lock_fds=(lock_fd,),
                available_gpus=available if gpu_limit is not None else None,
                gpu_limit=gpu_limit,
            )
            collection["stages"] = {s: collection["stages"][s] for s in STAGES}
        else:
            for stage in STAGES:
                collection["stages"][stage] = train_stage(
                    directory, rows, stage, config, output, mode, collection["prepared_sha256"], resume
                )
                write_json(output / "models.json", collection)
        collection["status"] = "complete"
        write_json(output / "models.json", collection)
        report.update(status="passed", models=str(output / "models.json"), stages=list(STAGES))
        return report
    except BaseException as error:
        if not validated:
            raise
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        collection.update(status="failed", error=str(error))
        write_json(output / "models.json", collection)
        raise
    finally:
        if validated:
            write_json(output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHD 六阶段训练：只读取预处理缓存")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(config_path("chd.yaml")))
    parser.add_argument("--mode", choices=["check", "smoke", "preflight", "train"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-ids", help="内部多卡队列：逗号分隔的已分配 GPU UUID")
    parser.add_argument(
        "--gpu-limit", type=int, help="弹性队列并发上限（1—5）；gpu-ids 为候选池，无空闲卡时等待"
    )
    parser.add_argument("--resume", action="store_true", help="从最后完整 epoch 恢复模型、优化器和随机状态")
    args = parser.parse_args(argv)
    train(
        args.prepared,
        args.output,
        args.config,
        args.mode,
        args.device,
        args.gpu_ids.split(",") if args.gpu_ids else None,
        args.resume,
        args.gpu_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
