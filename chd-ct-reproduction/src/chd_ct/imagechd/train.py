"""Six-stage training from reusable caches; no raw-image preprocessing."""

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..models.chd_unet import ARCHITECTURE, GRID_ADAPTER
from ..paths import config_path
from ..recovery import atomic_path, file_hash, output_lock, read_json
from .augment import augment_batch
from .common import LABELS, MODEL_FORMAT, NORMALIZATION, write_json
from .config import STAGES, build_model, load_config, weights_for
from .dataset import StageDataset
from .loading import close_loader, make_loader, settings, start_epoch
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
    from .autotune import cache_directory

    torch.set_num_threads(config["threads"])
    stage_seed = config["seed"] + 10000 * config.get("fold", {}).get("index", 0) + STAGES.index(stage)
    loaders = {}
    started = time.perf_counter()
    try:
        for split, selected in rows.items():
            dataset = StageDataset(directory, selected, stage, config, cache_directory(directory, config))
            loaders[split] = make_loader(dataset, config, stage, split, stage_seed)
        print(f"[input-ready] {stage}: {time.perf_counter() - started:.2f}s", flush=True)
        return _train_stage(directory, rows, stage, config, output, mode, manifest_hash, resume, loaders)
    finally:
        for loader in loaders.values():
            close_loader(loader)


def _train_stage(directory, rows, stage, config, output, mode, manifest_hash, resume, loaders):
    torch.set_num_threads(config["threads"])
    stage_seed = config["seed"] + 10000 * config.get("fold", {}).get("index", 0) + STAGES.index(stage)
    torch.manual_seed(stage_seed)
    np.random.seed(stage_seed)
    random.seed(stage_seed)
    spec = config["stages"][stage]
    model = build_model(stage, spec).to(config["device"])
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
    timing_path = output / (stage + ".timing.json")
    timings = read_json(timing_path) if resume and timing_path.is_file() else []
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
        epoch_timing = {"epoch": epoch + 1, "splits": {}}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            split_start = time.perf_counter()
            iterator = start_epoch(loader, "data_loading" in config)
            waiting, compute = 0.0, 0.0
            last_progress = split_start
            steps = min(maximum, len(loader)) if maximum else len(loader)
            for step in range(steps):
                before = time.perf_counter()
                image, target = next(iterator)
                waiting += time.perf_counter() - before
                before = time.perf_counter()
                non_blocking = config["device"] == "cuda" and settings(config)["pin_memory"]
                image, target = (
                    image.to(config["device"], non_blocking=non_blocking),
                    target.to(config["device"], non_blocking=non_blocking),
                )
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
                # loss.item above synchronizes CUDA; this bucket includes transfer,
                # augmentation and model work, not a misleading CPU launch time.
                compute += time.perf_counter() - before
                now = time.perf_counter()
                if now - last_progress >= settings(config)["progress_seconds"]:
                    print(
                        json.dumps(
                            dict(
                                event="batch_progress",
                                stage=stage,
                                epoch=epoch + 1,
                                split=split,
                                batch=step + 1,
                                batches=steps,
                                samples=count,
                                elapsed_seconds=now - split_start,
                                data_wait_seconds=waiting,
                            )
                        ),
                        flush=True,
                    )
                    last_progress = now
            if not count:
                raise ValueError("阶段没有可用样本")
            losses[split], samples[split] = total / count, count
            epoch_timing["splits"][split] = dict(
                samples=count,
                seconds=time.perf_counter() - split_start,
                data_wait_seconds=waiting,
                transfer_augment_compute_seconds=compute,
            )
        timings = [t for t in timings if t["epoch"] < epoch + 1] + [epoch_timing]
        write_json(timing_path, timings)
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
    return {
        "file": stage + ".pt",
        "sha256": file_hash(output / (stage + ".pt")),
        "best_val_loss": best,
        "best_epoch": min(history, key=lambda r: r["loss"]["val"])["epoch"],
        "epochs_completed": len(history),
    }


def train(
    prepared,
    output,
    config_path,
    mode="smoke",
    device="cpu",
    gpu_ids=None,
    resume=False,
    gpu_limit=None,
    *,
    fold_context=None,
    inherited_lock_fds=(),
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output) as lock_fd:
        if fold_context is None and load_config(config_path)["cross_validation"]["folds"] == 3:
            from .cross_validation import train_threefold

            return train_threefold(
                prepared, output, config_path, mode, device, gpu_ids, resume, gpu_limit, lock_fd
            )
        return train_locked(
            prepared,
            output,
            config_path,
            mode,
            device,
            gpu_ids,
            resume,
            lock_fd,
            gpu_limit,
            fold_context=fold_context,
            inherited_lock_fds=inherited_lock_fds,
        )


def train_locked(
    prepared,
    output,
    config_path,
    mode,
    device,
    gpu_ids,
    resume,
    lock_fd,
    gpu_limit=None,
    *,
    fold_context=None,
    inherited_lock_fds=(),
):
    from .gpu_queue import options
    from .training_session import commit_stage, finish_session, training_session

    queue_options = options(gpu_ids, gpu_limit, device, mode)
    with training_session(
        prepared,
        output,
        config_path,
        mode,
        device,
        gpu_limit or len(gpu_ids or []) or 1,
        resume,
        fold_context,
    ) as session:
        if mode == "check":
            return session["report"]
        directory, output = session["directory"], session["output"]
        config, rows = session["config"], session["rows"]
        if queue_options:
            from .stage_queue import run_queue

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

            run_queue(
                STAGES,
                output=output,
                command_for=command_for,
                env=os.environ.copy(),
                on_complete=lambda stage, info: commit_stage(session, stage, info),
                lock_fds=(*inherited_lock_fds, lock_fd),
                **queue_options,
            )
        else:
            for stage in STAGES:
                info = train_stage(
                    directory,
                    rows,
                    stage,
                    config,
                    output,
                    mode,
                    session["collection"]["prepared_sha256"],
                    resume,
                )
                commit_stage(session, stage, info)
        finish_session(session)
        return session["report"]


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
