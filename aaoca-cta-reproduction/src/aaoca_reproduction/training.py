from pathlib import Path
import json
import os
import random
import subprocess
from time import perf_counter

import numpy as np
import torch

from .augmentations import AuthorAugmentation
from .finetuning import make_optimizer, set_training_phase, settings_for_finetuning
from .classification import LEARNING_RATES, build_model, coverage_reviews, load_weights, model_contract, model_metadata, require_direction_review, training_strategy, validate_model_metadata
from .data import guard_inputs, load_batch, prepared_receipt
from .labels import task_partitions
from .records import event, identity, write_json
from .resources import admit_model, training_devices, configure_cpu, memory_guard, smoke_shape


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    return {"sha": git("rev-parse", "HEAD"), "modified_files": git("status", "--porcelain", "--untracked-files=no", "--", ".").splitlines()}


def random_state(device, devices=None):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "cuda_device": str(device) if device.type == "cuda" else None,
            "torch_cuda_devices": {str(d): torch.cuda.get_rng_state(d) for d in (devices or [])}}


def restore_random(state, device):
    random.setstate(state["python"])
    item = state["numpy"]
    np.random.set_state((item[0], np.asarray(item[1], dtype=np.uint32), item[2], item[3], item[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if state["cuda_device"] != str(device) or device.type != "cuda":
            raise ValueError("恢复时所选CUDA随机状态设备不一致")
        torch.cuda.set_rng_state(state["torch_cuda"], device)
        for name, value in state.get("torch_cuda_devices", {}).items():
            torch.cuda.set_rng_state(value, torch.device(name))


def save_checkpoint(path, model, optimizer, scheduler, state, run_identity, meta, initialization):
    parallel = isinstance(model, torch.nn.DataParallel)
    core = model.module if parallel else model
    devices = [torch.device("cuda", i) for i in model.device_ids] if parallel else []
    value = {"schema": 1, "identity": run_identity, "model_meta": meta, "model": core.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": None,
             "state": dict(state), "random": random_state(next(model.parameters()).device, devices), "initialization": initialization,
             "data_workers": 0, "prefetch": 0, "augmentation_rng": "numpy-derived-per-sample-MONAI-and-global-python-torch"}
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    pending.replace(path)


def save_update_attempts(path, run_identity, attempted_updates):
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump({"schema": 1, "identity": run_identity, "attempted_updates": attempted_updates}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def run_budget(directory, budget, pending_estimate):
    files = list(directory.rglob("*"))
    if any(p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()) for p in files):
        raise ValueError("训练目录包含外部链接")
    used = sum(p.stat().st_size for p in files if p.is_file())
    if used + pending_estimate > budget:
        raise OSError("训练检查点及日志将超出获准产物容量")


@guard_inputs
def train(config, manifest, resume=False):
    settings = config["classification"]
    code = code_provenance()
    task = settings["task"]
    strategy = training_strategy(settings, task)
    fine = settings_for_finetuning(settings)
    meta = model_metadata(task)
    if settings["learning_rate_recipe"] not in LEARNING_RATES:
        raise ValueError("必须具名选择论文或发布版学习率")
    if settings.get("precision", "float32") != "float32":
        raise ValueError("本训练配方仅实现作者float32；其他精度需要独立配方及恢复验收")
    epochs, batch_size = int(settings["epochs"]), int(settings["batch_size"])
    checkpoint_interval = int(settings.get("checkpoint_interval_updates", 10))
    member, seed = int(settings["member_index"]), int(settings["seed"])
    update_limit = settings.get("max_update_attempts")
    budget = settings.get("artifact_budget_bytes")
    if budget is not None and (not isinstance(budget, int) or budget <= 0):
        raise ValueError("显式产物预算必须为正整数；省略时按真实参数自动估算")
    if update_limit is not None and (not isinstance(update_limit, int) or update_limit < 1):
        raise ValueError("更新尝试上限必须为明确正整数")
    if not 2 <= epochs <= 300 or batch_size < 1 or member not in range(5) or checkpoint_interval < 1:
        raise ValueError("轮数需2到300、batch与保存间隔需正数、初始化成员需0到4")
    train_rows, validation_rows = task_partitions(config, manifest, task)
    train_inputs = [prepared_receipt(row, config) for row in train_rows]
    validation_inputs = [prepared_receipt(row, config) for row in validation_rows]
    direction = require_direction_review(config, train_inputs + validation_inputs)
    coverage = coverage_reviews(config, train_rows + validation_rows, train_inputs + validation_inputs,
                                require_verified=config.get("training_policy") != "local_adaptation")
    small = smoke_shape(config)
    pretrained = settings.get("pretrained_detection")
    if task != "aaoca_present":
        if pretrained is None:
            raise ValueError("起源与风险训练需要有来源的检测模型初始化")
        known_training = set(pretrained["training_patient_keys"])
        if not known_training or known_training.intersection(row["patient_key"] for row in validation_rows):
            raise ValueError("检测初始化患者来源缺失或与当前验证患者重叠")
    elif pretrained is not None and fine is None:
        raise ValueError("作者检测配方需要从头初始化")
    directory = Path(settings["run_directory"]).resolve()
    course = Path(config["course_root"]).resolve()
    if directory == course or not directory.is_relative_to(course):
        raise ValueError("训练状态和日志须存于course专用运行目录")
    latest = directory / "latest.pt"
    attempts_path = directory / "update-attempts.json"
    if resume != latest.exists():
        raise ValueError("已有完整latest需显式resume；新任务不接受缺失latest的恢复")
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError("新训练目录含既有内容")
    semantic_settings = {k: v for k, v in settings.items() if k != "max_update_attempts"}
    run_identity = identity({"settings": semantic_settings, "model": model_contract(task), "direction": direction, "coverage": coverage,
                             "train": [(r["patient_key"], r["target"], p["identity"]) for r, p in zip(train_rows, train_inputs)],
                             "validation": [(r["patient_key"], r["target"], p["identity"]) for r, p in zip(validation_rows, validation_inputs)]})
    if config.get("label_policy", "confirmed") in {"document_explicit", "research_sources", "patient_research"}:
        run_identity = identity({"base": run_identity, "label_policy": config["label_policy"],
                                 "labels": [(row["patient_key"], row["label_provenance"])
                                            for row in train_rows + validation_rows]})
    if config.get("training_policy") == "local_adaptation" or small:
        run_identity = identity({"base": run_identity, "training_policy": config.get("training_policy"),
                                 "run_purpose": config.get("run_purpose", "training")})
    if resume:
        inspected_at = perf_counter()
        # 完成成员只读取真实检查点元数据，不再为模型与优化器预留整套工作内存。
        completed = torch.load(latest, map_location="cpu", weights_only=True, mmap=True)
        if completed["schema"] != 1 or completed["identity"] != run_identity:
            raise ValueError("恢复检查点的输入、方法、模型或配置身份不匹配")
        validate_model_metadata(completed["model_meta"], task)
        if completed["state"]["strategy"] != strategy:
            raise ValueError("恢复检查点的实际训练策略不符")
        if completed["state"]["complete"]:
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            if (attempts["schema"] != 1 or attempts["identity"] != run_identity
                    or type(attempts["attempted_updates"]) is not int
                    or attempts["attempted_updates"] < completed["state"]["attempted_updates"]):
                raise ValueError("完成成员的更新次数记录与检查点不一致")
            return {"directory": str(directory), "identity": run_identity, "state": completed["state"],
                    "reused_complete_member": True, "seconds_this_invocation": perf_counter() - inspected_at}
        del completed
    resources = configure_cpu(config)
    admission = admit_model(config)
    budget = budget if budget is not None else admission["plan"]["checkpoint_capacity_estimate_bytes"]
    devices = training_devices(config)
    device = devices[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        for selected in devices:
            with torch.cuda.device(selected):
                torch.cuda.manual_seed(seed)
    with memory_guard(config) as memory:
        model = build_model(task).to(device)
        initialization = {"source": "MONAI_Kaiming_from_scratch", "seed": seed, "code": code}
        if not resume and pretrained is not None:
            initialization = load_weights(model, pretrained, "aaoca_present")
        optimizer = make_optimizer(model, fine, LEARNING_RATES[settings["learning_rate_recipe"]])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(0.75 * epochs), eta_min=1e-6)
        if settings.get("augmentation", "author") == "source_robust_v1":
            from .augmentations import SourceRobustAugmentation
            augmentation = SourceRobustAugmentation()
        else:
            augmentation = AuthorAugmentation(probability=0.15)
        state = {"epoch": 0, "phase": "train", "order": [], "cursor": 0,
                 "train_loss_sum": 0.0, "train_batches": 0, "val_loss_sum": 0.0, "val_batches": 0,
                 "attempted_updates": 0, "successful_updates": 0, "best_loss": float("inf"),
                 "bad_epochs": 0, "complete": False, "member_index": member, "seed": seed, "strategy": strategy,
                 "coverage_reviews": coverage,
                 "test_only": bool(small), "smoke_input_shape": list(small) if small else None,
                 "training_patient_keys": [r["patient_key"] for r in train_rows],
                 "validation_patient_keys": [r["patient_key"] for r in validation_rows],
                 "cohort_identity": identity({"task": task,
                     "train": [(r["patient_key"], r["target"], p["identity"]) for r, p in zip(train_rows, train_inputs)],
                     "validation": [(r["patient_key"], r["target"], p["identity"]) for r, p in zip(validation_rows, validation_inputs)]})}
        if fine is not None:
            state.update(best_average_precision=-1.0, val_targets=[], val_probabilities=[])
        if config.get("label_policy", "confirmed") in {"document_explicit", "research_sources", "patient_research"}:
            state["label_policy"] = config["label_policy"]
        if any("label_provenance" in row for row in train_rows + validation_rows):
            state["label_sources"] = {row["patient_key"]: row["label_provenance"]
                                      for row in train_rows + validation_rows if "label_provenance" in row}
        if resume:
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            if checkpoint["schema"] != 1 or checkpoint["identity"] != run_identity:
                raise ValueError("恢复检查点的输入、方法、模型或配置身份不匹配")
            validate_model_metadata(checkpoint["model_meta"], task)
            if checkpoint["state"]["strategy"] != strategy:
                raise ValueError("恢复检查点的实际训练策略不符")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            state, initialization = checkpoint["state"], checkpoint["initialization"]
            restore_random(checkpoint["random"], device)
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            if (attempts["schema"] != 1 or attempts["identity"] != run_identity
                    or type(attempts["attempted_updates"]) is not int
                    or attempts["attempted_updates"] < state["attempted_updates"]):
                raise ValueError("更新尝试记录身份、次数或与检查点的先后关系不符")
            state["attempted_updates"] = attempts["attempted_updates"]
            del checkpoint
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=[d.index for d in devices], output_device=device.index)
        directory.mkdir(parents=True, exist_ok=True)
        parameter_count = sum(p.numel() for p in model.parameters())
        pending_bytes = parameter_count * 16 + 8 * 2**20
        if budget < pending_bytes * 4:
            raise OSError("获准容量不足以容纳完整latest/best及原子保存暂存")
        loss_function = torch.nn.BCEWithLogitsLoss()
        started = perf_counter()
        if not resume:
            write_json(directory / "run.json", {"identity": run_identity, "settings": settings, "model": meta,
                       "code": code, "resolved_config": json.loads(json.dumps(config, default=str)),
                       "label_policy": config.get("label_policy", "confirmed"),
                       "label_sources": state.get("label_sources", {}),
                       "resources": resources, "admission": admission, "direction": direction, "coverage_reviews": coverage,
                       "parameter_count": parameter_count, "precision": "float32", "workers": 0,
                       "train_patients": [r["patient_key"] for r in train_rows],
                       "validation_patients": [r["patient_key"] for r in validation_rows]})
            run_budget(directory, budget, pending_bytes + 1024)
            save_update_attempts(attempts_path, run_identity, 0)
            save_checkpoint(latest, model, optimizer, scheduler, state, run_identity, meta, initialization)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as log:
            while state["epoch"] < epochs and not state["complete"]:
                phase = set_training_phase(model, fine, state["epoch"])
                if state["phase"] == "train":
                    if not state["order"]:
                        state["order"] = np.random.permutation(len(train_rows)).tolist()
                    while state["cursor"] < len(state["order"]):
                        if update_limit is not None and state["attempted_updates"] >= update_limit:
                            run_budget(directory, budget, pending_bytes)
                            save_checkpoint(latest, model, optimizer, scheduler, state, run_identity, meta, initialization)
                            event("classification_update_budget_reached", attempted_updates=state["attempted_updates"])
                            return {"directory": str(directory), "identity": run_identity, "state": state,
                                    "memory": memory, "stop_reason": "authorized_update_limit"}
                        indices = state["order"][state["cursor"]:state["cursor"] + batch_size]
                        # 在供数及模型计算前持久预占次数，中断后也保留该次尝试。
                        run_budget(directory, budget, 1024)
                        save_update_attempts(attempts_path, run_identity, state["attempted_updates"] + 1)
                        state["attempted_updates"] += 1
                        images, targets = load_batch(train_rows, train_inputs, indices, augmentation, smoke_shape=small)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(model(images.to(device)), targets.to(device))
                        if not torch.isfinite(loss):
                            raise ValueError("训练损失不是有限值")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                        optimizer.step()
                        state["cursor"] += len(indices)
                        state["successful_updates"] += 1
                        state["train_loss_sum"] += float(loss.item())
                        state["train_batches"] += 1
                        if state["attempted_updates"] % checkpoint_interval == 0:
                            run_budget(directory, budget, pending_bytes)
                            save_checkpoint(latest, model, optimizer, scheduler, state, run_identity, meta, initialization)
                    state.update(phase="validation", cursor=0)
                    run_budget(directory, budget, pending_bytes)
                    save_checkpoint(latest, model, optimizer, scheduler, state, run_identity, meta, initialization)
                    if update_limit is not None and state["attempted_updates"] >= update_limit:
                        event("classification_update_budget_reached", attempted_updates=state["attempted_updates"])
                        return {"directory": str(directory), "identity": run_identity, "state": state,
                                "memory": memory, "stop_reason": "authorized_update_limit"}
                if state["phase"] == "validation":
                    model.eval()
                    with torch.no_grad():
                        while state["cursor"] < len(validation_rows):
                            indices = list(range(state["cursor"], min(state["cursor"] + batch_size, len(validation_rows))))
                            images, targets = load_batch(validation_rows, validation_inputs, indices, smoke_shape=small)
                            logits = model(images.to(device))
                            loss = loss_function(logits, targets.to(device))
                            if not torch.isfinite(loss):
                                raise ValueError("验证损失不是有限值")
                            state["val_loss_sum"] += float(loss.item())
                            if fine is not None:
                                state["val_targets"].extend(targets.reshape(-1).tolist())
                                state["val_probabilities"].extend(torch.sigmoid(logits).reshape(-1).cpu().tolist())
                            state["val_batches"] += 1
                            state["cursor"] += len(indices)
                            if state["val_batches"] % checkpoint_interval == 0:
                                run_budget(directory, budget, pending_bytes)
                                save_checkpoint(latest, model, optimizer, scheduler, state, run_identity, meta, initialization)
                    train_loss = state["train_loss_sum"] / state["train_batches"]
                    validation_loss = state["val_loss_sum"] / state["val_batches"]
                    scheduler.step()
                    score = None
                    if fine is not None and len(set(state["val_targets"])) == 2:
                        from sklearn.metrics import average_precision_score
                        score = float(average_precision_score(state["val_targets"], state["val_probabilities"]))
                        improved = (score > state["best_average_precision"] or
                                    (score == state["best_average_precision"] and validation_loss <= state["best_loss"]))
                    else:
                        improved = validation_loss <= state["best_loss"]
                    state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
                    if improved:
                        state["best_loss"] = validation_loss
                        if fine is not None and score is not None:
                            state["best_average_precision"] = score
                    state["epoch"] += 1
                    state.update(phase="train", cursor=0, order=[], train_loss_sum=0.0, train_batches=0,
                                 val_loss_sum=0.0, val_batches=0,
                                 complete=state["bad_epochs"] >= (fine["patience"] if fine is not None else 100) or state["epoch"] >= epochs)
                    if fine is not None:
                        state.update(val_targets=[], val_probabilities=[])
                    record = {"epoch": state["epoch"], "train_loss_mean_of_batches": train_loss,
                              "validation_loss_mean_of_batches": validation_loss, "improved_or_equal": improved,
                              "successful_updates": state["successful_updates"], "lr": optimizer.param_groups[0]["lr"],
                              "elapsed_seconds": perf_counter() - started}
                    if fine is not None:
                        record.update(validation_average_precision=score, training_phase=phase,
                                      selection_metric="average_precision_then_validation_loss" if score is not None else
                                      "smoke_single_class_validation_loss_no_performance_claim")
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    event("classification_epoch_completed", **record)
                    if improved:
                        run_budget(directory, budget, pending_bytes)
                        save_checkpoint(directory / "best.pt", model, optimizer, scheduler, state, run_identity, meta, initialization)
                    run_budget(directory, budget, pending_bytes)
                    save_checkpoint(latest, model, optimizer, scheduler, state, run_identity, meta, initialization)
    return {"directory": str(directory), "identity": run_identity, "state": state,
            "memory": memory, "seconds_this_invocation": perf_counter() - started}
