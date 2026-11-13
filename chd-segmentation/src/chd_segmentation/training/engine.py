from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
import psutil
import torch

from ..artifacts.store import digest, operation_lock, read_json, require_output, write_json
from ..data.dataset import CaseDataset, sample_patch
from ..models.stunet.adapter import StunetAdapter, WEIGHT_SHA256
from ..observability.events import event
from ..preprocessing.prepare import load_prepared, validate_prepared_config
from .checkpoint import load_checkpoint, save_checkpoint, tensors_equal


def code_identity():
    root = Path(__file__).parents[1]
    hashes = sha256()
    for path in sorted(root.rglob("*.py")):
        hashes.update(str(path.relative_to(root)).replace("\\", "/").encode())
        hashes.update(path.read_bytes())
    return hashes.hexdigest()


def device_for(config):
    name = config["resources"]["device"]
    if name != "cpu":
        # 此阶段仅开放 CPU，避免未接入的 GPU 准入规则被绕过。
        raise ValueError("当前 STU 本地验收实现仅开放 CPU；GPU 资源准入尚未验收")
    return torch.device("cpu")


def arithmetic(config, device):
    if config["training"]["precision"] == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("BF16 配方需要已准入的 CUDA 设备")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def get_batch(dataset, rows, config, rng, device):
    selected = rng.choice(rows, size=config["training"]["batch"], replace=True).tolist()
    pairs, unknown = [], []
    for name in selected:
        entry = dataset.get(name)
        pairs.append(sample_patch(entry, config["training"]["patch"], rng, config["training"]["foreground_fraction"]))
        unknown.append(entry["unknown_classes"])
    image, target = [np.stack(items) for items in zip(*pairs)]
    return {"image": torch.from_numpy(image).to(device), "label": torch.from_numpy(target).to(device),
            "unknown_classes": unknown, "case_ids": selected}


def train(config, prepared_path, resume=False, stop_after=None):
    started = perf_counter()
    prepared = load_prepared(prepared_path)
    validate_prepared_config(prepared, config)
    if prepared["dataset"] != config["dataset"] or prepared["scope"] != config["scope"] or prepared["fold"] != config["fold"]:
        raise ValueError("训练配置与预处理描述不匹配")
    output = require_output(config["run_dir"], config["work_root"], [prepared["manifest"]["data_root"]])
    if resume and not (output / "latest.json").is_file():
        raise FileNotFoundError("恢复训练需要已有完整检查点")
    torch.set_num_threads(config["resources"]["cpu_threads"])
    device = device_for(config)
    rows = prepared["manifest"]["cases"]
    names = [row["case_id"] for row in rows if "train" in row["roles"]]
    validation = [row["case_id"] for row in rows if "validation" in row["roles"]]
    if not names or not validation or any(row.get("label") is None for row in rows if set(row["roles"]) & {"train", "validation"}):
        raise ValueError("训练与验证需要对应图像、标签及非空角色")
    specification = StunetAdapter.spec(config, prepared["plan"])
    scientific = {"prepared_identity": prepared["identity"], "model": "stunet", "spec": asdict(specification),
                  "training": config["training"], "pretrained": WEIGHT_SHA256,
                  "code_identity": code_identity(),
                  "versions": {name: version(name) for name in ("torch", "monai", "nnunetv2", "numpy", "scipy")}}
    identity = digest(scientific)
    if not resume and (output / "run.json").exists():
        raise FileExistsError("运行目录已存在，请明确选择恢复训练或新目录")
    with operation_lock(output), ExitStack() as cleanup:
        state = load_checkpoint(output, identity) if resume else None
        if state and read_json(output / "run.json")["identity"] != identity:
            raise ValueError("运行记录的训练身份不一致")
        torch.manual_seed(config["training"]["seed"])
        rng = np.random.default_rng(config["training"]["seed"])
        model = StunetAdapter.build(specification).to(device)
        settings = config["training"]
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"],
                                    nesterov=True, weight_decay=settings["weight_decay"])
        restored = False
        if state:
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            rng.bit_generator.state = state["sampler_rng"]
            restored = (tensors_equal(model.state_dict(), state["model"]) and
                        tensors_equal(optimizer.state_dict(), state["optimizer"]) and
                        torch.equal(torch.get_rng_state(), state["torch_rng"]) and
                        rng.bit_generator.state == state["sampler_rng"])
            if not restored:
                raise ValueError("模型、优化器或随机状态未完整恢复")
            attempt, updates, skipped = state["attempts"], state["updates"], state["skipped"]
            pretrained_receipt = state["pretrained_receipt"]
            del state
        else:
            pretrained_receipt = StunetAdapter.load_pretrained(model, config["pretrained"])
            attempt, updates, skipped = 0, 0, 0
            write_json(output / "run.json", {"identity": identity, "scientific": scientific, "config": config,
                                            "prepared": prepared, "pretrained_receipt": pretrained_receipt})
        dataset = CaseDataset(prepared, config["resources"])
        cleanup.callback(dataset.clear)
        before_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        stopping = {"requested": False}

        def request_stop(_signum, _frame):
            stopping["requested"] = True

        old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig, handler in old_handlers.items():
            cleanup.callback(signal.signal, sig, handler)
        event(output, "resume" if resume else "start", identity=identity, attempts=attempt, restored_exact=restored)
        timings = {"input_wait_seconds": 0., "compute_seconds": 0., "validation_seconds": 0., "checkpoint_seconds": 0.}
        last_loss = None
        while attempt < settings["attempts"] and not stopping["requested"]:
            if stop_after is not None and attempt >= stop_after:
                break
            if psutil.virtual_memory().available < config["resources"]["reserve_ram_bytes"]:
                stopping["requested"] = True
                event(output, "waiting_memory", available_bytes=psutil.virtual_memory().available)
                break
            mark = perf_counter()
            batch = get_batch(dataset, names, config, rng, device)
            timings["input_wait_seconds"] += perf_counter() - mark
            mark = perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with arithmetic(config, device):
                loss = StunetAdapter.training_step(model, batch, config["dataset"])
            if not torch.isfinite(loss):
                raise ValueError("训练损失为非有限数值")
            attempt += 1
            if (batch["label"] != -1).any():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            else:
                skipped += 1
            last_loss = float(loss.detach())
            timings["compute_seconds"] += perf_counter() - mark
            event(output, "training", attempts=attempt, updates=updates, loss=last_loss,
                  case_ids=batch["case_ids"], data_seconds=timings["input_wait_seconds"], compute_seconds=timings["compute_seconds"])
            del batch, loss
            if attempt % settings["save_every_attempts"] == 0:
                mark = perf_counter()
                save_checkpoint(output, {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "sampler_rng": rng.bit_generator.state, "attempts": attempt,
                    "updates": updates, "skipped": skipped, "spec": asdict(specification), "config": config,
                    "pretrained_receipt": pretrained_receipt})
                timings["checkpoint_seconds"] += perf_counter() - mark
        changed = any(not torch.equal(parameter.detach(), before_parameters[name]) for name, parameter in model.named_parameters())
        del before_parameters
        mark = perf_counter()
        validation_loss = None
        if not stopping["requested"]:
            model.eval()
            val_rng = np.random.default_rng(settings["seed"] + 1)
            batch = get_batch(dataset, validation, config, val_rng, device)
            with torch.no_grad(), arithmetic(config, device):
                value = StunetAdapter.training_step(model, batch, config["dataset"])
            if not torch.isfinite(value):
                raise ValueError("验证损失为非有限数值")
            validation_loss = float(value)
            del value, batch
        timings["validation_seconds"] += perf_counter() - mark
        mark = perf_counter()
        pointer = save_checkpoint(output, {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "sampler_rng": rng.bit_generator.state, "attempts": attempt,
            "updates": updates, "skipped": skipped, "spec": asdict(specification), "config": config,
            "pretrained_receipt": pretrained_receipt})
        timings["checkpoint_seconds"] += perf_counter() - mark
        summary = {"state": "completed" if attempt == settings["attempts"] else "stopped", "scope": config["scope"],
                   "identity": identity, "attempts": attempt, "updates": updates, "skipped": skipped,
                   "parameters_changed_this_invocation": changed, "restored_exact": restored,
                   "last_training_loss": last_loss, "validation_loss": validation_loss, "checkpoint": pointer,
                   "seconds": perf_counter() - started, "timing": timings, "cache_hits": dataset.hits,
                   "cache_misses": dataset.misses, "rss_bytes": psutil.Process().memory_info().rss,
                   "device": str(device), "peak_vram_bytes": None}
        write_json(output / "summary.json", summary)
        event(output, "stop", state=summary["state"], attempts=attempt, updates=updates)
        dataset.clear()
        return summary
