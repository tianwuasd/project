from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from hashlib import sha256
from importlib.metadata import version
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
from time import perf_counter

import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..artifacts.store import digest, operation_lock, read_json, require_output, write_json
from ..compatibility import compatible_code
from ..data.supply import BatchSupply
from ..models.stunet.adapter import StunetAdapter, WEIGHT_SHA256
from ..observability.events import event, logged, stage
from ..preprocessing.prepare import load_prepared, validate_prepared_config
from .checkpoint import load_checkpoint, save_checkpoint, tensors_equal
from .build_gate import BuildGate
from .resources import (available_world, close_members, host_capacity, layout, process_vram,
                        record_peaks, refresh_supply, register_member, resource_lease)


def code_identity():
    root = Path(__file__).parents[1]
    hashes = sha256()
    for path in sorted(root.rglob("*.py")):
        hashes.update(str(path.relative_to(root)).replace("\\", "/").encode())
        hashes.update(path.read_bytes())
    return hashes.hexdigest()


def validate_initialization_restart(output, scientific):
    output = Path(output)
    previous = read_json(output / "run.json")
    expected = {**scientific, "code_identity": previous["scientific"]["code_identity"]}
    if digest(previous["scientific"]) != digest(expected) or previous["identity"] != digest(expected) or not compatible_code(
            previous["scientific"]["code_identity"], scientific["code_identity"]):
        raise ValueError("初始化重试的配置、数据、代码或依赖版本改变")
    if (output / "latest.json").exists() or (output / "summary.json").exists() or any(output.glob("state-*.pt*")):
        raise FileExistsError("运行目录已有检查点、写入中的状态或训练摘要，请使用 --resume")
    event(output, "initialization_restarted", previous_identity=previous["identity"],
          identity=digest(scientific), previous_code=previous["scientific"]["code_identity"],
          applied_code=scientific["code_identity"], reason="已核对输入，尚无任何检查点或训练摘要")


def device_for(config):
    if config["resources"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，当前环境无法使用 CUDA")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def arithmetic(config, device):
    precision = config["training"]["precision"]
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("混合精度需要 CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("所选显卡不支持原生 BF16")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def all_objects(value, world):
    if world == 1:
        return [value]
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def any_rank(value, device, world):
    flag = torch.tensor(int(value), device=device)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def supply_settings(plan):
    return {key: plan[key] for key in ("workers", "cache_bytes", "reader_slots", "prefetch")}


def rebalance_supply(job, supply, gate, device, rank, stop_event):
    world, config = job["world"], job["config"]
    proposal = refresh_supply(config, job["prepared"], job) if rank == 0 else None
    proposal = all_objects(proposal, world)[0]
    if proposal["admitted"] and supply_settings(proposal) == supply_settings(job["supply"]):
        return "unchanged"
    if rank == 0:
        event(config["run_dir"], "supply_resize_proposed", previous=supply_settings(job["supply"]), proposed=proposal)
    # 所有 rank 先结束旧池，未消费的确定性请求由全局游标重新派发。
    supply.__exit__(None, None, None)
    if world > 1:
        dist.barrier()
    proposal = refresh_supply(config, job["prepared"], job, apply=True) if rank == 0 else None
    proposal = all_objects(proposal, world)[0]
    if not proposal["admitted"]:
        return "pressure"
    if rank == 0:
        gate.set_limit(proposal["reader_slots"])
    if world > 1:
        dist.barrier()
    job["supply"] = proposal
    supply.workers, supply.cache_bytes = proposal["workers"][rank], proposal["cache_bytes"][rank]
    supply.__enter__()
    if rank == 0:
        event(config["run_dir"], "supply_resize_applied", applied=proposal,
              replay="从已消费的全局游标重放预取", model_retained=True)
    return "applied"


def checkpoint(output, model, optimizer, scaler, progress, job, receipt, device, rank):
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
    states = all_objects(rng, job["world"])
    pointer = None
    if rank == 0:
        pointer = save_checkpoint(output, {"identity": job["identity"], "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rank_rng": states,
            "torch_rng": states[0]["torch"], "sampler_rng": {"protocol": "global-slot-v1", "next": progress["attempts"]},
            "attempts": progress["attempts"], "updates": progress["updates"], "skipped": progress["skipped"],
            "progress": dict(progress), "spec": job["spec"], "config": job["config"], "pretrained_receipt": receipt})
    return all_objects(pointer, job["world"])[0]


@logged("training_rank", config_argument=1, config_key="config")
def run_rank(rank, job, gate, stop_event):
    config, world, output = job["config"], job["world"], Path(job["config"]["run_dir"])
    settings, resources = config["training"], config["resources"]
    register_member(config, job["owner"])
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if job["gpus"] else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        observed_uuid = str(torch.cuda.get_device_properties(device).uuid)
        observed_uuid = observed_uuid if observed_uuid.startswith("GPU-") else "GPU-" + observed_uuid
        if observed_uuid.lower() != job["gpus"][rank]["uuid"].lower():
            raise ValueError("CUDA 设备 UUID 与已准入设备不同")
        torch.cuda.reset_peak_memory_stats(device)
    if world > 1:
        binding = {"device_id": device} if device.type == "cuda" else {}
        with stage(config, "distributed_init", rank=rank, world_size=world, backend="nccl" if job["gpus"] else "gloo"):
            dist.init_process_group("nccl" if job["gpus"] else "gloo", init_method=job["rendezvous"],
                                    rank=rank, world_size=world, timeout=timedelta(seconds=180), **binding)
    try:
        with ExitStack() as cleanup:
            old_handlers = {sig: signal.signal(sig, lambda *_: stop_event.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
            for sig, handler in old_handlers.items():
                cleanup.callback(signal.signal, sig, handler)
            torch.manual_seed(settings["seed"])
            with stage(config, "model_build", rank=rank, device=str(device), spec=job["spec"]):
                model = StunetAdapter.build(job["spec"]).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"],
                                        nesterov=True, weight_decay=settings["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["precision"] == "fp16")
            state = load_checkpoint(output, job["identity"]) if job["resume"] else None
            restored = False
            if state:
                if "progress" not in state or not state.get("rank_rng"):
                    raise ValueError("恢复训练需要完整进度及逐 rank 随机状态")
                old_world = len(state["rank_rng"])
                changed_world = old_world != world
                boundary = state["progress"]["phase"] == "train" and (
                    state["attempts"] == 0 or settings["epochs"] is not None and
                    state["attempts"] % settings["steps_per_epoch"] == 0)
                if changed_world and not boundary:
                    raise ValueError("进程数量变化需要完整 epoch 检查点")
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                scaler.load_state_dict(state["scaler"])
                rng = state["rank_rng"][rank] if rank < old_world else None
                if rng is None:
                    seed = int.from_bytes(sha256(state["rank_rng"][0]["torch"].numpy().tobytes() +
                        f":{world}:{rank}:{state['attempts']}".encode()).digest()[:8], "little") % (2 ** 63)
                    torch.manual_seed(seed)
                    rng = {"torch": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
                torch.set_rng_state(rng["torch"])
                if device.type == "cuda":
                    torch.cuda.set_rng_state(rng["cuda"], device)
                restored = (tensors_equal(model.state_dict(), state["model"]) and
                            tensors_equal(optimizer.state_dict(), state["optimizer"]) and
                            tensors_equal(scaler.state_dict(), state["scaler"]) and torch.equal(torch.get_rng_state(), rng["torch"]))
                if device.type == "cuda":
                    restored = restored and torch.equal(torch.cuda.get_rng_state(device), rng["cuda"])
                if not restored:
                    raise ValueError("模型、优化器、scaler 或随机状态恢复不完整")
                restored_state = restored
                restored = restored and not changed_world
                if changed_world and rank == 0:
                    event(output, "rank_count_changed", previous_world=old_world, applied_world=world,
                          optimizer_restored=True, scaler_restored=True, sampler=state["sampler_rng"],
                          rng_policy="保留已有 rank 状态，新增 rank 使用检查点派生种子",
                          reduction="全局样本统计归约；浮点加和顺序可能变化", bitwise_equivalent=False)
                progress, receipt = state["progress"], state["pretrained_receipt"]
                del state
            else:
                restored_state = False
                with stage(config, "pretrained_load", rank=rank, path=config["pretrained"]):
                    receipt = StunetAdapter.load_pretrained(model, config["pretrained"])
                progress = {"attempts": 0, "updates": 0, "skipped": 0, "epoch": 0, "phase": "train", "val_step": 0,
                            "val_sum": 0., "validation_loss": None}
            before_parameter = next(model.parameters()).detach().cpu().clone()
            with stage(config, "model_ready", rank=rank, distributed=world > 1):
                wrapped = (DistributedDataParallel(model, device_ids=[rank] if job["gpus"] else None,
                            broadcast_buffers=False, find_unused_parameters=True) if world > 1 else model)
            # 最后一个深监督输出权重为零，DDP 需要识别没有梯度的参数。
            rows = job["prepared"]["manifest"]["cases"]
            train_ids = [r["case_id"] for r in rows if "train" in r["roles"]]
            val_ids = [r["case_id"] for r in rows if "validation" in r["roles"]]
            offset = sum(job["shares"][:rank])
            slots = list(range(offset, offset + job["shares"][rank]))
            supply = cleanup.enter_context(BatchSupply(job["prepared"], config, slots,
                job["supply"]["workers"][rank], job["supply"]["cache_bytes"][rank], gate, job["owner"]))
            if rank == 0:
                event(output, "resume" if job["resume"] else "start", identity=job["identity"], restored_exact=restored,
                      attempts=progress["attempts"], global_batch=settings["batch"], layout=job["shares"],
                      devices=job["gpus"], cpu=job["supply"])
            if not job["resume"]:
                checkpoint(output, model, optimizer, scaler, progress, job, receipt, device, rank)
            started = perf_counter()
            timings = {"input_wait_seconds": 0., "compute_seconds": 0., "validation_seconds": 0., "checkpoint_seconds": 0.}
            last_loss, changed, reason, completed = None, False, None, False
            peak_reserved, peak_non_allocator = 0, 0
            queued_phase, submitted, phase_end = None, 0, 0
            next_adapt = perf_counter() + resources["adapt_seconds"]
            next_growth = perf_counter() + resources["growth_seconds"]
            total = settings["epochs"] * settings["steps_per_epoch"] if settings["epochs"] else settings["attempts"]
            while True:
                completed = progress["epoch"] >= (settings["epochs"] or 1)
                if completed:
                    break
                if any_rank(stop_event.is_set(), device, world):
                    reason = "resource_pressure" if job["pressure_event"].is_set() else "signal"
                    break
                if job["stop_after"] is not None and progress["attempts"] >= job["stop_after"] and progress["phase"] == "train":
                    reason = "bounded_stop"
                    break
                pressure = host_capacity()["available_bytes"] < resources["reserve_ram_bytes"]
                if device.type == "cuda":
                    free, total_vram = torch.cuda.mem_get_info(device)
                    pressure |= free < max(resources["vram_reserve_bytes"], total_vram * resources["vram_reserve_fraction"])
                if any_rank(pressure, device, world):
                    reason = "resource_pressure"
                    break
                resize_due = rank == 0 and perf_counter() >= next_adapt
                if any_rank(resize_due, device, world):
                    if rank == 0 and job["gpus"]:
                        available_world(config, job["spec"], job)
                    candidate = refresh_supply(config, job["prepared"], job) if rank == 0 else None
                    candidate = all_objects(candidate, world)[0]
                    growing = candidate["admitted"] and (sum(candidate["workers"]) > sum(job["supply"]["workers"]) or
                        candidate["reader_slots"] > job["supply"]["reader_slots"] or
                        sum(candidate["cache_bytes"]) > sum(job["supply"]["cache_bytes"]))
                    permitted = rank == 0 and (not growing or perf_counter() >= next_growth)
                    if any_rank(permitted, device, world):
                        resized = rebalance_supply(job, supply, gate, device, rank, stop_event)
                        if resized == "pressure":
                            reason = "resource_pressure"
                            break
                        if resized == "applied":
                            queued_phase = None
                        next_growth = perf_counter() + resources["growth_seconds"]
                    next_adapt = perf_counter() + resources["adapt_seconds"]
                phase = progress["phase"]
                cursor = progress["attempts"] if phase == "train" else progress["val_step"]
                if queued_phase != (phase, progress["epoch"]):
                    if supply.queue:
                        raise RuntimeError("阶段切换时供数队列未消费完毕")
                    queued_phase = (phase, progress["epoch"])
                    submitted = cursor
                    if phase == "train":
                        phase_end = min(total, (progress["epoch"] + 1) * settings["steps_per_epoch"]) if settings["epochs"] else total
                        if job["stop_after"] is not None:
                            phase_end = min(phase_end, job["stop_after"])
                    else:
                        phase_end = settings["validation_steps"] if settings["epochs"] else 1
                    if rank == 0:
                        event(output, "phase_started", phase=phase, epoch=progress["epoch"] + 1,
                              epochs=settings["epochs"], completed=cursor, total=phase_end)
                depth = job["supply"]["prefetch"]
                while submitted < phase_end and len(supply.queue) < depth:
                    sample_index = submitted if phase == "train" else progress["epoch"] * settings["validation_steps"] + submitted
                    supply.submit(train_ids if phase == "train" else val_ids, 0 if phase == "train" else 1, sample_index)
                    submitted += 1
                if not supply.queue:
                    reason = "bounded_stop"
                    break
                mark = perf_counter()
                with stage(config, "batch_supply", console=False, rank=rank, phase=phase,
                           epoch=progress["epoch"] + 1, step=cursor + 1, total=phase_end):
                    batch = supply.take(device)
                timings["input_wait_seconds"] += perf_counter() - mark
                mark = perf_counter()
                if phase == "train":
                    wrapped.train()
                    if settings["epochs"]:
                        lr = settings["learning_rate"] * (1 - progress["epoch"] / settings["epochs"]) ** settings["poly_power"]
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    with arithmetic(config, device):
                        loss = StunetAdapter.training_step(wrapped, batch, config["dataset"])
                    if any_rank(not bool(torch.isfinite(loss)), device, world):
                        raise ValueError("训练损失出现非有限数值")
                    progress["attempts"] += 1
                    if any_rank(bool((batch["label"] != -1).any()), device, world):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12., error_if_nonfinite=not scaler.is_enabled())
                        overflow = any_rank(not bool(torch.isfinite(norm)), device, world) if scaler.is_enabled() else False
                        if scaler.is_enabled() and overflow:
                            # 所有进程共同跳过溢出的更新，降低 scale 后继续既定尝试预算。
                            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
                            progress["skipped"] += 1
                        else:
                            scaler.step(optimizer)
                            scaler.update()
                            progress["updates"] += 1
                            changed = True
                    else:
                        progress["skipped"] += 1
                    last_loss = float(loss.detach())
                else:
                    wrapped.eval()
                    with torch.no_grad(), arithmetic(config, device):
                        loss = StunetAdapter.training_step(wrapped, batch, config["dataset"])
                    if any_rank(not bool(torch.isfinite(loss)), device, world):
                        raise ValueError("验证损失出现非有限数值")
                    progress["val_sum"] += float(loss)
                    progress["val_step"] += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                    actual_vram = process_vram(job["gpus"][rank]["uuid"], os.getpid())
                    peak_non_allocator = max(peak_non_allocator, actual_vram - torch.cuda.memory_reserved(device))
                elapsed = perf_counter() - mark
                timings["compute_seconds" if phase == "train" else "validation_seconds"] += elapsed
                if rank == 0:
                    event(output, phase, attempts=progress["attempts"], updates=progress["updates"], epoch=progress["epoch"],
                          skipped=progress["skipped"],
                          loss=float(loss.detach()), local_cases=batch["case_ids"], learning_rate=optimizer.param_groups[0]["lr"],
                          input_wait_seconds=timings["input_wait_seconds"], compute_seconds=elapsed,
                          local_cache_hits=batch["cache_hits"], local_cache_misses=batch["cache_misses"],
                          local_supply_seconds=batch["supply_seconds"], validation_step=progress["val_step"],
                          total_attempts=total, validation_steps=settings["validation_steps"] if settings["epochs"] else 1)
                del batch, loss
                if settings["epochs"] and phase == "train" and progress["attempts"] % settings["steps_per_epoch"] == 0:
                    progress.update(phase="validation", val_step=0, val_sum=0.)
                if not settings["epochs"] and phase == "train" and progress["attempts"] == total:
                    progress.update(phase="validation", val_step=0, val_sum=0.)
                val_steps = settings["validation_steps"] if settings["epochs"] else 1
                if phase == "validation" and progress["val_step"] == val_steps:
                    progress.update(epoch=progress["epoch"] + 1, phase="train",
                                    validation_loss=progress["val_sum"] / val_steps)
                    if rank == 0:
                        event(output, "epoch", epoch=progress["epoch"], validation_loss=progress["validation_loss"])
                if (phase == "train" and progress["attempts"] % settings["save_every_attempts"] == 0) or \
                        (phase == "validation" and progress["phase"] == "train"):
                    mark = perf_counter()
                    checkpoint(output, model, optimizer, scaler, progress, job, receipt, device, rank)
                    timings["checkpoint_seconds"] += perf_counter() - mark
                if phase == "validation" and progress["phase"] == "train" and progress["epoch"] < (settings["epochs"] or 1):
                    peaks = all_objects(peak_reserved + peak_non_allocator, world)
                    if rank == 0:
                        record_peaks(config, job["spec"], job["gpus"], job["shares"], peaks)
                    decision = {"world": available_world(config, job["spec"], job),
                                "replacement": job.get("fleet_replacement", False)} if rank == 0 else None
                    decision = all_objects(decision, world)[0]
                    desired = decision["world"]
                    if desired != world or decision["replacement"]:
                        reason = "regroup"
                        if rank == 0:
                            event(output, "gpu_regroup_requested", previous_world=world, desired_world=desired,
                                  minimum=resources["min_gpus"], target=resources["gpu_count"], epoch=progress["epoch"])
                        break
            mark = perf_counter()
            pointer = checkpoint(output, model, optimizer, scaler, progress, job, receipt, device, rank)
            timings["checkpoint_seconds"] += perf_counter() - mark
            changed = changed and not torch.equal(next(model.parameters()).detach().cpu(), before_parameter)
            measurements = all_objects({"rank": rank, "timing": timings, "rss_bytes": psutil.Process().memory_info().rss,
                                       "peak_vram_bytes": peak_reserved + peak_non_allocator if job["gpus"] else None,
                                       "peak_reserved_bytes": peak_reserved if job["gpus"] else None}, world)
            if rank == 0:
                record_peaks(config, job["spec"], job["gpus"], job["shares"],
                             [row["peak_vram_bytes"] for row in measurements] if job["gpus"] and completed else [])
                summary = {"state": "completed" if completed else "stopped", "reason": reason, "scope": config["scope"],
                    "identity": job["identity"], **progress, "restored_exact": restored,
                    "restored_training_state": restored_state, "checkpoint": pointer,
                    "parameters_changed_this_invocation": changed, "last_training_loss": last_loss,
                    "seconds": perf_counter() - started, "timing": timings, "world_size": world,
                    "global_batch": settings["batch"], "batch_layout": job["shares"], "devices": job["gpus"],
                    "workers": job["supply"]["workers"], "ranks": measurements, "device": str(device),
                    "peak_vram_bytes": max(row["peak_vram_bytes"] for row in measurements) if job["gpus"] else None}
                write_json(output / "summary.json", summary)
                event(output, "stop", state=summary["state"], attempts=progress["attempts"], reason=reason)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@logged("training")
def train(config, prepared_path, resume=False, stop_after=None):
    with stage(config, "training_inputs", prepared=str(prepared_path), resume=resume):
        prepared = load_prepared(prepared_path)
        validate_prepared_config(prepared, config)
    output = require_output(config["run_dir"], config["work_root"], [prepared["manifest"]["data_root"]])
    rows = prepared["manifest"]["cases"]
    if not all(any(role in r["roles"] for r in rows) for role in ("train", "validation")) or \
            any(r.get("label") is None for r in rows if set(r["roles"]) & {"train", "validation"}):
        raise ValueError("训练与验证需要非空的图像、标签及角色")
    spec = asdict(StunetAdapter.spec(config, prepared["plan"]))
    scientific = {"prepared_identity": prepared["identity"], "model": "stunet", "spec": spec,
        "training": config["training"], "pretrained": WEIGHT_SHA256, "code_identity": code_identity(),
        "versions": {n: version(n) for n in ("torch", "monai", "nnunetv2", "numpy", "scipy")}}
    identity = digest(scientific)
    with operation_lock(output):
        if resume:
            previous = read_json(output / "run.json")
            expected = {**scientific, "code_identity": previous["scientific"]["code_identity"]}
            if digest(previous["scientific"]) != digest(expected) or previous["identity"] != digest(expected) or not compatible_code(
                    previous["scientific"]["code_identity"], scientific["code_identity"]):
                raise ValueError("恢复训练的配置、数据、代码或依赖版本改变")
            identity = previous["identity"]
            if previous["scientific"]["code_identity"] != scientific["code_identity"]:
                event(output, "code_compatibility_verified", source=previous["scientific"]["code_identity"],
                      target=scientific["code_identity"], training_identity=identity)
            load_checkpoint(output, identity)
        elif (output / "run.json").exists():
            validate_initialization_restart(output, scientific)
        pressure_retries = 0
        while True:
            admission = deepcopy(config)
            old_world = None
            if resume:
                previous_state = load_checkpoint(output, identity)
                old_world = len(previous_state["rank_rng"])
                boundary = previous_state["progress"]["phase"] == "train" and (previous_state["attempts"] == 0 or
                    config["training"]["epochs"] is not None and
                    previous_state["attempts"] % config["training"]["steps_per_epoch"] == 0)
                if not boundary and config["resources"]["device"] == "cuda":
                    admission["resources"].update(gpu_count=old_world, min_gpus=None)
                    event(output, "fixed_world_resume", world=old_world, reason="当前完整检查点位于 epoch 内部")
                del previous_state
            with resource_lease(admission, prepared, spec, previous_world=old_world) as allocation:
                if not resume:
                    write_json(output / "run.json", {"identity": identity, "scientific": scientific,
                               "config": config, "prepared": prepared})
                summary = launch_group(config, prepared, spec, identity, allocation, resume, stop_after)
            if summary["reason"] not in ("resource_pressure", "regroup"):
                return summary
            if summary["reason"] == "resource_pressure":
                pressure_retries += 1
                if pressure_retries > config["resources"]["pressure_retries"]:
                    raise MemoryError("达到已配置的内存压力恢复次数；完整检查点已保留")
                config["resources"]["max_workers"] = max(1, config["resources"]["max_workers"] // 2)
                config["resources"]["cache_bytes"] //= 2
                config["resources"]["prefetch"] = max(1, config["resources"]["prefetch"] // 2)
                config["resources"]["max_readers"] = max(1, config["resources"]["max_readers"] // 2)
            event(output, "training_readmission", reason=summary["reason"], pressure_retries=pressure_retries,
                  progress={key: summary[key] for key in ("epoch", "attempts", "updates", "phase")},
                  requested_resources=config["resources"])
            resume = True


def launch_group(config, prepared, spec, identity, allocation, resume, stop_after):
    world = allocation["world"]
    if resume:
        state = load_checkpoint(config["run_dir"], identity)
        boundary = state["progress"]["phase"] == "train" and (state["attempts"] == 0 or
            config["training"]["epochs"] is not None and state["attempts"] % config["training"]["steps_per_epoch"] == 0)
        if len(state["rank_rng"]) != world and not boundary:
            raise ValueError("当前检查点处于 epoch 内部，恢复时需要原进程数量")
        del state
    shares = layout(config["training"]["batch"], world, config["resources"]["batch_layout"])
    with socket.socket() as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        port = rendezvous.getsockname()[1]
    job = {**allocation, "config": config, "prepared": prepared, "spec": spec, "identity": identity,
           "shares": shares, "resume": resume, "stop_after": stop_after, "rendezvous": f"tcp://127.0.0.1:{port}"}
    context = mp.get_context("spawn")
    gate, stop_event = BuildGate(context, allocation["supply"]["reader_slots"]), context.Event()
    job["pressure_event"] = context.Event()
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    job["allowed_visible_devices"] = previous_visible
    if allocation["gpus"]:
        if torch.cuda.is_initialized():
            raise RuntimeError("协调进程已初始化 CUDA，请使用新的训练命令启动")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in allocation["gpus"])
    env_previous = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    for key in env_previous:
        os.environ[key] = "1"
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop_event.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        processes = torch.multiprocessing.spawn(run_rank, args=(job, gate, stop_event), nprocs=world, join=False)
        while not processes.join(timeout=1):
            facts = host_capacity()
            if facts["available_bytes"] < config["resources"]["reserve_ram_bytes"]:
                if not job["pressure_event"].is_set():
                    event(config["run_dir"], "host_pressure_detected", facts=facts,
                          reserve_bytes=config["resources"]["reserve_ram_bytes"])
                job["pressure_event"].set()
                stop_event.set()
    finally:
        close_members(config, allocation["owner"])
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for key, value in {**env_previous, "CUDA_VISIBLE_DEVICES": previous_visible}.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return read_json(Path(config["run_dir"]) / "summary.json")
