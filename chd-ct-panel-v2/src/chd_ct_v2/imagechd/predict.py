"""Image-only multi-stage inference; prediction never opens cached targets."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..cpu_runtime import allocation, iter_cases, volume_bytes
from ..recovery import atomic_path, output_lock, read_json
from .checkpoints import load_stage, read_collection
from .common import LABELS, load_case, read_prepared, write_json
from .config import UNAVAILABLE, build_model
from .folds import select_members, select_rows
from .fusion import weighted_vote
from .geometry import resize
from .gpu_memory import check_memory
from .methods import PROCESSING, require_aligned_cache
from .spacing import sample_grid
from .train import file_hash


def inference_members(directory, collection):
    return collection.get("_members", [{"directory": directory, "collection": collection}])


def anatomy_probabilities(image, stage, directory, collection, device):
    check_memory()
    model, record = load_stage(directory, collection, stage, device)
    size = record["spec"]["size"]
    with torch.inference_mode():
        tensor = torch.from_numpy(resize(image, (size,) * 3))[None, None].to(device)
        result = model(tensor).softmax(1)[0].cpu().numpy()
        check_memory()
        return result


def anatomy(image, stage, directory, collection, device):
    total = None
    for member in inference_members(directory, collection):
        probabilities = anatomy_probabilities(image, stage, member["directory"], member["collection"], device)
        if total is None:
            total = probabilities
        else:
            total += probabilities
    # Dividing by the member count cannot change the argmax.
    return total.argmax(0).astype(np.uint8)


def blood_probabilities(image, directory, collection, device):
    """Yield one slice at a time from this fold's recurrent/encoder pair."""
    check_memory()
    recurrent, record = load_stage(directory, collection, "blood_lstm", device)
    encoder = build_model("blood2d", record["encoder_spec"])
    encoder.load_state_dict(record["encoder_state_dict"])
    encoder.to(device).eval()
    size, sequence = record["spec"]["size"], record["spec"]["sequence"]
    # Cache only one sequence of feature maps, bounding memory on native volumes.
    features = {}
    with torch.inference_mode():
        for z in range(image.shape[2]):
            check_memory()
            indices = np.clip(np.arange(z - sequence // 2, z + sequence // 2 + 1), 0, image.shape[2] - 1)
            needed = set(int(i) for i in indices)
            features = {i: value for i, value in features.items() if i in needed}
            for i in needed:
                if i not in features:
                    x = torch.from_numpy(resize(image[:, :, i], (size, size)))[None, None].to(device)
                    features[i] = encoder.features(x)
            window = torch.stack([features[int(i)] for i in indices], dim=1)
            probabilities = recurrent(window).softmax(1)[0].cpu().numpy()
            check_memory()
            yield z, probabilities


def blood_prediction(image, directory, collection, device):
    members = inference_members(directory, collection)
    size = collection["config"]["stages"]["blood_lstm"]["size"]
    small = np.empty((size, size, image.shape[2]), np.uint8)
    if len(members) == 1:
        member = members[0]
        for z, probabilities in blood_probabilities(image, member["directory"], member["collection"], device):
            small[:, :, z] = probabilities.argmax(0)
    else:
        # One fold on GPU at a time. A temporary disk-backed accumulator avoids
        # holding three 2D encoders or a whole 512x512xdepth probability volume
        # in Python RAM. Slice-major layout keeps updates contiguous.
        with tempfile.TemporaryFile() as scratch:
            shape = (image.shape[2], 3, size, size)
            scratch.truncate(int(np.prod(shape)) * np.dtype(np.float32).itemsize)
            totals = np.memmap(scratch, dtype=np.float32, mode="r+", shape=shape)
            try:
                for index, member in enumerate(members):
                    for z, probabilities in blood_probabilities(
                        image, member["directory"], member["collection"], device
                    ):
                        if index == 0:
                            totals[z] = probabilities
                        else:
                            totals[z] += probabilities
                for z in range(image.shape[2]):
                    small[:, :, z] = totals[z].argmax(0)
            finally:
                totals._mmap.close()
    return resize(small, image.shape, labels=True).astype(np.uint8)


def predict_volume(image, directory, collection, device, *, return_stages=False, policy_override=None):
    from .calibration import load_policy
    from .postprocessing import apply_policy, limited_growth
    from .roi_inputs import locate

    policy = policy_override if policy_override is not None else load_policy(directory, collection)
    crops, roi, details = locate(image, directory, collection, device)
    cropped = image[roi]
    size = collection["config"]["stages"]["all128"]["size"]
    aligned = (size,) * 3
    votes = [resize(result[roi], aligned, labels=True) for result in crops]
    for stage in ("all64", "all128"):
        votes.append(resize(anatomy(cropped, stage, directory, collection, device), aligned, labels=True))
    fused = resize(weighted_vote(votes), cropped.shape, labels=True).astype(np.uint8)
    blood = blood_prediction(cropped, directory, collection, device)
    limited = limited_growth(fused, blood, policy["max_growth_steps"])
    result = np.zeros(image.shape, np.uint8)
    result[roi] = apply_policy(fused, limited, policy)
    details["postprocessing_policy"] = policy
    details["stage_volume_voxels"] = {
        name: {str(label): int((a == label).sum()) for label in range(1, 8)}
        for name, a in (("vote", fused), ("limited", limited), ("final", result))
    }
    if return_stages:
        arrays = {}
        for name, value in (("vote", fused), ("limited", limited), ("blood", blood)):
            arrays[name] = np.zeros(image.shape, np.uint8)
            arrays[name][roi] = value
        details["stage_outputs"] = arrays
    return result, details


def restore_native(prediction, row):
    if row.get("resampling"):
        geometry = row["resampling"]
        resampled = resize(prediction, geometry["resampled_shape"], labels=True).astype(np.uint8)
        canonical = sample_grid(
            resampled,
            row["canonical_shape"],
            np.linalg.inv(np.asarray(geometry["output_to_canonical"])),
            labels=True,
        )
    else:
        canonical = resize(prediction, row["canonical_shape"], labels=True).astype(np.uint8)
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ras, np.asarray(row["native_orientation"]))
    native = nib.orientations.apply_orientation(canonical, transform)
    if list(native.shape) != row["native_shape"]:
        raise ValueError("还原尺寸失败")
    result = nib.Nifti1Image(native, np.asarray(row["native_affine"]))
    result.set_sform(np.asarray(row["native_affine"]), code=1)
    result.set_qform(np.asarray(row["native_affine"]), code=0)
    result.header.set_xyzt_units(*row["units"])
    return result


def predict(
    prepared,
    models,
    output,
    device="cpu",
    case_id=None,
    split=None,
    allow_smoke=False,
    resume=False,
    gpu_ids=None,
    gpu_limit=None,
    phase="all",
    inherited_lock_fds=(),
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output) as lock_fd:
        return predict_locked(
            prepared,
            models,
            output,
            device,
            case_id,
            split,
            allow_smoke,
            resume,
            gpu_ids,
            gpu_limit,
            phase,
            lock_fd,
            inherited_lock_fds,
        )


def predict_locked(
    prepared,
    models,
    output,
    device,
    case_id,
    split,
    allow_smoke,
    resume,
    gpu_ids=None,
    gpu_limit=None,
    phase="all",
    lock_fd=None,
    inherited_lock_fds=(),
):
    worker_locks = (*inherited_lock_fds, *((lock_fd,) if lock_fd is not None else ()))
    if phase not in {"all", "gpu", "cpu"}:
        raise ValueError("无效预测阶段")
    cache, data = read_prepared(prepared)
    require_aligned_cache(data)
    directory, collection = read_collection(models, allow_smoke)
    plan, fold = collection.get("split_plan"), collection["config"].get("fold")
    requested_split = split
    rows, split = select_rows(data, split, plan=plan, fold=fold)
    rows = [r for r in rows if case_id is None or r["case_id"] == case_id]
    if not rows:
        raise ValueError("没有匹配的预测病例")
    output = Path(output).expanduser().resolve()

    report = {
        "status": "running",
        "task": "chd_multistage_prediction",
        "processing_method": PROCESSING,
        "labels": list(LABELS),
        "prepared_sha256": file_hash(cache / "dataset.json"),
        "models_sha256": file_hash(directory / "models.json"),
        "model_mode": collection["mode"],
        "segmentation_prepared_sha256": collection["prepared_sha256"],
        "cache_resolution": data["resolution"],
        "split": split,
        "split_plan": plan,
        "fold_context": fold,
        "cases": [],
        "unavailable": UNAVAILABLE,
        "geometry": "supplied original grid restored; millimetre calibration unverified",
    }
    contract = {k: report[k] for k in ("prepared_sha256", "models_sha256", "model_mode")}
    contract.update(
        processing_method=PROCESSING,
        case_ids=[r["case_id"] for r in rows],
        split=split,
        case_id=case_id,
        cache_hashes={r["case_id"]: file_hash(cache / r["cache"]) for r in rows},
        recovery_version=1,
    )
    contract_file = output / "prediction-contract.json"
    if contract_file.exists() and read_json(contract_file) != contract:
        raise ValueError("预测输入、病例范围或模型改变，不能混用旧结果")
    previous_file = output / "prediction-report.json"
    previous = read_json(previous_file) if resume and previous_file.exists() else {}
    if previous and any(previous.get(k) != report[k] for k in ("prepared_sha256", "models_sha256")):
        raise ValueError("旧预测来源不同，不能续跑")
    done = {r["case_id"]: r for r in previous.get("cases", [])}
    report["cases"] = [done[r["case_id"]] for r in rows if r["case_id"] in done]
    write_json(contract_file, contract)
    scratch = output / ".gpu-results"
    scratch.mkdir(exist_ok=True)
    write_json(
        scratch / "context.json",
        dict(
            cache=str(cache), directory=str(directory), collection=collection, output=str(output), rows=rows
        ),
    )
    try:
        threads = int(os.environ.get("CHD_CPU_THREADS", collection["config"]["threads"]))
        torch.set_num_threads(threads)
        valid = {row["case_id"] for row in rows if valid_record(output, done.get(row["case_id"]))}
        remaining = [row for row in rows if row["case_id"] not in valid]
        needed = [row for row in remaining if not valid_grid(scratch, row["case_id"])]
        if phase == "cpu" and needed:
            raise ValueError("GPU 中间结果缺失或校验失败；请先续跑预测阶段")
        if phase != "cpu" and needed:
            if gpu_ids:
                from .gpu_queue import options
                from .stage_queue import run_queue

                selected = allocation("gpu_inference", output, len(needed), volume_bytes(needed, 64))
                queue_options = options(gpu_ids, gpu_limit, device, "predict")
                queue_options["gpu_limit"] = (
                    min(gpu_limit or len(gpu_ids), selected["processes"])
                    if os.environ.get("CHD_CPU_AUTO") == "1"
                    else gpu_limit or len(gpu_ids)
                )
                from ..progress import Progress

                progress = Progress(output, "gpu_inference", len(needed))
                peak_active = 0

                def observe(state):
                    nonlocal peak_active
                    peak_active = max(peak_active, state["active_gpus"])
                    state["peak_active_gpus"] = peak_active
                    progress.update(
                        sum(r["status"] == "complete" for r in state["stages"].values()), state["status"]
                    )

                tokens = {f"case_{index:04d}": row for index, row in enumerate(needed)}

                def command(token):
                    return [
                        sys.executable,
                        "-m",
                        "chd_ct_v2.imagechd.inference_worker",
                        str(scratch / "context.json"),
                        tokens[token]["case_id"],
                    ]

                run_queue(
                    list(tokens),
                    output=output,
                    command_for=command,
                    env=dict(os.environ, CHD_CPU_THREADS=str(threads)),
                    lock_fds=worker_locks,
                    dependencies={},
                    result_path_for=lambda token: scratch / (tokens[token]["case_id"] + ".json"),
                    on_state=observe,
                    **queue_options,
                )
            elif device == "cpu" and os.environ.get("CHD_CPU_AUTO") == "1":
                selected = allocation("cpu_inference", output, len(needed), volume_bytes(needed, 64))
                items = [
                    dict(
                        cache=str(cache),
                        directory=str(directory),
                        collection=collection,
                        output=str(output),
                        row=row,
                        device=device,
                    )
                    for row in needed
                ]
                list(
                    iter_cases(
                        "chd_ct_v2.imagechd.predict",
                        "infer_case",
                        items,
                        output,
                        selected,
                        worker_locks,
                    )
                )
            else:
                for row in needed:
                    infer_case(str(cache), str(directory), collection, str(output), row, device)
                    if phase == "all":
                        record = finish_case(str(output), row)
                        done[row["case_id"]] = record
                        valid.add(row["case_id"])
                        report["cases"] = [done[r["case_id"]] for r in rows if r["case_id"] in done]
                        write_json(output / "prediction-report.json", report)
                        # Direct serial API keeps its original per-case durable boundary.
                        (scratch / (row["case_id"] + ".json")).unlink()
                        (scratch / (row["case_id"] + ".npy")).unlink()
        remaining = [row for row in remaining if row["case_id"] not in valid]
        if phase != "cpu":
            case_files = {}
            for row in rows:
                case = row["case_id"]
                if case in valid:
                    record = done[case]
                    case_files[case] = {
                        record[key]: record[key + "_sha256"] for key in ("prediction", "grid_prediction")
                    }
                    case_files[case].update(
                        {v["file"]: v["sha256"] for v in record.get("stage_predictions", {}).values()}
                    )
                else:
                    info_path = scratch / (case + ".json")
                    info = read_json(info_path)
                    case_files[case] = {
                        ".gpu-results/" + info["raw_grid"]: info["raw_sha256"],
                        ".gpu-results/" + info_path.name: file_hash(info_path),
                    }
                    if info.get("raw_stages"):
                        case_files[case][".gpu-results/" + info["raw_stages"]] = info["raw_stages_sha256"]
            write_json(
                output / "gpu-phase-ready.json",
                dict(
                    contract=contract,
                    cases=case_files,
                    request=dict(split=requested_split, case_id=case_id),
                ),
            )
        if phase == "gpu":
            report.update(status="pending_cpu", gpu_cases_completed=len(rows))
            return report
        selected = allocation(
            "prediction_postprocess", output, len(remaining), volume_bytes(remaining, 48, native=True)
        )
        items = [dict(output=str(output), row=row) for row in remaining]
        for record in iter_cases(
            "chd_ct_v2.imagechd.predict",
            "finish_case",
            items,
            output,
            selected,
            worker_locks,
        ):
            done[record["case_id"]] = record
            report["cases"] = [done[r["case_id"]] for r in rows if r["case_id"] in done]
            write_json(output / "prediction-report.json", report)
            print("已预测并保存：" + record["case_id"], flush=True)
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "prediction-report.json", report)


def valid_record(output, record):
    if not (
        bool(record)
        and record.get("stage_outputs_format") == "ct-panel-v2-imagechd-stage-outputs-v1"
        and all(
            matches_file(output, record.get(key), record.get(key + "_sha256"))
            for key in ("prediction", "grid_prediction")
        )
    ):
        return False
    if record.get("stage_outputs_format"):
        refs = record.get("stage_predictions", {})
        return set(refs) == {"vote", "limited"} and all(
            matches_file(output, r.get("file"), r.get("sha256"))
            for r in refs.values()
        )
    return True


def matches_file(directory, name, digest):
    from .common import safe_child

    try:
        return bool(digest) and file_hash(safe_child(directory, name)) == digest
    except (OSError, ValueError, TypeError):
        return False


def valid_grid(scratch, case_id):
    path = scratch / (case_id + ".json")
    if not path.exists():
        return False
    try:
        record = read_json(path)
    except (OSError, ValueError):
        return False
    if record.get("stage_outputs_format") != "ct-panel-v2-imagechd-stage-outputs-v1" or not record.get("raw_stages"):
        return False
    valid = record.get("case_id") == case_id and matches_file(scratch, record.get("raw_grid"), record.get("raw_sha256"))
    if valid and record.get("stage_outputs_format"):
        valid = matches_file(scratch, record["raw_stages"], record.get("raw_stages_sha256"))
    return valid


def infer_case(cache, directory, collection, output, row, device="cuda"):
    from .gpu_memory import gpu_budget

    image, _ = load_case(Path(cache), row, labels=False)
    selected, protocol = collection, "single_split"
    fold_indices = []
    if collection.get("split_plan"):
        members, protocol = select_members(collection, row)
        selected = {**collection, "_members": members}
        fold_indices = [m["index"] for m in members]
    elif collection["config"].get("fold"):
        fold_indices = [collection["config"]["fold"]["index"]]
        protocol = "single_fold"
    scratch = Path(output) / ".gpu-results"
    with gpu_budget(device, scratch / (row["case_id"] + ".gpu-memory.json"), collection["config"]) as guard:
        guard.check()
        prediction, details = predict_volume(image, directory, selected, device, return_stages=True)
    arrays = details.pop("stage_outputs")
    stage_file = row["case_id"] + ".stages.npz"
    with atomic_path(scratch / stage_file) as temporary:
        np.savez_compressed(temporary, **arrays)
    raw = row["case_id"] + ".npy"
    with atomic_path(scratch / raw) as temporary:
        np.save(temporary, prediction, allow_pickle=False)
    record = dict(
        case_id=row["case_id"],
        inference_protocol=protocol,
        fold_indices=fold_indices,
        raw_grid=raw,
        raw_sha256=file_hash(scratch / raw),
        stage_outputs_format="ct-panel-v2-imagechd-stage-outputs-v1",
        raw_stages=stage_file,
        raw_stages_sha256=file_hash(scratch / stage_file),
        **details,
    )
    write_json(scratch / (row["case_id"] + ".json"), record)
    return record


def finish_case(output, row):
    from .common import safe_child

    output = Path(output)
    scratch = output / ".gpu-results"
    if not valid_grid(scratch, row["case_id"]):
        raise ValueError("GPU 中间结果校验失败：" + row["case_id"])
    info = read_json(scratch / (row["case_id"] + ".json"))
    prediction = np.load(safe_child(scratch, info["raw_grid"]), mmap_mode="c", allow_pickle=False)
    stages, stage_predictions = {}, {}
    if info.get("stage_outputs_format"):
        with np.load(safe_child(scratch, info["raw_stages"]), allow_pickle=False) as source:
            stages = {k: source[k] for k in ("vote", "limited", "blood")}
        for name in ("vote", "limited"):
            stage_name = row["case_id"] + "_" + name + ".nii.gz"
            with atomic_path(output / stage_name) as temporary:
                nib.save(restore_native(stages[name], row), temporary)
            stage_predictions[name] = {"file": stage_name, "sha256": file_hash(output / stage_name)}
    filename = row["case_id"] + "_seg.nii.gz"
    with atomic_path(output / filename) as temporary:
        nib.save(restore_native(prediction, row), temporary)
    grid = row["case_id"] + "_grid.npz"
    with atomic_path(output / grid) as temporary:
        np.savez_compressed(temporary, prediction=prediction, **stages)
    return {
        k: v
        for k, v in info.items()
        if k not in {"raw_grid", "raw_sha256", "raw_stages", "raw_stages_sha256"}
    } | dict(
        prediction=filename,
        grid_prediction=grid,
        prediction_sha256=file_hash(output / filename),
        grid_prediction_sha256=file_hash(output / grid),
        stage_predictions=stage_predictions,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHD 六阶段独立预测：只读影像缓存与完整模型集合")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--models", required=True, help="包含 models.json 的模型目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--case-id")
    parser.add_argument("--split", choices=["auto", "cv", "train", "val", "test"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--gpu-limit", type=int)
    parser.add_argument("--phase", choices=["all", "gpu", "cpu"], default="all")
    args = parser.parse_args(argv)
    predict(
        args.prepared,
        args.models,
        args.output,
        args.device,
        args.case_id,
        args.split,
        args.allow_smoke,
        args.resume,
        args.gpu_ids.split(",") if args.gpu_ids else None,
        args.gpu_limit,
        args.phase,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
