"""Image-only multi-stage inference; prediction never opens cached targets."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..recovery import atomic_path, output_lock, read_json
from .checkpoints import load_stage, read_collection
from .common import LABELS, load_case, read_prepared, write_json
from .config import UNAVAILABLE, build_model
from .fusion import refine_with_blood, weighted_vote
from .geometry import bounding_box, resize
from .spacing import sample_grid
from .train import file_hash


def anatomy(image, stage, directory, collection, device):
    model, record = load_stage(directory, collection, stage, device)
    size = record["spec"]["size"]
    with torch.inference_mode():
        tensor = torch.from_numpy(resize(image, (size,) * 3))[None, None].to(device)
        return model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)


def blood_prediction(image, directory, collection, device):
    recurrent, record = load_stage(directory, collection, "blood_lstm", device)
    encoder = build_model("blood2d", record["encoder_spec"])
    encoder.load_state_dict(record["encoder_state_dict"])
    encoder.to(device).eval()
    size, sequence = record["spec"]["size"], record["spec"]["sequence"]
    small = np.empty((size, size, image.shape[2]), np.uint8)
    # Cache only one sequence of feature maps, bounding memory on native volumes.
    features = {}
    with torch.inference_mode():
        for z in range(image.shape[2]):
            indices = np.clip(np.arange(z - sequence // 2, z + sequence // 2 + 1), 0, image.shape[2] - 1)
            needed = set(int(i) for i in indices)
            features = {i: value for i, value in features.items() if i in needed}
            for i in needed:
                if i not in features:
                    x = torch.from_numpy(resize(image[:, :, i], (size, size)))[None, None].to(device)
                    features[i] = encoder.features(x)
            window = torch.stack([features[int(i)] for i in indices], dim=1)
            small[:, :, z] = recurrent(window).argmax(1)[0].cpu().numpy()
    return resize(small, image.shape, labels=True).astype(np.uint8)


def predict_volume(image, directory, collection, device):
    crops = [
        resize(anatomy(image, stage, directory, collection, device), image.shape, labels=True).astype(
            np.uint8
        )
        for stage in ("crop64", "crop128")
    ]
    foreground = (crops[0] > 0) | (crops[1] > 0)
    roi = bounding_box(foreground, collection["config"]["roi_margin"])
    cropped = image[roi]
    size = collection["config"]["stages"]["all128"]["size"]
    aligned = (size,) * 3
    votes = [resize(result[roi], aligned, labels=True) for result in crops]
    for stage in ("all64", "all128"):
        votes.append(resize(anatomy(cropped, stage, directory, collection, device), aligned, labels=True))
    fused = resize(weighted_vote(votes), cropped.shape, labels=True).astype(np.uint8)
    blood = blood_prediction(cropped, directory, collection, device)
    result = np.zeros(image.shape, np.uint8)
    result[roi] = refine_with_blood(fused, blood)
    return result, {
        "roi_slices": [[s.start, s.stop] for s in roi],
        "empty_roi_fallback": not bool(foreground.any()),
    }


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
    prepared, models, output, device="cpu", case_id=None, split=None, allow_smoke=False, resume=False
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output):
        return predict_locked(prepared, models, output, device, case_id, split, allow_smoke, resume)


def predict_locked(prepared, models, output, device, case_id, split, allow_smoke, resume):
    cache, data = read_prepared(prepared)
    directory, collection = read_collection(models, allow_smoke)
    rows = [
        r
        for r in data["cases"]
        if (case_id is None or r["case_id"] == case_id) and (split is None or r.get("split") == split)
    ]
    if not rows:
        raise ValueError("没有匹配的预测病例")
    output = Path(output).expanduser().resolve()

    report = {
        "status": "running",
        "task": "chd_multistage_prediction",
        "labels": list(LABELS),
        "prepared_sha256": file_hash(cache / "dataset.json"),
        "models_sha256": file_hash(directory / "models.json"),
        "model_mode": collection["mode"],
        "segmentation_prepared_sha256": collection["prepared_sha256"],
        "cache_resolution": data["resolution"],
        "cases": [],
        "unavailable": UNAVAILABLE,
        "geometry": "supplied original grid restored; millimetre calibration unverified",
    }
    contract = {k: report[k] for k in ("prepared_sha256", "models_sha256", "model_mode")}
    contract.update(
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
    try:
        torch.set_num_threads(collection["config"]["threads"])
        for row in rows:
            record = done.get(row["case_id"])
            if record and all(
                record.get(key + "_sha256")
                and (output / record[key]).is_file()
                and file_hash(output / record[key]) == record[key + "_sha256"]
                for key in ("prediction", "grid_prediction")
            ):
                print("已校验，跳过预测：" + row["case_id"], flush=True)
                continue
            image, _ = load_case(cache, row, labels=False)
            prediction, details = predict_volume(image, directory, collection, device)
            filename = row["case_id"] + "_seg.nii.gz"
            with atomic_path(output / filename) as temporary:
                nib.save(restore_native(prediction, row), temporary)
            grid = row["case_id"] + "_grid.npz"
            with atomic_path(output / grid) as temporary:
                np.savez_compressed(temporary, prediction=prediction)
            done[row["case_id"]] = {
                "case_id": row["case_id"],
                "prediction": filename,
                "grid_prediction": grid,
                "prediction_sha256": file_hash(output / filename),
                "grid_prediction_sha256": file_hash(output / grid),
                **details,
            }
            report["cases"] = [done[r["case_id"]] for r in rows if r["case_id"] in done]
            write_json(output / "prediction-report.json", report)
            print("已预测：" + row["case_id"], flush=True)
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "prediction-report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHD 六阶段独立预测：只读影像缓存与完整模型集合")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--models", required=True, help="包含 models.json 的模型目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--case-id")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
