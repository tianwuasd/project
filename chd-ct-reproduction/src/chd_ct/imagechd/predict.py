"""Image-only multi-stage inference; prediction never opens cached targets."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .checkpoints import load_stage, read_collection
from .common import LABELS, load_case, read_prepared, write_json
from .config import UNAVAILABLE, build_model
from .fusion import refine_with_blood, weighted_vote
from .geometry import bounding_box, resize
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


def predict(prepared, models, output, device="cpu", case_id=None, split=None, allow_smoke=False):
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
    output.mkdir(parents=True, exist_ok=False)
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
    try:
        torch.set_num_threads(collection["config"]["threads"])
        for row in rows:
            image, _ = load_case(cache, row, labels=False)
            prediction, details = predict_volume(image, directory, collection, device)
            filename = row["case_id"] + "_seg.nii.gz"
            nib.save(restore_native(prediction, row), output / filename)
            grid = row["case_id"] + "_grid.npz"
            np.savez_compressed(output / grid, prediction=prediction)
            report["cases"].append(
                {"case_id": row["case_id"], "prediction": filename, "grid_prediction": grid, **details}
            )
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
    args = parser.parse_args(argv)
    predict(args.prepared, args.models, args.output, args.device, args.case_id, args.split, args.allow_smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
