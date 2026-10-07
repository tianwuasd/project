import json
from pathlib import Path

import numpy as np
import torch

from ..config import build_model
from ..data import bounding_box, load_volume, normalize_ct, resize, save_prediction
from ..training.engine import load_checkpoint, recurrent_features
from .fusion import refine_with_blood, weighted_vote


def predict_stage(image, stage, directories, device):
    probabilities, reference_record = None, None
    for directory in directories:
        model, record = load_checkpoint(Path(directory) / f"{stage}.pt", device, stage)
        if reference_record and (
            record["spec"] != reference_record["spec"]
            or record["config"]["window"] != reference_record["config"]["window"]
        ):
            raise ValueError("Incompatible fold model configurations")
        reference_record = record
        size = record["spec"]["size"]
        x = torch.from_numpy(resize(image, (size,) * 3))[None, None].to(device)
        with torch.inference_mode():
            p = model(x).softmax(1)[0].cpu().numpy()
        probabilities = p if probabilities is None else probabilities + p
        del model
    return probabilities.argmax(0).astype(np.uint8), reference_record


def predict_blood(image, directories, device):
    total, reference_spec = None, None
    for directory in directories:
        recurrent, record = load_checkpoint(Path(directory) / "blood_lstm.pt", device, "blood_lstm")
        spec = record["spec"]
        if reference_spec is not None and spec != reference_spec:
            raise ValueError("Blood fold configurations differ")
        reference_spec = spec
        encoder = build_model("blood2d", record["encoder_spec"])
        encoder.load_state_dict(record["encoder_state_dict"])
        encoder.to(device).eval()
        size, sequence = spec["size"], spec["sequence"]
        depth = image.shape[2]
        # Keep input slices on CPU; feature windows stay bounded by sequence length.
        slices = [torch.from_numpy(resize(image[:, :, z], (size, size)))[None] for z in range(depth)]
        probabilities = np.empty((3, size, size, depth), dtype=np.float32)
        with torch.inference_mode():
            for z in range(depth):
                indices = np.clip(np.arange(z - sequence // 2, z + sequence // 2 + 1), 0, depth - 1)
                x = torch.stack([slices[i] for i in indices])[None].to(device)
                probabilities[:, :, :, z] = (
                    recurrent(recurrent_features(encoder, x)).softmax(1)[0].cpu().numpy()
                )
        total = probabilities if total is None else total + probabilities
        del recurrent, encoder
    return total.argmax(0).astype(np.uint8)


def run_inference(image_path, model_directories, output, device="cpu", allow_smoke=False):
    if not model_directories:
        raise ValueError("At least one complete model directory required")
    records = {}
    # Validate metadata for ALL checkpoints before computation, including full config equality within each fold.
    for directory in model_directories:
        fold_manifest = None
        for stage in ("crop64", "crop128", "all64", "all128", "init64", "init128", "blood_lstm"):
            path = Path(directory) / f"{stage}.pt"
            if not path.is_file():
                raise ValueError(f"Missing model: {path}")
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record.get("stage") != stage or record.get("format_version") != 1:
                raise ValueError("Checkpoint stage mismatch")
            if records and record["config"] != next(iter(records.values()))["config"]:
                raise ValueError("All model configurations must match")
            if record["spec"] != record["config"]["stages"][stage]:
                raise ValueError("Checkpoint specification differs from its declared configuration")
            if fold_manifest is not None and record["manifest_sha256"] != fold_manifest:
                raise ValueError("Model stages within one fold have different manifest hashes")
            fold_manifest = record["manifest_sha256"]
            records[f"{directory}:{stage}"] = {
                k: v for k, v in record.items() if k not in ("state_dict", "encoder_state_dict")
            }
            del record
    config = next(iter(records.values()))["config"]
    if config["profile"] == "smoke" and not allow_smoke:
        raise ValueError("Smoke weights are synthetic test artifacts; pass --allow-smoke only for a demo")
    torch.set_num_threads(config.get("threads", 2))
    volume = load_volume(image_path)
    image = normalize_ct(volume.data, config["window"])
    crop64, _ = predict_stage(image, "crop64", model_directories, device)
    crop128, _ = predict_stage(image, "crop128", model_directories, device)
    crop64 = resize(crop64, image.shape, labels=True).astype(np.uint8)
    crop128 = resize(crop128, image.shape, labels=True).astype(np.uint8)
    # Two-voter ties are unspecified in the paper: foreground union avoids losing the ROI.
    roi = bounding_box((crop64 > 0) | (crop128 > 0), config["roi_margin"])
    cropped = image[roi]
    size = config["stages"]["all128"]["size"]
    aligned_shape = (size,) * 3
    predictions = [
        resize(crop128[roi], aligned_shape, labels=True),
        resize(crop64[roi], aligned_shape, labels=True),
    ]
    for stage in ("all128", "init128", "all64", "init64"):
        result, _ = predict_stage(cropped, stage, model_directories, device)
        predictions.append(resize(result, aligned_shape, labels=True))
    fused = weighted_vote(predictions, ["crop", "crop", "all", "init", "all", "init"])
    # Refine at native ROI resolution to retain the 2D network's high-resolution boundaries.
    fused = resize(fused, cropped.shape, labels=True).astype(np.uint8)
    blood = resize(predict_blood(cropped, model_directories, device), cropped.shape, labels=True).astype(
        np.uint8
    )
    refined = refine_with_blood(fused, blood)
    final = np.zeros(image.shape, dtype=np.uint8)
    final[roi] = refined
    initial = np.zeros(image.shape, dtype=np.uint8)
    initial[roi] = resize(predictions[3], cropped.shape, labels=True).astype(np.uint8)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_prediction(final, volume, output / "segmentation.nii.gz")
    save_prediction(initial, volume, output / "initial_segmentation.nii.gz")
    metadata = {
        "profile": config["profile"],
        "clinical_validation": False,
        "fold_count": len(model_directories),
        "roi_slices": [[s.start, s.stop] for s in roi],
        "roi_empty_fallback": not bool(((crop64 > 0) | (crop128 > 0)).any()),
        "weights": [
            {
                "stage": r["stage"],
                "epoch": r["epoch"],
                "manifest_sha256": r["manifest_sha256"],
                "max_steps": r.get("max_steps"),
            }
            for r in records.values()
        ],
    }
    (output / "inference.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return final, initial, volume, metadata
