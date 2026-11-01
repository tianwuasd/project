"""Prepared images + checkpoint -> native-grid segmentations; no labels or training."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..data import resize
from ..models import UNet
from .common import FORMAT, LABELS, NORMALIZATION, read_prepared, write_json


def predict(prepared, checkpoint, output, device="cpu", case_id=None, split=None, allow_smoke=False):
    directory, data = read_prepared(prepared)
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if record.get("format") != FORMAT or record.get("labels") != list(LABELS):
        raise ValueError("权重不是 ImageCHD 七结构模型")
    if record.get("normalization") != NORMALIZATION or record["config"]["size"] != data["size"]:
        raise ValueError("权重与预处理尺寸/归一化协议不同")
    if record.get("mode") != "train" and not allow_smoke:
        raise ValueError("这是短测权重；软件试跑需显式 --allow-smoke")
    rows = [
        r
        for r in data["cases"]
        if (case_id is None or r["case_id"] == case_id) and (split is None or r.get("split") == split)
    ]
    if not rows:
        raise ValueError("没有匹配的预测病例")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "task": "imagechd7_segmentation",
        "labels": list(LABELS),
        "checkpoint": str(Path(checkpoint).resolve()),
        "cases": [],
        "geometry": "restored supplied native grid; real millimetre calibration is unverified",
    }
    try:
        config = record["config"]
        torch.set_num_threads(config["threads"])
        model = UNet(3, 8, config["base"], config["levels"]).to(device)
        model.load_state_dict(record["state_dict"], strict=True)
        model.eval()
        ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
        for row in rows:
            # Deliberately do not request a target key; prediction works without any labels.
            with np.load(directory / row["cache"], allow_pickle=False) as cached:
                image = cached["image"]
            if image.shape != (data["size"],) * 3 or not np.isfinite(image).all():
                raise ValueError("预测缓存形状/数值无效")
            if image.min() < 0 or image.max() > 1:
                raise ValueError("预测缓存未按约定归一化")
            with torch.inference_mode():
                tensor = torch.from_numpy(image.copy())[None, None].to(device)
                small = model(tensor).argmax(1)[0].cpu().numpy()
            canonical = resize(small, row["canonical_shape"], labels=True).astype(np.uint8)
            transform = nib.orientations.ornt_transform(ras, np.asarray(row["native_orientation"]))
            native = nib.orientations.apply_orientation(canonical, transform)
            if list(native.shape) != row["native_shape"]:
                raise ValueError("还原原始影像尺寸失败")
            result = nib.Nifti1Image(native, np.asarray(row["native_affine"]))
            result.set_sform(np.asarray(row["native_affine"]), code=1)
            result.set_qform(np.asarray(row["native_affine"]), code=0)
            result.header.set_xyzt_units(*row["units"])
            path = output / (row["case_id"] + "_seg.nii.gz")
            nib.save(result, path)
            report["cases"].append({"case_id": row["case_id"], "prediction": path.name})
            write_json(output / "prediction-report.json", report)
            print(f"已预测：{row['case_id']}", flush=True)
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "prediction-report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅预测：读取预处理缓存与权重，不读取标签、不训练")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--case-id")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args(argv)
    predict(
        args.prepared, args.checkpoint, args.output, args.device, args.case_id, args.split, args.allow_smoke
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
