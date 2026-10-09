"""Shared ImageCHD cache protocol; no dependence on the archived paper package."""

import json
from pathlib import Path

import numpy as np

FORMAT = "imagechd7-v2"
MODEL_FORMAT = "imagechd7-multistage-v1"
LABELS = ("BG", "LV", "RV", "LA", "RA", "MYO", "AO", "PA")
IGNORE = 255
NORMALIZATION = {"kind": "per_image_percentile", "low": 1, "high": 99}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalize(image):
    if image.ndim != 3 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("影像必须为有限值三维体积")
    low, high = np.percentile(image, [1, 99])
    if not high > low:
        raise ValueError("影像强度范围退化")
    return ((np.clip(image, low, high) - low) / (high - low)).astype(np.float32), {
        "low": float(low),
        "high": float(high),
    }


def safe_child(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("缓存或权重路径无效")
    return path


def read_prepared(directory):
    directory = Path(directory).expanduser().resolve()
    data = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if data.get("status") != "prepared" or data.get("format") not in {FORMAT, "imagechd7-v1"}:
        raise ValueError("不是完整的 ImageCHD 预处理结果")
    if (
        data.get("labels") != list(LABELS)
        or data.get("normalization") != NORMALIZATION
        or data.get("ignore_index") != IGNORE
    ):
        raise ValueError("标签或归一化协议不同")
    if not data.get("cases"):
        raise ValueError("缓存为空")
    if data["format"] == "imagechd7-v1":
        if not isinstance(data.get("size"), int) or data["size"] < 8:
            raise ValueError("旧缓存尺寸无效")
        data["resolution"] = "resized"
    elif data.get("resolution") not in {"native", "resized"}:
        raise ValueError("缺少缓存分辨率记录")
    seen = set()
    for row in data["cases"]:
        case = row.get("case_id")
        if (
            not isinstance(case, str)
            or not case
            or case in {".", ".."}
            or any(c in case for c in "/\\:")
            or case in seen
        ):
            raise ValueError("病例编号无效或重复")
        seen.add(case)
        safe_child(directory, row["cache"])
        if "cache_shape" not in row:
            row["cache_shape"] = [data["size"]] * 3
        for key in ("cache_shape", "native_shape", "canonical_shape"):
            shape = row[key]
            if len(shape) != 3 or any(not isinstance(x, int) or x < 2 for x in shape):
                raise ValueError("缓存几何尺寸无效")
        affine = np.asarray(row["native_affine"], dtype=float)
        if (
            affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) < 1e-8
        ):
            raise ValueError("原始 affine 无效")
    return directory, data


def load_case(directory, row, labels=True):
    with np.load(safe_child(directory, row["cache"]), allow_pickle=False) as record:
        image = record["image"].astype(np.float32)
        target = record["target"].copy() if labels else None
    if (
        image.shape != tuple(row["cache_shape"])
        or not np.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError("缓存影像尺寸或归一化数值无效")
    if labels and (target.shape != image.shape or not np.isin(target, [*range(8), IGNORE]).all()):
        raise ValueError("缓存标签形状/编号无效")
    return image, target
