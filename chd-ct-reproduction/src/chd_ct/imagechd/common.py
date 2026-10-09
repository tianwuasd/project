"""Shared model specification and explicit per-image intensity transform."""

import json
from pathlib import Path

import numpy as np
import yaml

FORMAT = "imagechd7-v1"
LABELS = ("BG", "LV", "RV", "LA", "RA", "MYO", "AO", "PA")
IGNORE = 255
NORMALIZATION = {"kind": "per_image_percentile", "low": 1, "high": 99}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("size", "base", "levels", "epochs", "batch", "threads"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"配置需要正整数 {key}")
    if config["levels"] < 2 or config["size"] < 2 ** config["levels"]:
        raise ValueError("尺寸过小，不能支持该 U-Net 层数")
    if not isinstance(config.get("seed"), int) or config.get("learning_rate", 0) <= 0:
        raise ValueError("需要整数 seed 和正 learning_rate")
    return config


def normalize(image):
    if image.ndim != 3 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("影像必须为有限值三维体积")
    low, high = np.percentile(image, [NORMALIZATION["low"], NORMALIZATION["high"]])
    if not high > low:
        raise ValueError("影像强度范围退化，无法归一化")
    return ((np.clip(image, low, high) - low) / (high - low)).astype(np.float32), {
        "low": float(low),
        "high": float(high),
    }


def read_prepared(directory):
    directory = Path(directory).expanduser().resolve()
    data = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if data.get("status") != "prepared" or not isinstance(data.get("size"), int) or data["size"] < 8:
        raise ValueError("预处理尚未完成或尺寸无效")
    if data.get("format") != FORMAT or data.get("labels") != list(LABELS):
        raise ValueError("不是 ImageCHD 七结构预处理结果，标签协议不同")
    if data.get("normalization") != NORMALIZATION or not data.get("cases"):
        raise ValueError("预处理结果不完整或归一化协议不同")
    seen = set()
    for row in data["cases"]:
        case = row["case_id"]
        if not isinstance(case, str) or not case or case in {".", ".."} or any(c in case for c in "/\\:"):
            raise ValueError("病例编号不能包含路径字符")
        path = (directory / row["cache"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("预处理缓存路径无效或文件缺失")
        if row["case_id"] in seen:
            raise ValueError("预处理清单含重复 case_id")
        seen.add(row["case_id"])
    return directory, data
