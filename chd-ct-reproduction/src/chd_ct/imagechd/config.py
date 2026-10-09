"""Six data-supported stages of the original multi-stage method."""

from pathlib import Path

import yaml

from ..models import BiConvLSTM, GridUNet
from ..models.unet import ARCHITECTURE

STAGES = ("crop64", "crop128", "all64", "all128", "blood2d", "blood_lstm")
UNAVAILABLE = {
    "init64/init128": "ImageCHD has no initial-vessel annotation",
    "clinical_diagnosis": "Missing PV/SVC/IVC supervision, verified physical calibration and validated diagnostic rules",
}


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config.get("stages", {})) != set(STAGES):
        raise ValueError("配置必须包含全部六个 CHD 阶段")
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError("旧配置或主干 architecture 不匹配；请使用当前 configs/chd.yaml")
    config["stages"] = {stage: config["stages"][stage] for stage in STAGES}
    if (
        config.get("learning_rate", 0) <= 0
        or config.get("weight_decay", 0) < 0
        or config.get("roi_margin", -1) < 0
    ):
        raise ValueError("学习率、权重衰减、ROI 边距无效")
    if (
        not isinstance(config.get("seed"), int)
        or not isinstance(config.get("threads"), int)
        or config["threads"] < 1
    ):
        raise ValueError("seed/threads 无效")
    for stage, spec in config["stages"].items():
        if stage != "blood_lstm":
            spec.setdefault("spatial_gate", not stage.startswith("blood"))
            if not isinstance(spec["spatial_gate"], bool):
                raise ValueError("spatial_gate 必须是 YAML 布尔值 true/false")
        for key in ("size", "base", "epochs", "batch"):
            if not isinstance(spec.get(key), int) or spec[key] < 1:
                raise ValueError(f"{stage}: {key} 需为正整数")
        if stage == "blood_lstm":
            if not isinstance(spec.get("sequence"), int) or spec["sequence"] < 1 or spec["sequence"] % 2 != 1:
                raise ValueError("序列长度需要正奇数 odd")
            if not isinstance(spec.get("layers"), int) or spec["layers"] < 1:
                raise ValueError("循环层数无效")
            if any(spec[k] != config["stages"]["blood2d"][k] for k in ("size", "base")):
                raise ValueError("blood2d/blood_lstm 的尺寸和通道数必须匹配")
        elif (
            not isinstance(spec.get("levels"), int)
            or spec["levels"] < 2
            or spec["levels"] > (5 if stage == "blood2d" else 4)
        ):
            raise ValueError("网络深度需在 2 到官方层数之间；大于官方深度不是本配置支持的结构")
    return config


def build_model(stage, spec):
    if stage not in STAGES:
        raise ValueError("未知阶段")
    if stage == "blood_lstm":
        return BiConvLSTM(spec["base"], spec["layers"], 3)
    return GridUNet(
        2 if stage == "blood2d" else 3,
        3 if stage == "blood2d" else 8,
        spec["base"],
        spec["levels"],
        spatial_gate=spec.get("spatial_gate", not stage.startswith("blood")),
    )


def weights_for(stage):
    return [1, 2, 16] if stage.startswith("blood") else [1] + [2] * 7
