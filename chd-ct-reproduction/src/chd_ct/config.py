from pathlib import Path

import yaml

from .labels import STAGES
from .models import BiConvLSTM, UNet


def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if set(config.get("stages", {})) != set(STAGES):
        raise ValueError("Configuration must declare all eight train stages")
    for stage, spec in config["stages"].items():
        if any(spec.get(k, 0) <= 0 for k in ("size", "base", "epochs", "batch")):
            raise ValueError(f"Positive size/base/epochs/batch required: {stage}")
        if stage == "blood_lstm":
            if spec.get("sequence", 0) < 1 or spec["sequence"] % 2 != 1:
                raise ValueError("Sequence length must be positive and odd")
            if (
                spec["base"] != config["stages"]["blood2d"]["base"]
                or spec["size"] != config["stages"]["blood2d"]["size"]
            ):
                raise ValueError("blood2d and blood_lstm feature size/base must match")
        elif spec["levels"] < 2 or spec["size"] < 2 ** spec["levels"]:
            raise ValueError("Input too small for the requested U-Net depth")
    return config


def classes_for(stage):
    return 3 if stage.startswith("blood") else (7 if stage.startswith("init") else 11)


def weights_for(stage):
    return (
        [1, 2, 16]
        if stage.startswith("blood")
        else ([1, 2, 2, 2, 2, 2, 3] if stage.startswith("init") else [1] + [2] * 10)
    )


def build_model(stage, spec):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "blood_lstm":
        return BiConvLSTM(spec["base"], spec["layers"], 3)
    return UNet(
        2 if stage == "blood2d" else 3,
        classes_for(stage),
        spec["base"],
        spec["levels"],
        spatial_gate=not stage.startswith("blood"),
    )
