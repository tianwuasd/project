"""Verify a complete, coherent six-stage collection before any prediction."""

import json
from pathlib import Path

import torch

from .common import LABELS, MODEL_FORMAT, NORMALIZATION, safe_child
from .config import STAGES, build_model
from .train import file_hash


def read_collection(directory, allow_smoke=False):
    directory = Path(directory).expanduser().resolve()
    if directory.is_file():
        if directory.name != "models.json":
            raise ValueError("需要完整模型目录或 models.json；旧 best.pt 位于 back 流程")
        directory = directory.parent
    record = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    if (
        record.get("status") != "complete"
        or record.get("format") != MODEL_FORMAT
        or set(record.get("stages", {})) != set(STAGES)
    ):
        raise ValueError("模型集合未完成或缺少阶段")
    if record.get("labels") != list(LABELS) or record.get("normalization") != NORMALIZATION:
        raise ValueError("模型标签/归一化协议不同")
    if record.get("mode") != "train" and not allow_smoke:
        raise ValueError("短测 smoke/preflight 权重需显式 --allow-smoke")
    for stage in STAGES:
        info = record["stages"][stage]
        path = safe_child(directory, info["file"])
        if file_hash(path) != info["sha256"]:
            raise ValueError("权重校验失败，模型集合混用或被修改")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            any(
                checkpoint.get(key) != record.get(key)
                for key in ("format", "labels", "normalization", "config", "mode", "prepared_sha256")
            )
            or checkpoint.get("stage") != stage
            or checkpoint.get("spec") != record["config"]["stages"][stage]
        ):
            raise ValueError("阶段权重来自不同配置/数据划分")
        if stage == "blood_lstm" and (
            checkpoint.get("encoder_sha256") != record["stages"]["blood2d"]["sha256"]
            or checkpoint.get("encoder_spec") != record["config"]["stages"]["blood2d"]
        ):
            raise ValueError("循环模型的冻结编码器不匹配")
    return directory, record


def load_stage(directory, collection, stage, device):
    checkpoint = torch.load(
        safe_child(directory, collection["stages"][stage]["file"]), map_location="cpu", weights_only=True
    )
    model = build_model(stage, checkpoint["spec"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
