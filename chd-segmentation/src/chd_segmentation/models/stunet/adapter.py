from dataclasses import asdict

import numpy as np
import torch
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

from ...artifacts.store import file_digest
from ...contracts import ModelSpec, StageSpec
from ...losses.partial import deep_supervision_loss
from .network import STUNet


WEIGHT_SHA256 = "f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6"


class StunetAdapter:
    @staticmethod
    def spec(config, plan):
        patch = config["training"]["patch"]
        spacing = plan["configurations"]["3d_fullres"]["spacing"]
        maximum_pools = 5 if config["dataset"] == "chd1282" else 999
        _, pools, _, adjusted, divisors = get_pool_and_conv_props(spacing, patch, 4, maximum_pools)
        if list(map(int, adjusted)) != patch or np.prod(np.asarray(patch) // divisors) <= 1:
            raise ValueError("训练块不满足 STU-Net 的步长或归一化要求")
        strides = [tuple(map(int, value)) for value in pools[1:]][:5]
        strides += [(1, 1, 1)] * (5 - len(strides))
        return ModelSpec(11 if config["dataset"] == "chd1282" else 8, tuple(patch), tuple(strides))

    @staticmethod
    def build(spec):
        if isinstance(spec, dict):
            spec = ModelSpec(**spec)
        return STUNet(spec.num_classes, spec.strides, spec.kernels)

    @staticmethod
    def stages(config):
        settings = config["training"]
        attempts = settings["epochs"] * settings["steps_per_epoch"] if settings["epochs"] else settings["attempts"]
        return (StageSpec("segmentation", attempts, settings["batch"], config["dataset"]),)

    @staticmethod
    def load_pretrained(model, path):
        if file_digest(path) != WEIGHT_SHA256:
            raise ValueError("官方 STU-Net-S 权重摘要不匹配")
        allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                   type(np.dtype(np.float32)), type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(allowed):
            source = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        own = model.state_dict()
        names = {key for key in own if not key.startswith("seg_outputs.")}
        if names != {key for key in source if not key.startswith("seg_outputs.")}:
            raise ValueError("预训练骨干层名不匹配")
        missing, unexpected = model.load_state_dict({key: source[key] for key in names}, strict=False)
        if unexpected or set(missing) != set(own) - names:
            raise ValueError("预训练骨干加载不完整")
        if any(not torch.equal(model.state_dict()[key], source[key]) for key in names):
            raise ValueError("预训练骨干数值不一致")
        return {"sha256": WEIGHT_SHA256, "loaded_tensors": len(names), "reset_head_tensors": len(missing)}

    @staticmethod
    def training_step(model, batch, dataset):
        outputs = model(batch["image"])
        return deep_supervision_loss(outputs, batch["label"], batch["unknown_classes"], dataset)
