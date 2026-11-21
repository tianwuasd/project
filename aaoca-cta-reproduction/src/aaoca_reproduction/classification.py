from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import SENet154

from .records import RELEASE, signature


LABEL_MAPS = {"aaoca_present": {"0": "normal_origin", "1": "AAOCA"},
              "origin_left": {"0": "anomalous_right_coronary", "1": "anomalous_left_coronary_system"},
              "high_risk": {"0": "paper_low_risk_course", "1": "interarterial_or_subpulmonic"}}
LEARNING_RATES = {"paper_lr_1e-3": 0.001, "release_default_lr_1e-4": 0.0001}


def model_contract(task):
    if task not in LABEL_MAPS:
        raise ValueError("未知分类任务")
    return {"architecture": "MONAI_SENet154_3D_1channel_1logit", "dropout": 0.3,
            "layers": [3, 8, 36, 3], "groups": 64, "reduction": 16,
            "task": task, "labels": LABEL_MAPS[task], "author_release": RELEASE}


def model_metadata(task):
    return {**model_contract(task), "monai_requested": "1.4.0", "monai_actual": version("monai"),
            "compatibility_basis": "source-audit-SENet154-and-19-symbols-20261121",
            "forward_numerically_verified": False}


def validate_model_metadata(stored, task):
    if any(stored.get(key) != value for key, value in model_contract(task).items()):
        raise ValueError("本地检查点网络、算法或任务身份不匹配")


def build_model(task):
    model_metadata(task)
    model = SENet154(spatial_dims=3, in_channels=1, num_classes=1, dropout_prob=0.3,
                     layers=(3, 8, 36, 3), groups=64, reduction=16, pretrained=False,
                     dropout_dim=1, inplanes=128, downsample_kernel_size=3, input_3x3=True)
    if [len(getattr(model, f"layer{i}")) for i in range(1, 5)] != [3, 8, 36, 3]:
        raise ValueError("实际SENet154阶段结构不符合作者配方")
    if model.layer0.conv1.in_channels != 1 or model.last_linear.in_features != 2048 or model.last_linear.out_features != 1:
        raise ValueError("实际网络输入/分类参数形状不符合作者配方")
    if model.layer1[0].conv2.conv.groups != 64:
        raise ValueError("实际SEBottleneck分组参数不符合作者配方")
    return model.to(dtype=torch.float32)


def validate_descriptor(descriptor, task):
    if descriptor["task"] != task or descriptor["labels"] != LABEL_MAPS[task]:
        raise ValueError("权重任务或类别映射不匹配")
    if descriptor["author_release"] != RELEASE or not descriptor["source"].strip():
        raise ValueError("权重需要固定作者版本及真实来源")
    path = Path(descriptor["path"]).resolve(strict=True)
    return path, signature(path)


def load_weights(model, descriptor, task):
    path, stamp = validate_descriptor(descriptor, task)
    value = torch.load(path, map_location="cpu", weights_only=True)
    provenance = {"descriptor": descriptor, "file_stat": stamp}
    if descriptor["format"] == "local_complete_checkpoint":
        validate_model_metadata(value["model_meta"], task)
        if int(descriptor["member_index"]) != value["state"]["member_index"]:
            raise ValueError("本地权重实际初始化成员与声明不符")
        if "training_patient_keys" in descriptor and set(descriptor["training_patient_keys"]) != set(value["state"]["training_patient_keys"]):
            raise ValueError("检测初始化权重的实际患者来源与声明不一致")
        provenance.update(member_index=value["state"]["member_index"], seed=value["state"]["seed"],
                          cohort_identity=value["state"]["cohort_identity"])
        weights = value["model"]
    elif descriptor["format"] == "author_state_dict":
        weights = value
        if descriptor.get("legacy_fc_keys", False):
            weights = dict(weights)
            weights["last_linear.weight"] = weights.pop("loss_func_layer.fc.weight")
            weights["last_linear.bias"] = weights.pop("loss_func_layer.fc.bias")
    else:
        raise ValueError("权重格式需显式指定")
    model.load_state_dict(weights, strict=True)
    return provenance


def require_direction_review(config, receipts):
    review = config.get("direction_review")
    if not review or review.get("approved") is not True or not review.get("reviewer") or not review.get("reference"):
        raise ValueError("正式分类缺少经确认的作者参照方向或获准独立适配配方")
    if review["recipe"] != config["classification_recipe"]:
        raise ValueError("方向审核与实际预处理配方不匹配")
    reference = np.asarray(review["reference_direction_matrix_RAS"], dtype=float)
    if reference.shape != (3, 3) or not np.isfinite(reference).all() or not np.allclose(
            reference.T @ reference, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("参照方向需为明确的正交方向矩阵")
    for receipt in receipts:
        affine = np.asarray(receipt["geometry"]["target_affine"])
        directions = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        if not np.allclose(directions, reference, atol=1e-5, rtol=0):
            raise ValueError("本例实际分类方向与已审核参照不一致")
    return review
