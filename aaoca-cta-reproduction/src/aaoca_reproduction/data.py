from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .preprocessing import TARGET, method, preprocessing_identity
from .records import read_json, signature


def prepared_receipt(case, config):
    for name in ("image", "label"):
        if signature(case[name]) != case["source_stat"][name]:
            raise ValueError("选中源输入自冻结后发生变化")
    directory = Path(config.get("prepared_inputs", {}).get(case["case_id"],
                     Path(config["data_root"]) / "aaoca-classification" / case["case_id"])).resolve(strict=True)
    receipt = read_json(directory / "classification.json")
    if receipt["stage"] != "normalized_xyz" or receipt["identity"] != preprocessing_identity(case, method(config)):
        raise ValueError("已有预处理输入阶段、源身份或方法不匹配")
    if receipt["geometry"]["target_shape_xyz"] != list(TARGET):
        raise ValueError("预处理分类尺寸不匹配")
    if set(receipt["outputs"]) != {"normalized.nii.gz", "training_xyz.npy"}:
        raise ValueError("预处理输出清单不匹配")
    for name, stamp in receipt["outputs"].items():
        if signature(directory / name) != stamp:
            raise ValueError("预处理文件状态已经变化")
    return {**receipt, "directory": str(directory)}


def load_array(receipt, training):
    directory = Path(receipt["directory"])
    if training:
        array = np.load(directory / "training_xyz.npy", allow_pickle=False)
        if array.dtype != np.float16:
            raise ValueError("作者训练打包路径需要float16存储")
    else:
        array = np.asarray(nib.load(directory / "normalized.nii.gz").dataobj, dtype=np.float32)
    if tuple(array.shape) != TARGET or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("分类真实输入形状、有限值或归一化范围不合法")
    return array


def load_batch(rows, receipts, indices, augmentation=None, training_storage=True):
    images = []
    for index in indices:
        array = load_array(receipts[index], training=training_storage)
        image = augmentation(array) if augmentation is not None else torch.tensor(array, dtype=torch.float32)
        if tuple(image.shape) != TARGET or not torch.isfinite(image).all():
            raise ValueError("增强结果尺寸或有限值不合法")
        images.append(image.permute(2, 0, 1)[None])
    return torch.stack(images), torch.tensor([rows[i]["target"] for i in indices], dtype=torch.float32)[:, None]
