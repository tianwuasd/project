from pathlib import Path
from functools import wraps

import nibabel as nib
import numpy as np
import torch

from .preprocessing import TARGET, method, preprocessing_identity
from .records import identity, read_json, signature


def guard_inputs(function):
    @wraps(function)
    def guarded(config, *args, **kwargs):
        from .preprocessing_batch import batch_lock
        with batch_lock(Path(config["data_root"]) / "aaoca-classification" / ".batch.lock", shared=True):
            return function(config, *args, **kwargs)
    return guarded


def prepared_receipt(case, config):
    if config.get("prepared_input_policy") == "compatible":
        directory = Path(config.get("prepared_inputs", {}).get(case["case_id"],
                         Path(config["data_root"]) / "aaoca-classification" / case["case_id"])).resolve(strict=True)
        image = nib.load(directory / "normalized.nii.gz")
        training = np.load(directory / "training_xyz.npy", mmap_mode="r", allow_pickle=False)
        direction = image.affine[:3, :3] / np.linalg.norm(image.affine[:3, :3], axis=0)
        if image.shape != TARGET or training.shape != TARGET or training.dtype != np.float16 or not np.allclose(direction, np.eye(3), atol=1e-5, rtol=0):
            raise ValueError("兼容输入须为215×215×85、float16训练数组及正交RAS物理网格")
        outputs = {name: signature(directory / name) for name in ("normalized.nii.gz", "training_xyz.npy")}
        contract = {"target_xyz": list(TARGET), "axes": "XYZ", "network_axes": "BCZXY", "orientation": "RAS", "range": [0, 1]}
        provenance = read_json(directory / "classification.json") if (directory / "classification.json").exists() else None
        return {"identity": identity({"contract": contract, "outputs": outputs}), "stage": "normalized_xyz",
                "geometry": {"target_shape_xyz": list(TARGET), "target_affine": image.affine.tolist(), "target_axis_codes": ["R", "A", "S"]},
                "method": {"recipe": "canonical_ras_v1"}, "directory": str(directory),
                "input_contract": contract, "outputs": outputs, "producer_provenance": provenance}
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


def load_batch(rows, receipts, indices, augmentation=None, training_storage=True, smoke_shape=None):
    images = []
    for index in indices:
        array = load_array(receipts[index], training=training_storage)
        if smoke_shape is not None:
            # 小测保留真实影像内容，中心裁剪只用于明确登记的有限更新检查。
            start = [(n - s) // 2 for n, s in zip(TARGET, smoke_shape)]
            array = array[tuple(slice(a, a + s) for a, s in zip(start, smoke_shape))]
        image = augmentation(array) if augmentation is not None else torch.tensor(array, dtype=torch.float32)
        if tuple(image.shape) != (tuple(smoke_shape) if smoke_shape is not None else TARGET) or not torch.isfinite(image).all():
            raise ValueError("增强结果尺寸或有限值不合法")
        images.append(image.permute(2, 0, 1)[None])
    return torch.stack(images), torch.tensor([rows[i]["target"] for i in indices], dtype=torch.float32)[:, None]
