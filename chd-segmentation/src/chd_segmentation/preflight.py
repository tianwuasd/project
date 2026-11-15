import os
from inspect import signature
import math
from importlib.metadata import version
from pathlib import Path
import shutil

import torch

from .artifacts.store import file_digest, read_json, digest
from .datasets.manifest import load_manifest
from .datasets.splits import apply_split
from .models.stunet.adapter import WEIGHT_SHA256
from .preprocessing.offline_store import validate_storage, recipe, case_key, case_directory, inspect_case
from .preprocessing.prepare import load_prepared, validate_prepared_config


def _directory_for(path):
    candidate = Path(path).expanduser()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(str(candidate))
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate.resolve(strict=True)


def _path_check(path, writable=False, require_file=False):
    target = Path(path).expanduser()
    if require_file:
        target = target.resolve(strict=True)
        if not target.is_file() or not os.access(target, os.R_OK):
            raise PermissionError(f"文件不存在、不是普通文件或不可读取：{target}")
    directory = _directory_for(path)
    mode = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    if not os.access(directory, mode):
        raise PermissionError(f"路径不可{'读写' if writable else '读取'}：{directory}")
    usage = shutil.disk_usage(directory)
    return {"path": str(target.resolve()), "checked_directory": str(directory),
            "readable": True, "writable": bool(writable), "free_bytes": usage.free}


def environment_check(resources):
    checks = {}
    checks["dependencies"] = {name: version(name) for name in ("torch", "monai", "nnunetv2", "numpy", "scipy", "nibabel", "psutil")}
    from scipy.ndimage import map_coordinates, spline_filter1d
    from batchgenerators.augmentations.utils import _resample_seg_by_argmax
    from numpy.lib.format import open_memmap
    functions = ((map_coordinates, {"order", "mode", "prefilter"}),
                 (spline_filter1d, {"output", "axis", "mode"}),
                 (_resample_seg_by_argmax, {"labels", "score_fn", "nearest_fn", "seg_tiebreak"}),
                 (open_memmap, {"mode", "dtype", "shape"}))
    for function, required in functions:
        if not required <= signature(function).parameters.keys():
            raise RuntimeError("当前环境的必要预处理 API 不兼容：" + function.__name__)
    checks["apis"] = [function.__name__ for function, _ in functions]
    if resources["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，当前环境无法使用 CUDA")
        checks["cuda"] = {"available": True, "device_count": torch.cuda.device_count()}
    else:
        checks["cuda"] = {"available": torch.cuda.is_available(), "required": False}
    return checks


def preprocessing_disk_check(config, record, content=None):
    from .training.budget import case_estimates
    estimates = case_estimates(config, record, 'preprocess')
    identity = digest(recipe(record, config) if content is None else content)
    base = Path(config['preprocessing']['root'])
    index = read_json(base/'index.json') if (base/'index.json').is_file() else {}
    root = (base/index[identity]).resolve() if identity in index else None
    needed = []
    for row, estimate in zip(record['manifest']['cases'], estimates):
        key = case_key(row, identity)
        if root is None or inspect_case(case_directory(root, key), key, identity, full=False) is None:
            needed.append(estimate)
    final = sum(value['output_array_bytes'] for value in needed)
    intermediate = max((value['output_array_bytes']+32*math.prod(value['input_shape_xyz']) for value in needed), default=0)
    required = 0 if not needed else final+intermediate*min(len(needed), config['resources']['max_readers'])+config['preprocessing']['block_bytes']*8
    path = _path_check(base, writable=True)
    result = dict(pending_cases=len(needed), estimated_remaining_bytes=required,
                  free_bytes=path['free_bytes'], basis='未复用病例输出上界、分块中间文件和实际读盘并发')
    if path['free_bytes'] < required:
        raise OSError('data目标磁盘剩余空间不足，保留已有数据并释放其他空间后重试')
    return result


def quick_check(config, stage, prepared=None):
    """执行不读取影像、不初始化模型的配置与输入检查。"""
    if stage not in ("prepare", "preprocess", "train"):
        raise ValueError("stage 必须为 prepare、preprocess 或 train")
    if stage == "train" and prepared is None:
        raise ValueError("train 快速检查必须提供 prepared 描述")
    report = {"ok": False, "stage": stage, "checks": {}, "limits": [
        "未读取 CT 或标签体素", "离线病例仅检查元数据和文件 stamp，未重算数组摘要"]}
    checks = report["checks"]
    checks["configuration"] = {"dataset": config["dataset"], "scope": config["scope"],
                                "fold": config["fold"], "statistics_source": config["preprocessing"]["statistics_source"]}
    checks.update(environment_check(config['resources']))

    manifest = apply_split(load_manifest(config["manifest"], config["dataset"], config["scope"],
                                         allow_unassigned=config["split"]["path"] is not None), config)
    roles = {role: sum(role in row["roles"] for row in manifest["cases"])
             for role in ("train", "validation", "test", "predict")}
    if stage in ("preprocess", "train"):
        if not roles["train"] or not roles["validation"]:
            raise ValueError("训练准备要求非空 train 与 validation 分组")
        if config["scope"] == "formal" and manifest.get("split_source", {}).get("unit", "patient") == "patient" and \
                any(not row.get("patient") for row in manifest["cases"]):
            raise ValueError("正式分组缺少患者身份")
        if any(row.get("label") is None for row in manifest["cases"] if set(row["roles"]) & {"train", "validation"}):
            raise ValueError("训练或验证病例缺少标签")
    checks["groups"] = {"case_count": len(manifest["cases"]), "roles": roles,
                        "split_source": manifest.get("split_source"),
                        "statistics_source": config["preprocessing"]["statistics_source"]}
    checks["source_paths"] = {row["case_id"]: {
        kind: _path_check(row[kind], require_file=True) for kind in ("image", "label") if row.get(kind) is not None
    } for row in manifest["cases"]}

    weight = Path(config["pretrained"]).resolve(strict=True)
    if not weight.is_file() or file_digest(weight) != WEIGHT_SHA256:
        raise ValueError("预训练权重缺失或 SHA256 与模型要求不符")
    checks["pretrained"] = {"path": str(weight), "sha256": WEIGHT_SHA256}
    checks["paths"] = {"manifest": _path_check(config["manifest"], require_file=True), "weight": _path_check(weight, require_file=True),
                       "work_root": _path_check(config["work_root"], writable=True),
                       "run_dir": _path_check(config["run_dir"], writable=True)}
    if config["preprocessing"]["root"]:
        checks["paths"]["preprocessing_root"] = _path_check(config["preprocessing"]["root"], writable=True)
    checks["capacity"] = {key: value["free_bytes"] for key, value in checks["paths"].items()}

    if prepared is not None:
        record = load_prepared(prepared) if isinstance(prepared, (str, Path)) else prepared
        validate_prepared_config(record, config)
        storage_root = validate_storage(record, full=False, config=config) if stage == "train" or record.get("storage") is not None else None
        checks["prepared"] = {"identity": record["identity"], "path": str(prepared) if isinstance(prepared, (str, Path)) else None,
                              "offline_root": None if storage_root is None else str(storage_root),
                              "offline_validation": "metadata_and_file_stamps" if storage_root is not None else "not_applicable"}
        if stage == "preprocess" and config["preprocessing"]["root"]:
            checks['disk_plan'] = preprocessing_disk_check(config, record)
    report["ok"] = True
    return report
