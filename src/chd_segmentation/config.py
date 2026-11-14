from copy import deepcopy
import json
import math
from pathlib import Path

import yaml


DEFAULTS = {
    "model": "stunet", "scope": "formal", "fold": 0, "result_root": None, "plan": None,
    "split": {"path": None, "format": "auto", "id_field": "case_id", "unit": "auto"},
    "logging": {"heartbeat_seconds": 15, "case_details": True},
    "planning": {"mode": "auto", "samples_per_case": 100000, "max_samples": 2000000,
                 "seed": 1234, "target_spacing_xyz": None, "clip_range": None,
                 "mean": None, "std": None},
    "training": {"patch": [128, 128, 128], "batch": 4, "attempts": 1000,
                 "epochs": None, "steps_per_epoch": 250, "validation_steps": 50, "poly_power": 0.9,
                 "learning_rate": 0.001, "momentum": 0.99, "weight_decay": 3e-5,
                 "seed": 20261113, "precision": "fp32", "foreground_fraction": 0.33,
                 "save_every_attempts": 250},
    "prediction": {"overlap": 0.5, "window_batch": 1, "blend": "gaussian", "export_space": "source"},
    "resources": {"device": "cpu", "cpu_threads": 2, "cache_bytes": 1024 ** 3,
                  "reserve_ram_bytes": 4 * 1024 ** 3, "cpu_fraction": 0.8, "ram_fraction": 0.8,
                  "workers": "auto", "max_workers": 32, "prefetch": 2, "readers": "auto", "max_readers": 8,
                  "gpu_count": 1, "gpu_ids": None, "batch_layout": None, "cpu_processes": 1,
                  "wait_seconds": None, "poll_seconds": 2, "share": True,
                  "share_window_seconds": 30, "share_utilization": 50,
                  "vram_reserve_bytes": 4 * 1024 ** 3, "vram_reserve_fraction": 0.1,
                  "peak_margin": 1.2},
    "technical": {"roi_xyz": None, "allow_role_overlap": False},
}


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError("未知配置字段：" + key)
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path, overrides=None):
    path = Path(path).resolve(strict=True)
    content = path.read_text(encoding="utf-8")
    source = json.loads(content) if path.suffix.lower() == ".json" else yaml.safe_load(content)
    required = {"dataset", "manifest", "pretrained", "work_root", "run_dir"}
    if not isinstance(source, dict) or not required <= source.keys():
        raise ValueError("配置缺少数据集、清单、权重或工作路径")
    config = _merge({**DEFAULTS, **{key: None for key in required}}, source)
    if overrides:
        config = _merge(config, overrides)
    if config["model"] != "stunet" or config["dataset"] not in ("chd1282", "imagechd"):
        raise ValueError("当前实现支持 chd1282/imagechd 的 STU-Net")
    if config["scope"] not in ("formal", "technical"):
        raise ValueError("scope 必须是 formal 或 technical")
    logging = config["logging"]
    if type(logging["heartbeat_seconds"]) not in (int, float) or not math.isfinite(logging["heartbeat_seconds"]) or logging["heartbeat_seconds"] <= 0:
        raise ValueError("日志心跳间隔必须为有限正数，单位为秒")
    if type(logging["case_details"]) is not bool:
        raise ValueError("logging.case_details 必须为布尔值")
    if config["scope"] == "formal" and (config["technical"]["roi_xyz"] or config["technical"]["allow_role_overlap"]):
        raise ValueError("正式任务不能使用技术子区域或角色重叠")
    for key in required - {"dataset"}:
        target = Path(config[key]).expanduser()
        config[key] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
    if type(config["fold"]) is not int or config["fold"] < 0:
        raise ValueError("fold 必须为非负整数；每折清单分别声明 train/validation 角色")
    if config["split"]["path"] is not None:
        target = Path(config["split"]["path"]).expanduser()
        config["split"]["path"] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
    if config["split"]["unit"] not in ("auto", "case", "patient"):
        raise ValueError("split.unit 必须为 auto、case 或 patient")
    if config["dataset"] == "chd1282" and config["split"]["unit"] == "case":
        raise ValueError("1282 正式划分必须按患者分组")
    planning = config["planning"]
    if planning["mode"] not in ("auto", "existing"):
        raise ValueError("planning.mode 必须是 auto 或 existing")
    if any(type(planning[key]) is not int or planning[key] < 1 for key in ("samples_per_case", "max_samples")):
        raise ValueError("计划采样数量必须为正整数")
    if type(planning["seed"]) is not int or planning["seed"] < 0:
        raise ValueError("计划随机种子必须为非负整数")
    for key, size in (("target_spacing_xyz", 3), ("clip_range", 2)):
        values = planning[key]
        if values is not None and (not isinstance(values, list) or len(values) != size or
                                  any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("计划覆盖值无效：" + key)
    if planning["target_spacing_xyz"] is not None and min(planning["target_spacing_xyz"]) <= 0:
        raise ValueError("目标间距必须为正数，顺序为 XYZ")
    if planning["clip_range"] is not None and planning["clip_range"][0] >= planning["clip_range"][1]:
        raise ValueError("灰度截断范围必须递增")
    for key in ("mean", "std"):
        if planning[key] is not None and (type(planning[key]) not in (int, float) or not math.isfinite(planning[key])):
            raise ValueError("归一化覆盖值无效：" + key)
    if planning["std"] is not None and planning["std"] <= 0:
        raise ValueError("归一化标准差必须为正数")
    if config["plan"] is None:
        if planning["mode"] == "existing":
            raise ValueError("existing 模式必须提供已有计划路径")
        config["plan"] = str(Path(config["work_root"]) / "plans" / config["dataset"] /
                             f"fold_{config['fold']}" / config["scope"] / "preprocessing_plan.json")
    else:
        target = Path(config["plan"]).expanduser()
        config["plan"] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
    if planning["mode"] == "existing" and any(planning[key] is not None for key in
                                             ("target_spacing_xyz", "clip_range", "mean", "std")):
        raise ValueError("existing 模式使用计划原值，手动覆盖需使用 auto 模式和新输出路径")
    if config["result_root"] is not None:
        if not isinstance(config["result_root"], str) or not config["result_root"].strip():
            raise ValueError("result_root 必须为非空路径或 null")
        target = Path(config["result_root"]).expanduser()
        config["result_root"] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
    t, p, r = config["training"], config["prediction"], config["resources"]
    if len(t["patch"]) != 3 or any(type(x) is not int or x < 8 for x in t["patch"]):
        raise ValueError("patch 必须为三个不小于 8 的整数")
    if any(type(t[key]) is not int or t[key] < 1 for key in ("batch", "attempts", "save_every_attempts")):
        raise ValueError("批量、尝试次数及保存间隔必须为正整数")
    if t["epochs"] is not None and (type(t["epochs"]) is not int or t["epochs"] < 1):
        raise ValueError("epochs 必须为正整数或 null")
    if any(type(t[k]) is not int or t[k] < 1 for k in ("steps_per_epoch", "validation_steps")) or t["poly_power"] <= 0:
        raise ValueError("每轮步数和学习率指数必须为正数")
    if t["precision"] not in ("fp32", "fp16", "bf16") or not 0 <= t["foreground_fraction"] <= 1:
        raise ValueError("精度或前景比例无效")
    if any(type(t[key]) not in (int, float) for key in ("learning_rate", "momentum", "weight_decay")):
        raise ValueError("优化器参数必须是数值；YAML 中指数形式应写为 3.0e-5")
    if t["learning_rate"] <= 0 or not 0 < t["momentum"] < 1 or t["weight_decay"] < 0:
        raise ValueError("学习率、动量或权重衰减无效")
    if not 0 <= p["overlap"] < 1 or p["window_batch"] < 1 or p["blend"] not in ("constant", "gaussian"):
        raise ValueError("预测窗口参数无效")
    if p["export_space"] not in ("source", "working"):
        raise ValueError("导出空间必须为 source 或 working")
    if r["cpu_threads"] < 1 or r["cache_bytes"] < 0 or r["reserve_ram_bytes"] < 0:
        raise ValueError("资源预算无效")
    if r["device"] not in ("cpu", "cuda"):
        raise ValueError("device 必须为 cpu 或 cuda；具体显卡使用 gpu_ids 选择")
    if not 1 <= r["gpu_count"] <= 4 or not 1 <= r["cpu_processes"] <= 4:
        raise ValueError("每组进程数必须为 1–4")
    if r["gpu_ids"] is not None and (not isinstance(r["gpu_ids"], list) or
            len(set(map(str, r["gpu_ids"]))) != len(r["gpu_ids"]) or len(r["gpu_ids"]) != r["gpu_count"]):
        raise ValueError("gpu_ids 必须为不重复的物理编号或 UUID 列表，数量等于 gpu_count")
    world = r["gpu_count"] if r["device"] == "cuda" else r["cpu_processes"]
    shares = r["batch_layout"]
    if t["batch"] < world or (shares is not None and
            (len(shares) != world or any(type(v) is not int or v < 1 for v in shares) or sum(shares) != t["batch"])):
        raise ValueError("每个进程至少分配一个样本；batch_layout 的数量及合计必须匹配")
    if not 0 < r["cpu_fraction"] <= 1 or not 0 < r["ram_fraction"] <= 1:
        raise ValueError("CPU 和内存比例必须位于 (0, 1]")
    if r["workers"] != "auto" and (type(r["workers"]) is not int or r["workers"] < 0):
        raise ValueError("workers 必须为 auto 或非负整数，表示全组上限")
    if any(type(r[k]) is not int or r[k] < 1 for k in ("max_workers", "prefetch", "max_readers")) or \
            (r["readers"] != "auto" and (type(r["readers"]) is not int or r["readers"] < 1)):
        raise ValueError("worker 上限、预取与读盘并发必须为正整数")
    if r["wait_seconds"] is not None and (type(r["wait_seconds"]) not in (int, float) or
            not math.isfinite(r["wait_seconds"]) or r["wait_seconds"] < 0):
        raise ValueError("wait_seconds 必须为非负秒数或 null（持续等待）")
    if any(type(r[key]) not in (int, float) or not math.isfinite(r[key]) for key in ("poll_seconds", "share_window_seconds")) or \
            r["poll_seconds"] <= 0 or r["share_window_seconds"] < 0:
        raise ValueError("资源等待时间无效")
    if not 0 <= r["share_utilization"] <= 100 or not 0 <= r["vram_reserve_fraction"] < 1 or r["peak_margin"] < 1:
        raise ValueError("GPU 共享阈值或显存预算无效")
    return config
