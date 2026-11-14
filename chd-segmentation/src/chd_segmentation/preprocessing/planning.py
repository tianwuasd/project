from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
import psutil

from ..artifacts.store import digest, operation_lock, read_json, require_output, write_json
from ..datasets.manifest import LABELS
from .transforms import CaseTransform


def plan_request(config, manifest):
    rows = sorted((row for row in manifest["cases"] if "train" in row["roles"]), key=lambda row: row["case_id"])
    if not rows or any(row.get("label") is None for row in rows):
        raise ValueError("自动计划需要有真实标签的训练病例")
    if config["scope"] == "technical" and len(manifest["cases"]) > 2:
        raise ValueError("技术计划只允许一至两个真实病例")
    return {"protocol": "train-foreground-ct-v1", "dataset": config["dataset"], "fold": config["fold"],
            "scope": config["scope"], "technical": config["technical"], "settings": config["planning"],
            "training_cases": rows, "labels": LABELS[config["dataset"]],
            "versions": {name: version(name) for name in ("nnunetv2", "numpy", "nibabel")}}


def validate_auto_plan(plan, config, manifest):
    generation = plan.get("generation", {})
    if generation.get("request_identity") != digest(plan_request(config, manifest)):
        raise ValueError("已有自动计划与训练病例、折数、范围或设置不一致，请使用新的计划输出路径")
    if generation.get("content_identity") != digest({key: value for key, value in plan.items() if key != "generation"}):
        raise ValueError("自动计划内容摘要不一致，拒绝复用")


def foreground_sample(data, segmentation, count, rng):
    image, label = data.ravel(), segmentation.ravel()
    block = 1000000
    counts = [int(np.count_nonzero(label[start:start + block] > 0))
              for start in range(0, label.size, block)]
    total = sum(counts)
    if not total:
        raise ValueError("训练病例的所选区域没有已标注的有效前景")
    # 分块定位随机前景序号，避免为整卷前景创建大型坐标数组。
    ranks = np.sort(rng.choice(total, count, replace=True))
    result = np.empty(count, dtype=np.float32)
    seen = 0
    for index, amount in enumerate(counts):
        left, right = np.searchsorted(ranks, [seen, seen + amount])
        if right > left:
            start = index * block
            locations = np.flatnonzero(label[start:start + block] > 0)
            result[left:right] = image[start + locations[ranks[left:right] - seen]]
        seen += amount
    return result


def fit_plan(config, manifest, request):
    started = perf_counter()
    settings, rows = config["planning"], request["training_cases"]
    count = min(settings["samples_per_case"], settings["max_samples"] // len(rows))
    if count < 1:
        raise ValueError("max_samples 小于训练病例数，无法保证每例参与拟合")
    reserve = config["resources"]["reserve_ram_bytes"]
    if psutil.virtual_memory().available < reserve + len(rows) * count * 16:
        raise MemoryError("计划灰度采样超出 RAM 安全余量")
    prepared = {"technical": config["technical"],
                "dataset_json": {"labels": {name: i for i, name in enumerate(LABELS[config["dataset"]])}}}
    reader = CaseTransform(prepared, reserve)
    samples, spacings, shapes, timings = [], [], [], []
    for row in rows:
        mark = perf_counter()
        data, label, image, spacing, roi, read_seconds = reader.read_case(row)
        data, label, _ = crop_to_nonzero(data[None], label[None])
        samples.append(foreground_sample(data[0], label[0], count,
                                        np.random.default_rng(settings["seed"])))
        shapes.append(list(data.shape[1:][::-1]))
        spacings.append(spacing[::-1].tolist())
        reader.check_sources(row)
        timings.append({"case_id": row["case_id"], "read_seconds": read_seconds,
                        "seconds": perf_counter() - mark})
        del data, label, image
    values = np.concatenate(samples)
    low, median, high = np.percentile(values, [0.5, 50, 99.5])
    stats = {"mean": float(values.mean(dtype=np.float64)), "std": float(values.std(dtype=np.float64)),
             "min": float(values.min()), "max": float(values.max()), "median": float(median),
             "percentile_00_5": float(low), "percentile_99_5": float(high)}
    if settings["clip_range"] is not None:
        stats["percentile_00_5"], stats["percentile_99_5"] = settings["clip_range"]
    for key in ("mean", "std"):
        if settings[key] is not None:
            stats[key] = settings[key]
    if not np.isfinite(list(stats.values())).all() or stats["std"] <= 0:
        raise ValueError("前景灰度统计非有限或标准差为零")
    # 仅调用官方间距规则；完整网络规划不属于本项目的预处理计划。
    planner = ExperimentPlanner.__new__(ExperimentPlanner)
    planner.overwrite_target_spacing = (None if settings["target_spacing_xyz"] is None else
                                        settings["target_spacing_xyz"][::-1])
    planner.dataset_fingerprint = {"spacings": spacings, "shapes_after_crop": shapes}
    planner.anisotropy_threshold = 3
    spacing = planner.determine_fullres_target_spacing()
    forward = [int(np.argmax(spacing))]
    forward += [axis for axis in range(3) if axis not in forward]
    plan = {"plans_name": "CHDTrainOnlyCT", "dataset_name": config["dataset"],
            "scope": config["scope"], "fold": config["fold"], "image_reader_writer": "NibabelIO",
            "label_manager": "LabelManager", "transpose_forward": forward,
            "transpose_backward": np.argsort(forward).tolist(), "fit_ids": [row["case_id"] for row in rows],
            "foreground_intensity_properties_per_channel": {"0": stats},
            "configurations": {"3d_fullres": {
                "data_identifier": "chd_train_ct_v1", "preprocessor_name": "DefaultPreprocessor", "architecture": {},
                "spacing": spacing[forward].tolist(), "normalization_schemes": ["CTNormalization"],
                "use_mask_for_norm": [False], "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
                "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None}}}}
    plan["generation"] = {"request": request, "request_identity": digest(request), "content_identity": digest(plan),
                          "samples_per_case_applied": count, "total_samples": len(values),
                          "case_timings": timings, "seconds": perf_counter() - started}
    return plan


def resolve_plan(config, manifest):
    if config["planning"]["mode"] == "existing":
        return read_json(config["plan"])
    request = plan_request(config, manifest)
    path = require_output(config["plan"], config["work_root"],
                          [manifest["data_root"], config["manifest"], config["pretrained"], Path(__file__).parents[3]])
    with operation_lock(path.parent):
        if path.exists():
            plan = read_json(path)
            validate_auto_plan(plan, config, manifest)
            reader = CaseTransform({}, config["resources"]["reserve_ram_bytes"])
            for row in request["training_cases"]:
                reader.check_sources(row)
            return plan
        plan = fit_plan(config, manifest, request)
        write_json(path, plan)
        return plan
