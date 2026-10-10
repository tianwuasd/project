"""Task-specific server preferences; CPU steps never configure GPU resources."""

import json
from pathlib import Path

from ..quickstart.diagnosis import DIAGNOSIS_TASKS, INPUTS, resolve_inputs
from .launcher import ROOT, choose, personal_path

DEFAULT_DATASET = "/data_nas/zhangruichen/baidu_import/ImageCHD_dataset"
CPU_TASKS = {"preprocess", "evaluate", *DIAGNOSIS_TASKS}


def effective_device(args):
    if args.task in CPU_TASKS or (args.task == "train" and args.mode == "check"):
        return "cpu"
    return "cuda" if args.device == "auto" else args.device


def task_configuration(args):
    """Only serialize inputs used by this step, not unrelated saved preferences."""
    fields = {"task", "python"}
    by_task = {
        "environment": set(),
        "preprocess": {"dataset", "prepared", "size", "seed", "limit", "split_file", "for_prediction"},
        "train": {"prepared", "config", "mode"},
        "predict": {"prepared", "models", "case_id", "split", "allow_smoke"},
        "test": {"prepared", "models", "split", "allow_smoke"},
        "evaluate": {"prepared", "predictions", "split"},
        "demo": set(),
        "diagnosis-labels": {"blank_policy"},
        "diagnosis-features": set(),
        "diagnosis-train": {"tree_depth", "min_leaf", "seed", "allow_smoke"},
        "diagnose": {"method", "diagnosis_split", "evidence", "rules", "allow_smoke"},
        "diagnosis-evaluate": {"diagnosis_split"},
        "diagnosis-demo": set(),
    }
    fields.update(by_task[args.task])
    if args.task in DIAGNOSIS_TASKS:
        fields.update(
            name for name, _, _ in INPUTS[args.task] if name != "classifier" or args.method != "rules"
        )
    else:
        fields.add("device")
    if effective_device(args) == "cuda":
        fields.add("gpu")
        if args.task in {"train", "environment"}:
            fields.add("gpu_count")
    result = {name: getattr(args, name) for name in sorted(fields) if getattr(args, name, None) is not None}
    if "device" in result:
        result["device"] = effective_device(args)
    return result


def resolve_settings(args, profile, interactive, saved_override=None):
    runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", interactive))
    settings_file = runtime / ".server-settings.json"
    saved = (
        dict(saved_override)
        if saved_override is not None
        else json.loads(settings_file.read_text(encoding="utf-8"))
        if settings_file.exists() and not getattr(args, "reset_settings", False)
        else {}
    )
    if not isinstance(saved, dict):
        raise ValueError("服务器设置文件必须是 JSON 对象")
    results = personal_path(
        choose(args.results, saved.get("results"), profile["results"], "结果目录", interactive)
    )
    args.runtime, args.results = str(runtime), str(results)
    saved["results"] = str(results)
    if args.task in DIAGNOSIS_TASKS:
        defaults = {
            "prepared": profile["prepared"],
            "diagnosis_source": str(
                Path(args.dataset or saved.get("dataset") or profile.get("dataset", DEFAULT_DATASET))
                / "imageCHD_dataset_info.xlsx"
            ),
        }
        inputs = dict(saved)
        if args.dataset and not args.diagnosis_source:
            inputs.pop("diagnosis_source", None)
        resolve_inputs(args, inputs, choose, interactive, defaults)
        args.gpu, args.gpu_count = "auto", 1
        if args.task == "diagnosis-labels":
            saved["diagnosis_source"] = args.diagnosis_source
        return runtime, results, settings_file, saved

    if args.task == "preprocess":
        args.dataset = str(
            Path(
                choose(
                    args.dataset,
                    saved.get("dataset"),
                    profile.get("dataset", DEFAULT_DATASET),
                    "原始 ImageCHD 数据目录",
                    interactive,
                )
            )
            .expanduser()
            .resolve()
        )
        saved["dataset"] = args.dataset
    if args.task in {"preprocess", "train", "predict", "test", "evaluate"}:
        args.prepared = str(
            personal_path(
                choose(
                    args.prepared, saved.get("prepared"), profile["prepared"], "预处理缓存目录", interactive
                )
            )
        )
        saved["prepared"] = args.prepared
    if args.task == "train":
        default = Path(profile.get("config", "configs/chd.yaml"))
        if not default.is_absolute():
            default = ROOT / default
        args.config = str(
            Path(choose(args.config, saved.get("config"), str(default), "模型配置 YAML", interactive))
            .expanduser()
            .resolve()
        )
        args.mode = choose(
            args.mode, None, "smoke", "训练模式：check / smoke / preflight / train", interactive
        )
        if args.mode not in {"check", "smoke", "preflight", "train"}:
            raise ValueError("无效训练模式")
        saved["config"] = args.config
    if args.task in {"predict", "test"}:
        args.models = choose(args.models, saved.get("last_models"), "", "完整六阶段模型目录", interactive)
        if args.models:
            args.models = str(Path(args.models).expanduser().resolve())
    if args.task == "evaluate":
        args.predictions = choose(
            args.predictions, saved.get("last_predictions"), "", "预测结果目录", interactive
        )
        if args.predictions:
            args.predictions = str(Path(args.predictions).expanduser().resolve())
    if effective_device(args) != "cuda":
        args.gpu, args.gpu_count = "auto", 1
        return runtime, results, settings_file, saved
    count = choose(
        getattr(args, "gpu_count", None),
        saved.get("gpu_count"),
        profile.get("gpu_count", 1),
        "训练使用几张 GPU（1—5）",
        interactive and args.task in {"train", "environment"},
    )
    try:
        args.gpu_count = int(count)
    except (TypeError, ValueError) as error:
        raise ValueError("显卡数量需为 1 至 5 的整数") from error
    if isinstance(count, bool) or str(count).strip() != str(args.gpu_count) or not 1 <= args.gpu_count <= 5:
        raise ValueError("显卡数量需为 1 至 5 的整数")
    args.gpu = choose(
        getattr(args, "gpu", None),
        saved.get("gpu"),
        profile.get("gpu", "auto"),
        "GPU 编号/UUID，逗号分隔，或 auto",
        interactive,
    )
    saved["gpu"] = args.gpu
    if args.task in {"train", "environment"}:
        saved["gpu_count"] = args.gpu_count
    return runtime, results, settings_file, saved
