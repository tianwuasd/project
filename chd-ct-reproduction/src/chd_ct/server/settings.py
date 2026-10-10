"""Shared server preferences, independent of datasets and model implementations."""

import json
from pathlib import Path

from ..quickstart.diagnosis import DIAGNOSIS_TASKS, resolve_inputs
from .launcher import ROOT, choose, personal_path

DEFAULT_DATASET = "/data_nas/zhangruichen/baidu_import/ImageCHD_dataset"


def resolve_settings(args, profile, interactive):
    runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", interactive))
    settings_file = runtime / ".server-settings.json"
    saved = (
        json.loads(settings_file.read_text(encoding="utf-8"))
        if settings_file.exists() and not getattr(args, "reset_settings", False)
        else {}
    )
    if not isinstance(saved, dict):
        raise ValueError("服务器设置文件必须是 JSON 对象")
    results = personal_path(
        choose(args.results, saved.get("results"), profile["results"], "结果目录", interactive)
    )
    if args.task in DIAGNOSIS_TASKS:
        defaults = {
            "prepared": profile["prepared"],
            "diagnosis_source": str(
                Path(args.dataset or saved.get("dataset") or profile.get("dataset", DEFAULT_DATASET))
                / "imageCHD_dataset_info.xlsx"
            ),
        }
        # An explicit dataset overrides the saved source-table directory.
        input_settings = dict(saved)
        if args.dataset and not args.diagnosis_source:
            input_settings.pop("diagnosis_source", None)
        resolve_inputs(args, input_settings, choose, interactive, defaults)
        args.gpu, args.gpu_count = "auto", 1
        saved["results"] = str(results)
        if args.task == "diagnosis-labels":
            saved["diagnosis_source"] = args.diagnosis_source
        return runtime, results, settings_file, saved
    for name, label, default in (
        ("dataset", "原始 ImageCHD 数据目录", profile.get("dataset", DEFAULT_DATASET)),
        ("prepared", "预处理缓存目录", profile["prepared"]),
    ):
        ask = interactive and (name == "prepared" or args.task == "preprocess")
        value = choose(getattr(args, name), saved.get(name), default, label, ask)
        resolved = personal_path(value) if name == "prepared" else Path(value).expanduser().resolve()
        setattr(args, name, str(resolved))
    default_config = profile.get("config", "configs/chd.yaml")
    if not Path(default_config).is_absolute():
        default_config = str(ROOT / default_config)
    args.config = str(
        Path(
            choose(
                args.config,
                saved.get("config"),
                default_config,
                "模型配置 YAML",
                interactive and args.task == "train",
            )
        )
        .expanduser()
        .resolve()
    )
    count = choose(
        getattr(args, "gpu_count", None),
        saved.get("gpu_count"),
        profile.get("gpu_count", 1),
        "训练使用几张 GPU（1—5）",
        interactive and args.task in {"train", "environment"} and args.device != "cpu",
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
        interactive and args.device != "cpu" and args.task not in {"preprocess", "evaluate"},
    )
    if args.task == "train":
        args.mode = choose(
            args.mode, None, "smoke", "训练模式：check / smoke / preflight / train", interactive
        )
        if args.mode not in {"check", "smoke", "preflight", "train"}:
            raise ValueError("无效训练模式")
    if args.task in {"predict", "test"}:
        args.models = choose(args.models, saved.get("last_models"), "", "完整六阶段模型目录", interactive)
    if args.task == "evaluate":
        args.predictions = choose(
            args.predictions, saved.get("last_predictions"), "", "预测结果目录", interactive
        )
    saved.update(
        dataset=args.dataset,
        prepared=args.prepared,
        config=args.config,
        results=str(results),
        gpu_count=args.gpu_count,
        gpu=args.gpu,
    )
    return runtime, results, settings_file, saved
