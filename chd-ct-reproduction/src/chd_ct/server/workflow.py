"""Resolve a multi-select menu into explicit independent steps before detaching."""

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..quickstart.diagnosis import DIAGNOSIS_TASKS, INPUTS, OUTPUT_KEYS
from .launcher import choose, personal_path
from .settings import effective_device, resolve_settings, task_configuration

STEPS = (
    ("environment", "环境检查"),
    ("preprocess", "预处理"),
    ("smoke", "软件短测"),
    ("preflight", "正式尺寸预检"),
    ("train", "正式训练"),
    ("test", "测试集预测和评估"),
    ("predict", "独立预测（默认全病例）"),
    ("evaluate", "独立分割评估"),
    ("diagnosis-labels", "导入诊断标签"),
    ("diagnosis-features", "提取诊断特征"),
    ("diagnosis-train", "训练诊断树"),
    ("diagnose", "诊断与解释"),
    ("diagnosis-evaluate", "诊断评估"),
    ("demo", "合成分割测试"),
    ("diagnosis-demo", "合成诊断测试"),
)
NAMES = [name for name, _ in STEPS]


def parse_steps(value):
    tokens = re.split(r"[\s,，]+", value.strip())
    selected = set()
    for token in tokens:
        if token.isdigit() and int(token) < len(NAMES):
            token = NAMES[int(token)]
        if token not in NAMES:
            raise ValueError("未知步骤：" + token)
        selected.add(token)
    return [name for name in NAMES if name in selected]


def choose_steps(args, interactive):
    if args.task and args.tasks:
        raise ValueError("--task 与 --tasks 只能使用一种")
    if args.tasks:
        return parse_steps(args.tasks)
    if args.task:
        return [args.task]
    if not interactive:
        raise ValueError("非交互运行请指定 --task 或 --tasks")
    print("可多选，输入编号或名称，以逗号/空格分隔；按下列顺序执行，只运行选中的步骤：")
    for i, (_, label) in enumerate(STEPS):
        print(f"  {i:2d}  {label}")
    return parse_steps(input("功能 [0]（例如 0,1,2,4,5）：") or "0")


def validate_inputs(args, future):
    needed = []
    if args.task == "preprocess":
        needed.append(("dataset", args.dataset))
        if Path(args.prepared).exists():
            raise ValueError("预处理目录已存在；取消选择预处理以复用缓存，或指定新的 --prepared")
    elif args.task in {"train", "predict", "test", "evaluate"}:
        needed.append(("prepared", args.prepared))
    if args.task == "train":
        needed.append(("config", args.config))
    if args.task in {"test", "predict"}:
        needed.append(("models", args.models))
    if args.task == "test" and args.case_id:
        raise ValueError("test 评估完整 split；单病例请使用 predict")
    if args.task == "evaluate":
        needed.append(("predictions", args.predictions))
    if args.task in DIAGNOSIS_TASKS:
        needed += [
            (key, getattr(args, key, None))
            for key, _, _ in INPUTS[args.task]
            if key != "classifier" or args.method != "rules"
        ]
    for key, value in needed:
        if not value:
            raise ValueError(f"{args.task} 需要 --{key.replace('_', '-')}")
        path = Path(value).expanduser().resolve()
        if str(path) in future:
            continue
        if not path.exists():
            raise ValueError(f"{args.task} 输入不存在：{key}={path}")
        if key == "prepared" and not (path / "dataset.json").is_file():
            raise ValueError("预处理尚未完成：缺少 dataset.json")
    if args.task == "diagnosis-evaluate" and args.diagnosis_split == "all":
        raise ValueError("诊断评估须选择 train/val/test 中一个集合")


def build_plan(args, profile, interactive):
    names = choose_steps(args, interactive)
    if "diagnosis-features" in names and "diagnosis-train" in names:
        if ("test" in names and "predict" not in names) or (
            "predict" in names and (args.split or args.case_id)
        ):
            raise ValueError("诊断训练需要全病例预测：同时选择 predict，并去掉 --split/--case-id")
    runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", interactive))
    settings_file = runtime / ".server-settings.json"
    saved = (
        json.loads(settings_file.read_text(encoding="utf-8"))
        if settings_file.is_file() and not args.reset_settings
        else {}
    )
    if not isinstance(saved, dict):
        raise ValueError("服务器设置文件必须是 JSON 对象")
    results = personal_path(
        choose(args.results, saved.get("results"), profile["results"], "结果目录", interactive)
    )
    if runtime == results or runtime in results.parents or results in runtime.parents:
        raise ValueError("环境目录和结果目录必须互相独立")
    directory = results / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + "_workflow")
    base = copy.deepcopy(args)
    base.runtime, base.results = str(runtime), str(results)
    steps, future = [], set()
    prompted_gpu = False
    for i, name in enumerate(names):
        current = copy.deepcopy(base)
        current.task = "train" if name in {"smoke", "preflight"} else name
        if name in {"smoke", "preflight"}:
            current.mode = name
        elif name == "train" and not args.task:
            current.mode = args.mode or "train"
        # Resource prompts are shared once. Single-card steps still preserve training allocation.
        if prompted_gpu:
            current.gpu, current.gpu_count = base.gpu, base.gpu_count
        links = []
        if current.task in {"test", "predict"}:
            links.append(("models", "last_models"))
        if current.task == "evaluate":
            links.append(("predictions", "last_predictions"))
        if current.task in DIAGNOSIS_TASKS:
            links += [(key, source) for key, _, source in INPUTS[current.task]]
        for field, source in links:
            if getattr(current, field, None) is None and saved.get(source) in future:
                setattr(current, field, saved[source])
        _, _, _, saved = resolve_settings(current, profile, interactive, saved_override=saved)
        # Freeze caller-relative paths before the worker changes its working directory.
        for field in ("python", "split_file", "evidence", "rules"):
            if field in task_configuration(current) and getattr(current, field, None):
                setattr(current, field, str(Path(getattr(current, field)).expanduser().resolve()))
        validate_inputs(current, future)
        output = directory / "steps" / f"{i + 1:02d}_{name}"
        steps.append({"name": name, "output": str(output), "arguments": task_configuration(current)})
        for field in ("dataset", "prepared", "config"):
            if field in steps[-1]["arguments"]:
                setattr(base, field, getattr(current, field))
        if effective_device(current) == "cuda":
            base.gpu, base.gpu_count = current.gpu, current.gpu_count
            prompted_gpu = True
        produced = None
        if name == "preprocess":
            produced = current.prepared
        elif current.task == "train" and current.mode in {"smoke", "preflight", "train"}:
            produced = str(output / "train")
            saved["last_models"] = produced
        elif name in {"test", "predict"}:
            produced = str(output / ("test/predictions" if name == "test" else "predict"))
            saved["last_predictions"] = produced
        elif name in OUTPUT_KEYS:
            produced = str(output / name)
            saved[OUTPUT_KEYS[name]] = produced
        if produced:
            future.add(str(Path(produced).resolve()))
    for field in ("dataset", "prepared"):
        value = getattr(base, field, None)
        if value:
            path = Path(value).expanduser().resolve()
            for work in (runtime, results):
                if path == work or path in work.parents or work in path.parents:
                    raise ValueError(f"{field} 必须独立于环境和结果目录")
    return {
        "schema_version": 1,
        "reset_settings": args.reset_settings,
        "runtime": str(runtime),
        "directory": str(directory),
        "log": str(directory / "launcher.log"),
        "profile": profile,
        "steps": steps,
    }
