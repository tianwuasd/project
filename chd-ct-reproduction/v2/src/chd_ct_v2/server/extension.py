"""Append missing workflow steps without rebinding existing work to another experiment."""

import copy
from pathlib import Path

from ..quickstart.diagnosis import OUTPUT_KEYS
from ..recovery import file_hash
from .prompts import ask
from .workflow import STEPS, frozen_files, parse_steps, resolve_steps

INPUT_SETTINGS = {
    "dataset": "dataset",
    "prepared": "prepared",
    "config": "config",
    "models": "last_models",
    "predictions": "last_predictions",
    "diagnosis_source": "diagnosis_source",
    "diagnosis_labels": "last_diagnosis_labels",
    "features": "last_features",
    "classifier": "last_classifier",
    "diagnoses": "last_diagnoses",
    "split_file": "split_file",
    "spacing_excel": "spacing_excel",
    "evidence": "evidence",
    "rules": "rules",
}
LINK_INPUTS = {
    "models",
    "predictions",
    "diagnosis_source",
    "diagnosis_labels",
    "features",
    "classifier",
    "diagnoses",
}


def source_stamp(directory):
    return {
        name: file_hash(directory / name) if (directory / name).is_file() else None
        for name in ("plan.json", "job_state.json", "receipt.json")
    }


def verify_source(plan):
    expected = plan.get("resume_source")
    if expected is not None and source_stamp(Path(plan["directory"])) != expected:
        raise ValueError("原任务记录已改变，请重新选择续跑/追加，不能覆盖另一次提交")


def choose_append(args, plan, complete, interactive):
    present = {step["name"] for step in plan["steps"]}
    value = getattr(args, "append_tasks", None)
    if value is None and interactive and complete:
        print("已有步骤保留；成功步骤锁定复用，失败/中断步骤继续恢复。可追加：")
        for index, (name, label) in enumerate(STEPS):
            if name not in present:
                print(f"  {index:2d}  {label} ({name})")
        value = ask("追加步骤编号/名称，逗号分隔；b 返回：")
    if not value:
        if complete:
            available = ", ".join(name for name, _ in STEPS if name not in present)
            raise ValueError("原选步骤已完成；可用 --append-tasks 追加未执行步骤：" + available)
        return []
    names = parse_steps(value)
    repeated = present.intersection(names)
    if repeated:
        raise ValueError("原计划已有步骤不能再次追加，请直接续跑：" + ", ".join(sorted(repeated)))
    if "preprocess" in names and any(s["arguments"].get("prepared") for s in plan["steps"]):
        raise ValueError("原任务已经绑定预处理数据，不能追加预处理改变其来源；请新建任务")
    return names


def append_steps(args, plan, names, interactive):
    if not names:
        return plan
    base = copy.deepcopy(args)
    base.task = None
    base.tasks = ",".join(names)
    base.mode = None
    base.runtime = plan["runtime"]
    base.results = str(Path(plan["directory"]).parent)
    saved = {"results": base.results}
    future = set()
    # Use only this workflow's frozen inputs/artifacts, never another job's latest settings.
    for step in plan["steps"]:
        old = step["arguments"]
        for key in ("dataset", "prepared", "config", "python"):
            if old.get(key):
                setattr(base, key, old[key])
                saved[key] = old[key]
        for key, setting in INPUT_SETTINGS.items():
            if old.get(key):
                saved[setting] = old[key]
        if old.get("device") == "cuda":
            for key in ("gpu", "gpu_count"):
                if old.get(key) is not None:
                    saved[key] = old[key]
        output = Path(step["output"])
        name = step["name"]
        produced = None
        if name == "preprocess":
            produced = old["prepared"]
        elif name in {"train", "smoke", "preflight"}:
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
    for key, setting in INPUT_SETTINGS.items():
        bound = saved.get(setting)
        requested = getattr(args, key, None)
        if not bound:
            continue
        if requested and Path(requested).expanduser().resolve() != Path(bound).expanduser().resolve():
            raise ValueError(f"原任务已绑定 {key}，追加不能改换其来源；请新建任务")
        # Link fields are filled per new step, so newly produced predecessors still take priority.
        setattr(base, key, None if key in LINK_INPUTS else bound)
    if args.python:
        base.python = args.python
    if "diagnosis-features" in names and "diagnosis-train" in names:
        if ("test" in names and "predict" not in names) or (
            "predict" in names and (args.split or args.case_id)
        ):
            raise ValueError("诊断训练需要全病例预测：选择 predict，并去掉 --split/--case-id")
    new, _ = resolve_steps(
        base,
        plan.get("profile", {}),
        interactive,
        Path(plan["directory"]),
        saved,
        names,
        offset=len(plan["steps"]),
        future=future,
        freeze_links=True,
    )
    for step in new:
        if Path(step["output"]).exists():
            raise ValueError("追加步骤目录已存在，不能覆盖：" + step["output"])
    plan["steps"].extend(new)
    for path, digest in frozen_files(new).items():
        plan.setdefault("frozen_files", {}).setdefault(path, digest)
    plan["appended_steps"] = names
    return plan
