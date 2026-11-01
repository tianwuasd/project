"""Explicit reruns create a child workflow with fresh outputs and original inputs."""

import copy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..recovery import file_hash
from .cache_selection import new_path
from .launcher import choose, personal_path, show_gpus
from .lineage import load_source, verify_lineage
from .prompts import ask
from .resume import INPUTS
from .resume_checks import check_plan
from .workflow import parse_steps, validate_inputs


def remap_input(value, replacements):
    if not value:
        return value
    path = Path(value).expanduser().resolve()
    for old, new in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        root = Path(old).resolve()
        if path == root or path.is_relative_to(root):
            return str(Path(new) / path.relative_to(root))
    return value


def build_rerun_plan(args, profile, interactive):
    if (
        args.resume_job
        or args.input_job
        or args.task
        or args.tasks
        or args.append_tasks
        or args.preprocess_cache
    ):
        raise ValueError("重跑请使用 --rerun-job 和 --rerun-tasks，不与新建或续跑参数混用")
    if any(getattr(args, key, None) for key in INPUTS) or args.reset_settings:
        raise ValueError("重跑保留原任务输入与方法；修改输入请新建任务")
    from .launcher import make_parser

    defaults = make_parser().parse_args([])
    for key in (
        "size",
        "seed",
        "limit",
        "for_prediction",
        "target_spacing",
        "no_resample",
        "mode",
        "split",
        "case_id",
        "allow_smoke",
        "blank_policy",
        "tree_depth",
        "min_leaf",
        "method",
        "diagnosis_split",
        "resume",
    ):
        if getattr(args, key, None) != getattr(defaults, key, None):
            raise ValueError("重跑保留原方法，修改参数请新建任务：" + key)
    from ..recovery import read_json

    directory = Path(args.rerun_job).expanduser().resolve()
    original = read_json(directory / "plan.json")
    runtime = personal_path(args.runtime or original["runtime"])
    old, state, record = load_source(directory, runtime)
    if state["status"] != "passed":
        raise ValueError("主动重跑适用于已完成任务；未完成任务请使用续跑")
    available = {s["name"] for s in old["steps"]}
    value = args.rerun_tasks
    if value is None and interactive:
        print("可重跑步骤：" + ", ".join(s["name"] for s in old["steps"]))
        print("选择的步骤从头计算，写入新任务；未选择的步骤仅提供原输入，不重新运行。")
        value = ask("重跑步骤名称/全局菜单编号，逗号分隔 [all 全部]：") or "all"
    names = [s["name"] for s in old["steps"]] if value in {None, "all"} else parse_steps(value)
    if not set(names).issubset(available):
        raise ValueError("重跑只能选择原任务已有的步骤；增加步骤请使用追加")
    name = choose(
        args.name, None, (old.get("name", directory.name)[:72] + " / 重跑"), "新任务名称", interactive
    )
    if not name.strip() or len(name) > 80 or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("任务名称需要 1—80 个字符，不能包含控制字符")
    results = personal_path(args.results or directory.parent)
    if runtime == results or runtime in results.parents or results in runtime.parents:
        raise ValueError("环境目录和结果目录必须互相独立")
    new = results / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + "_workflow")
    plan = dict(
        schema_version=1,
        name=name,
        runtime=str(runtime),
        directory=str(new),
        log=str(new / "launcher.log"),
        profile=old.get("profile", profile),
        reset_settings=False,
        steps=[],
        rerun_of=record,
        frozen_files={},
    )
    replacements, future = {}, set()
    prompted_gpu = False
    for previous, prior in zip(old["steps"], state["steps"]):
        if previous["name"] not in names:
            continue
        if prior["status"] not in {"passed", "reused"}:
            raise ValueError("所选步骤尚未完成：" + previous["name"])
        arguments = copy.deepcopy(previous["arguments"])
        arguments.pop("resume", None)
        arguments.pop("inference_phase", None)
        task = arguments["task"]
        # Explicit manifest files within a selected producer must follow its new output too.
        for key in INPUTS:
            if arguments.get(key):
                arguments[key] = remap_input(arguments[key], replacements)
        output = new / "steps" / f"{len(plan['steps']) + 1:02d}_{previous['name']}"
        if task == "preprocess":
            cache = str(new_path(arguments["prepared"]))
            replacements[arguments["prepared"]] = cache
            arguments.update(prepared=cache, preprocess_cache="new", resume=False)
            future.add(cache)
        if "preprocess" in names and task in {"test", "predict", "evaluate", "diagnosis-features"}:
            # New preprocessing must never be silently paired with old model/prediction contracts.
            field = "models" if task in {"test", "predict"} else "predictions"
            if not any(Path(arguments.get(field, "")).is_relative_to(Path(root)) for root in future):
                raise ValueError("预处理重跑需同时选择生成模型/预测的依赖步骤，不能改绑旧模型或预测")
        if args.python:
            arguments["python"] = str(Path(args.python).expanduser().resolve())
        if arguments.get("device") == "cuda":
            if interactive and not prompted_gpu:
                show_gpus()
                args.gpu = choose(args.gpu, arguments.get("gpu"), "auto", "重跑允许的 GPU 编号/UUID", True)
                args.gpu_count = int(
                    choose(args.gpu_count, arguments.get("gpu_count"), 1, "最多并发 GPU 数量", True)
                )
                prompted_gpu = True
            if args.gpu is not None:
                arguments["gpu"] = args.gpu
            if args.gpu_count is not None:
                if not 1 <= args.gpu_count <= 5:
                    raise ValueError("显卡数量需为 1 至 5")
                arguments["gpu_count"] = args.gpu_count
        current = copy.deepcopy(args)
        for key, val in arguments.items():
            setattr(current, key, val)
        validate_inputs(current, future)
        plan["steps"].append(dict(name=previous["name"], output=str(output), arguments=arguments))
        old_root = str(Path(previous["output"]) / task)
        new_root = str(output / task)
        replacements[old_root] = new_root
        future.add(new_root)
        if task == "test":
            replacements[str(Path(old_root) / "predictions")] = str(Path(new_root) / "predictions")
            future.add(str(Path(new_root) / "predictions"))
        for key in ("config", "split_file", "rules", "evidence"):
            if arguments.get(key):
                path = str(Path(arguments[key]).resolve())
                digest = file_hash(path)
                if path in old.get("frozen_files", {}) and old["frozen_files"][path] != digest:
                    raise ValueError("原冻结配置已改变；请新建任务明确选择方法：" + path)
                plan["frozen_files"][path] = digest
    # Check only direct dependencies, using historical snapshots without marking new steps as reused.
    checking = copy.deepcopy(plan)
    checking["steps"] = [{**s, "reuse": p} for s, p in zip(old["steps"], state["steps"])] + checking["steps"]
    plan["input_verification"] = check_plan(checking, "simple")
    record.update(steps=names, frozen_files=plan["frozen_files"])
    verify_lineage(plan)
    print(f"重跑来源：{record['name']}\n独立新任务：{new}\n旧结果保持不变。", flush=True)
    return plan
