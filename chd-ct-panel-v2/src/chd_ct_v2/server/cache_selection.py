"""Choose cache destinations and persist a bounded, explicit fallback."""

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..recovery import archive_child, read_json, write_json
from .launcher import choose

REUSE_FAILED = 78
FAILURE_FILE = "cache-reuse-failure.json"


def new_path(previous):
    previous = Path(previous)
    stem = re.sub(r"(?:_\d{8}_\d{6}_[0-9a-f]{8})+$", "", previous.name) or "prepared"
    while True:
        path = previous.with_name(
            stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        )
        if not path.exists():
            return path


def select(args, interactive):
    value = choose(
        getattr(args, "preprocess_cache", None),
        None,
        "1",
        "预处理缓存策略：1 尝试沿用（失败自动新建） / 2 新建",
        interactive,
    )
    choices = {"1": "reuse", "reuse": "reuse", "2": "new", "new": "new"}
    if value not in choices:
        raise ValueError("缓存策略请选择 1/reuse 或 2/new")
    args.preprocess_cache = choices[value]
    old = Path(args.prepared)
    if args.preprocess_cache == "new" or (old.exists() and not old.is_dir()):
        args.prepared = str(new_path(old))
        args.resume = False
        print(f"新建预处理缓存：{args.prepared}\n原路径保留：{old}", flush=True)
    elif old.exists():
        args.resume = True
        print(f"尝试沿用预处理缓存：{old}\n将校验来源、参数和缓存；沿用失败自动新建并提示原因。", flush=True)
    else:
        args.resume = False
        print(f"尚无旧缓存，将新建：{old}", flush=True)


def fallback(plan, index, code):
    """Only a typed cache failure permits a retry, never a generic worker error."""
    step = plan["steps"][index]
    report = Path(step["output"]) / FAILURE_FILE
    if step["arguments"]["task"] != "preprocess" or code != REUSE_FAILED or not report.is_file():
        return None
    failure = read_json(report)
    old = step["arguments"]["prepared"]
    if failure.get("kind") != "cache_reuse_failed" or failure.get("prepared") != old:
        return None
    if any(s.get("reuse") or Path(s["output"]).exists() for s in plan["steps"][index + 1 :]):
        raise ValueError("旧缓存不能沿用，但后续步骤已有结果/检查点；请新建任务，不能改绑旧训练数据")
    destination = str(new_path(old))
    event = dict(previous=old, prepared=destination, reason=failure["reason"], action="fallback_new")
    # Persist before running anything at the new path so interruption can resume it.
    for following in plan["steps"][index:]:
        if following["arguments"].get("prepared") == old:
            following["arguments"]["prepared"] = destination
    step["arguments"]["resume"] = False
    step["cache_selection"] = event
    write_json(Path(plan["directory"]) / "plan.json", plan)
    kept = archive_child(Path(step["output"]))
    print(
        f"预处理缓存沿用失败：{failure['reason']}\n自动新建：{destination}\n旧缓存保留：{old}"
        f"\n原尝试日志保留：{kept}\n后续所选步骤已统一切换到新缓存。",
        flush=True,
    )
    return event
