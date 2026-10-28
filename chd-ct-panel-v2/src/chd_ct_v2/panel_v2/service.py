"""Explicit stage graph over the preserved CT 0.24 scientific pipeline."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

from ..recovery import file_hash, read_json, write_json
from ..server.launcher import ROOT, make_parser
from . import catalog, snapshot
from .identity import PANEL, check, claim, locations, path

STAGES = {
    "environment": ("environment",),
    "prepare": ("preprocess",),
    "train": ("train",),
    "predict": ("test",),
    "predict-all": ("predict",),
    "diagnosis": (
        "diagnosis-labels",
        "diagnosis-features",
        "diagnosis-train",
        "diagnose",
        "diagnosis-evaluate",
    ),
    "assessment": ("preprocess", "smoke", "test"),
    "synthetic": ("demo", "diagnosis-demo"),
}
SOURCE = {"train": {"prepare"}, "predict": {"train"}, "predict-all": {"train"}, "diagnosis": {"predict-all"}}
RAW_OPTIONS = {
    "dataset",
    "config",
    "size",
    "seed",
    "limit",
    "split_file",
    "spacing_excel",
    "target_spacing",
    "no_resample",
    "for_prediction",
    "gpu",
    "gpu_count",
    "device",
    "diagnosis_source",
    "blank_policy",
    "tree_depth",
    "min_leaf",
    "method",
    "rules",
    "evidence",
}


def plan_digest(plan):
    value = {
        k: plan[k]
        for k in (
            "panel",
            "instance",
            "v2_id",
            "v2_roots",
            "v2_stage",
            "directory",
            "runtime",
            "name",
            "frozen_files",
            "frozen_code",
            "code_sha256",
            "sources",
            "source_contract",
            "log",
            "profile",
            "request",
        )
    }
    value["steps"] = [
        {
            "name": s["name"],
            "output": s["output"],
            "arguments": {k: v for k, v in s["arguments"].items() if k != "resume"},
        }
        for s in plan["steps"]
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_completed(plan, state):
    from ..server.resume_checks import check_plan

    if len(plan["steps"]) != len(state["steps"]):
        raise ValueError("来源步骤数量不一致")
    steps = []
    for step, result in zip(plan["steps"], state["steps"]):
        if result["status"] not in ("passed", "reused") or result["name"] != step["name"]:
            raise ValueError("来源步骤未完成或不匹配")
        steps.append(dict(step, reuse=result))
    return check_plan(dict(plan, steps=steps), "full")


def artifact(plan, key):
    if key == "prepared":
        values = {s["arguments"]["prepared"] for s in plan["steps"] if s["arguments"].get("prepared")}
        if len(values) == 1:
            return path(values.pop())
    for step in reversed(plan["steps"]):
        if key == "models" and step["name"] == "train":
            return path(Path(step["output"]) / "train")
        if key == "predictions" and step["name"] in ("predict", "test"):
            return path(
                Path(step["output"]) / ("predict" if step["name"] == "predict" else "test/predictions")
            )
    raise ValueError("来源任务缺少产物：" + key)


def source_record(values, key, stage):
    record, plan = catalog.get(values, key)
    if plan["v2_stage"] not in SOURCE[stage]:
        raise ValueError("来源阶段不匹配；所需：" + ",".join(SOURCE[stage]))
    state = read_json(Path(record["directory"]) / "job_state.json")
    if state["status"] != "passed" or catalog.active(record["directory"]):
        raise ValueError("来源任务尚未成功完成或仍有活动进程")
    verify_completed(plan, state)
    paths = {"prepared": str(artifact(plan, "prepared"))}
    if stage in ("predict", "predict-all"):
        paths["models"] = str(artifact(plan, "models"))
    if stage == "diagnosis":
        paths["predictions"] = str(artifact(plan, "predictions"))
    return {"id": key, "plan_sha256": file_hash(Path(record["directory"]) / "plan.json"), "paths": paths}


def build(values, profile, stage, name, source=None, options=None):
    if stage not in STAGES:
        raise ValueError("未知V2阶段")
    roots = locations(values)
    instance = claim(values)
    options = dict(options or {})
    if set(options) - RAW_OPTIONS:
        raise ValueError("V2不接受外部任务/缓存/权重路径：" + ",".join(set(options) - RAW_OPTIONS))
    if stage in SOURCE and not source:
        raise ValueError("必须显式选择来源任务 --input-job，不自动沿用最近任务")
    contract = source_record(values, source, stage) if stage in SOURCE else None
    if source and not contract:
        raise ValueError("该阶段独立运行，不接受来源任务")
    if contract and any(
        k in options
        for k in (
            "dataset",
            "size",
            "seed",
            "limit",
            "split_file",
            "spacing_excel",
            "target_spacing",
            "no_resample",
            "for_prediction",
        )
    ):
        raise ValueError("来源任务已冻结预处理，不能另改输入参数")
    args = make_parser().parse_args([])
    for key, value in options.items():
        setattr(args, key, value)
    args.runtime, args.results = str(roots["runtime"]), str(roots["results"])
    args.name, args.tasks, args.reset_settings = name, ",".join(STAGES[stage]), True
    args.device = options.get(
        "device", "cpu" if stage in ("environment", "prepare", "diagnosis", "synthetic") else "cuda"
    )
    args.mode = "train"
    args.input_job = None
    if contract:
        for key, value in contract["paths"].items():
            setattr(args, key, value)
    else:
        args.prepared = str(roots["prepared"] / uuid4().hex)
        args.preprocess_cache = "new" if stage in ("prepare", "assessment") else None
    if stage == "assessment":
        args.limit, args.allow_smoke = options.get("limit", 6), True
    from ..server.workflow import build_plan

    plan = build_plan(args, dict(profile, **{k: str(v) for k, v in roots.items()}), False)
    plan.update(
        panel=PANEL,
        instance=instance,
        v2_id=Path(plan["directory"]).name,
        v2_roots={k: str(v) for k, v in roots.items()},
        v2_stage=stage,
        sources=[source] if contract else [],
        source_contract=contract,
        request={"stage": stage, "name": name, "source": source, "options": options},
    )
    frozen, digest = snapshot.freeze(ROOT, roots["runtime"])
    plan.update(frozen_code=str(frozen), code_sha256=digest)
    snapshot.freeze_inputs(plan, ROOT, frozen)
    plan["frozen_plan_sha256"] = plan_digest(plan)
    validate(plan)
    return plan


def validate(plan):
    if plan.get("frozen_plan_sha256") != plan_digest(plan):
        raise ValueError("冻结任务参数已改变，拒绝运行或恢复")
    values = plan["v2_roots"]
    roots = locations(values)
    if plan.get("panel") != PANEL or plan.get("instance") != check(values):
        raise ValueError("任务不是当前V2实例")
    folder = path(plan["directory"])
    if (
        folder.parent != roots["results"]
        or folder.name != plan["v2_id"]
        or path(plan["runtime"]) != roots["runtime"]
    ):
        raise ValueError("V2任务路径不匹配")
    frozen = snapshot.verify(plan)
    if frozen.parent != roots["runtime"] / "releases":
        raise ValueError("冻结代码不在本V2 runtime")
    for filename, digest in plan.get("frozen_files", {}).items():
        if file_hash(path(filename)) != digest:
            raise ValueError("冻结配置已改变：" + filename)
    for step in plan["steps"]:
        if not path(step["output"]).is_relative_to(folder / "steps"):
            raise ValueError("步骤输出不在当前任务")
        args = step["arguments"]
        if args.get("prepared") and not path(args["prepared"]).is_relative_to(roots["prepared"]):
            raise ValueError("不能读取V1或外部预处理缓存")
        if args.get("python"):
            raise ValueError("V2使用自己的环境，不能指定其他环境")
        if args.get("models") and not path(args["models"]).is_relative_to(roots["results"]):
            raise ValueError("不能读取外部模型")
    contract = plan.get("source_contract")
    if contract:
        for step in plan["steps"]:
            for field, expected in contract["paths"].items():
                actual = step["arguments"].get(field)
                if actual is not None and path(actual) != path(expected):
                    raise ValueError("任务输入与显式来源不一致：" + field)
        record, old = catalog.get(values, contract["id"])
        if file_hash(Path(record["directory"]) / "plan.json") != contract["plan_sha256"]:
            raise ValueError("来源任务计划已改变")
        if read_json(Path(record["directory"]) / "job_state.json")["status"] != "passed" or catalog.active(
            record["directory"]
        ):
            raise ValueError("来源任务状态已改变")
        verify_completed(old, read_json(Path(record["directory"]) / "job_state.json"))
    return plan


def submit(values, profile, stage, name, source=None, options=None, foreground=False):
    if sys.platform != "linux":
        raise ValueError("后台任务请在Linux服务器运行；本机可预览和只读查询")
    from ..server import jobs

    claim(values)
    with jobs.acquire_lock(locations(values)["runtime"]) as lock:
        plan = build(values, profile, stage, name, source, options)
        with catalog.transaction(values) as index:
            key = plan["v2_id"]
            previous = index.get("selected")
            index["jobs"][key] = {
                "id": key,
                "directory": plan["directory"],
                "name": name,
                "stage": stage,
                "sources": plan["sources"],
                "status": "pending",
            }
            index["selected"] = key
        try:
            return jobs.submit(plan, lock, foreground)
        except BaseException as error:
            if not (Path(plan["directory"]) / "plan.json").is_file():
                with catalog.transaction(values) as index:
                    index["jobs"].pop(key, None)
                    index["selected"] = previous
                (locations(values)["runtime"] / "failed-submissions").mkdir(exist_ok=True)
                write_json(
                    locations(values)["runtime"] / "failed-submissions" / (key + ".json"),
                    {"plan": plan, "error": str(error)},
                )
            raise


def resume(values, key):
    record, old = catalog.get(values, key)
    validate(old)
    from ..server import jobs
    from ..server.resume import build_resume_plan

    args = make_parser().parse_args(["--resume-job", record["directory"], "--resume-check", "full"])
    with jobs.acquire_lock(locations(values)["runtime"]) as lock:
        plan = build_resume_plan(args, old["profile"], False)
        validate(plan)
        return jobs.submit(plan, lock)


def rerun(values, profile, key):
    record, old = catalog.get(values, key)
    if catalog.active(record["directory"]):
        raise ValueError("原任务仍活动")
    request = copy.deepcopy(old["request"])
    request["name"] += " 重跑"
    # New task and new resource decision. Source provenance remains explicit.
    return submit(values, profile, **request)


def diagnostics(values, key):
    import hashlib
    import json
    import re
    import zipfile

    from ..recovery import atomic_path

    record, plan = catalog.get(values, key)
    root = path(record["directory"])
    destination = (
        locations(values)["runtime"] / "diagnostic-exports" / ("diagnostics-" + uuid4().hex + ".zip")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    from ..server.inspection import log_paths

    candidates = {root / "job_state.json", root / "launcher.log"}
    for step in plan["steps"]:
        candidates.update(log_paths(root, step["name"], "all"))
        candidates.add(Path(step["output"]) / "status.json")
        candidates.add(Path(step["output"]) / "server.log")
        candidates.update(Path(step["output"]).rglob("*.cpu-resources.json"))
        candidates.update(Path(step["output"]).rglob("*.gpu-memory.json"))
        candidates.update(Path(step["output"]).rglob("cpu-allocation.json"))
    records, content, total = [], {}, 0
    for file in sorted(candidates):
        file = path(file)
        if not file.is_relative_to(root):
            raise ValueError("诊断日志路径越界")
        label = file.relative_to(root).as_posix()
        if not file.is_file():
            records.append({"path": label, "status": "missing"})
            continue
        before = file.stat()
        total += before.st_size
        if total > 32 * 2**20:
            raise ValueError("诊断文本超过32MiB；未截断或导出，请单独查看日志")
        raw = file.read_bytes()
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("活动日志正在变化，请稍后重试")
        text = raw.decode("utf-8", errors="replace")
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*|(?:token|password|secret)\s*[:=]\s*)[^\s,;]+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(r"(?i)(?:ghp_|github_pat_|sk-)[A-Za-z0-9_\-]{8,}", "[REDACTED]", text)
        text = re.sub(
            r"(?s)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED KEY]", text
        )
        content[label] = text.encode("utf-8")
        records.append(
            {"path": label, "status": "exported", "source_sha256": hashlib.sha256(raw).hexdigest()}
        )
    with atomic_path(destination) as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for label, data in content.items():
                archive.writestr(label, data)
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {"panel": PANEL, "task_id": key, "code_sha256": plan["code_sha256"], "files": records},
                    ensure_ascii=False,
                ),
            )
    return destination
