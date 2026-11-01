"""Read-only links to a historical workflow, checked again before submission."""

from pathlib import Path

from ..recovery import file_hash, read_json
from . import inspection
from .extension import source_stamp
from .prompts import ask

CACHE_USERS = {
    "preprocess",
    "train",
    "smoke",
    "preflight",
    "predict",
    "test",
    "evaluate",
    "diagnosis-features",
}


def load_source(directory, runtime):
    directory = Path(directory).expanduser().resolve()
    stamp = source_stamp(directory)
    plan = read_json(directory / "plan.json")
    state = read_json(directory / "job_state.json")
    if not isinstance(plan, dict) or not isinstance(state, dict):
        raise ValueError("输入任务计划/状态必须是有效对象")
    if (
        plan.get("schema_version") != 1
        or Path(plan["directory"]).resolve() != directory
        or Path(plan["runtime"]).resolve() != Path(runtime).resolve()
    ):
        raise ValueError("输入任务的版本、目录或 runtime 不符；请使用原 runtime")
    if inspection.job_active(directory):
        raise ValueError("输入任务仍有存活进程，请等待完成或停止后再选择")
    from .jobs import identity, job_owner

    owner = job_owner(directory, state)
    if (
        owner.get("pid")
        and identity(owner["pid"]) is not None
        and (not owner.get("identity") or owner["identity"] == identity(owner["pid"]))
    ):
        raise ValueError("输入任务主进程仍存活，请等待退出")
    if len(plan["steps"]) != len(state["steps"]) or any(
        step["name"] != prior["name"] or step["output"] != prior["output"]
        for step, prior in zip(plan["steps"], state["steps"])
    ):
        raise ValueError("输入任务计划与状态不一致")
    record = dict(directory=str(directory), name=plan.get("name", directory.name), source_stamp=stamp)
    return plan, state, record


def cache_metadata(prepared):
    prepared = Path(prepared).resolve()
    manifest = prepared / "dataset.json"
    if not manifest.is_file():
        raise ValueError("输入任务预处理未完成或已移走：缺少 dataset.json")
    data = read_json(manifest)
    if data.get("status") != "prepared" or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("输入任务没有有效的完整预处理清单")
    for row in data["cases"]:
        cache = row.get("cache") if isinstance(row, dict) else None
        if not cache:
            raise ValueError("预处理清单缺少病例缓存路径")
        path = (prepared / cache).resolve()
        if not path.is_relative_to(prepared) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("预处理缓存缺失、为空或越界：" + str(path))
    return {
        str(path): file_hash(path)
        for path in (manifest, prepared / "preprocess-contract.json")
        if path.is_file()
    }


def choose_input(args, names, runtime, results, interactive):
    value = getattr(args, "input_job", None)
    if not set(names).intersection(CACHE_USERS):
        if value and value != "none":
            raise ValueError("所选步骤不使用预处理结果，无需输入任务")
        return None
    if value is None and interactive:
        records = inspection.history(runtime, results)
        print("输入任务：0 手动设置缓存；n 不复用、从头预处理；历史任务编号复用其最终缓存并跳过预处理。")
        for index, job in enumerate(records, 1):
            print(f"  {index}. {inspection.name(job)} | {inspection.status(job)}\n     {job}")
        print("d 手动输入历史 workflow 目录")
        choice = ask("输入任务 [0]：") or "0"
        if choice == "d":
            value = ask("历史 workflow 完整目录：").strip().strip("\"'")
            if not value:
                raise ValueError("输入任务目录不能为空")
        elif choice == "0":
            value = None
        elif choice.lower() == "n":
            value = "none"
        elif choice.isdigit() and 1 <= int(choice) <= len(records):
            value = str(records[int(choice) - 1])
        else:
            raise ValueError("请选择列表中的输入任务")
    if not value or value == "none":
        if value == "none" and args.preprocess_cache == "reuse":
            raise ValueError("不复用输入任务与 --preprocess-cache reuse 矛盾；请选择 new")
        if value == "none" and "preprocess" not in names:
            raise ValueError("不复用、从头预处理需要选择 preprocess；手工使用缓存请选择 0 或省略 --input-job")
        if value == "none" and "preprocess" in names and not args.preprocess_cache:
            args.preprocess_cache = "new"
        return None
    if args.preprocess_cache or args.dataset or args.prepared:
        raise ValueError("选择输入任务后不能同时指定 dataset/prepared/缓存策略；请直接使用它的缓存")
    # Explicit preprocessing changes cannot silently disappear when skipping that step.
    from .launcher import make_parser

    defaults = make_parser().parse_args([])
    for key in (
        "size",
        "seed",
        "limit",
        "split_file",
        "spacing_excel",
        "target_spacing",
        "no_resample",
        "for_prediction",
    ):
        if getattr(args, key, None) != getattr(defaults, key, None):
            raise ValueError("输入任务冻结预处理参数，不能另改：" + key)
    old, state, record = load_source(value, runtime)
    paths = {s["arguments"]["prepared"] for s in old["steps"] if s["arguments"].get("prepared")}
    if len(paths) != 1:
        raise ValueError("输入任务没有唯一的预处理缓存来源")
    preprocess = [r for r in state["steps"] if r["name"] == "preprocess"]
    if preprocess and preprocess[0]["status"] not in {"passed", "reused"}:
        raise ValueError("输入任务预处理尚未成功完成")
    if not any(r["status"] in {"passed", "reused"} for r in state["steps"]):
        raise ValueError("输入任务没有成功步骤")
    prepared = str(Path(paths.pop()).resolve())
    metadata = cache_metadata(prepared)
    # Existing historical hashes are evidence; freshly computed hashes only freeze this selection.
    from .resume_checks import check_plan

    checks = [
        {**s, "reuse": r} for s, r in zip(old["steps"], state["steps"]) if r["status"] in {"passed", "reused"}
    ]
    checks.append(dict(name="输入缓存", arguments=dict(task="train", prepared=prepared), output="unused"))
    verification = check_plan(dict(steps=checks), "simple")
    record.update(prepared=prepared, metadata=metadata, verification=verification)
    args.prepared = prepared
    print(
        f"输入任务：{record['name']}\n复用预处理：{prepared}\n不会执行预处理；新结果使用独立目录。",
        flush=True,
    )
    return record


def verify_lineage(plan):
    """Called under the destination runtime lock, and again on initial worker entry."""
    if plan.get("resume_existing"):
        return  # Recovery validates its own frozen plan, not the parent's later attempts.
    for key in ("input_job", "rerun_of"):
        record = plan.get(key)
        if not record:
            continue
        directory = Path(record["directory"])
        load_source(directory, plan["runtime"])
        if source_stamp(directory) != record["source_stamp"]:
            raise ValueError("输入任务记录已改变，请重新选择，未提交新任务")
        if record.get("prepared") and cache_metadata(record["prepared"]) != record["metadata"]:
            raise ValueError("输入任务缓存清单已改变，请重新选择")
        for path, digest in record.get("frozen_files", {}).items():
            if not Path(path).is_file() or file_hash(path) != digest:
                raise ValueError("重跑的冻结配置已改变：" + path)
