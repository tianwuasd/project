"""Resume a frozen workflow: verify completed outputs, retain partial long steps."""

import copy
from pathlib import Path

from ..recovery import file_hash, read_json
from .launcher import choose, personal_path, show_gpus

LONG_TASKS = {"preprocess", "train", "predict", "test"}
INPUTS = (
    "dataset",
    "prepared",
    "config",
    "models",
    "predictions",
    "split_file",
    "spacing_excel",
    "diagnosis_source",
    "diagnosis_labels",
    "features",
    "classifier",
    "evidence",
    "rules",
    "diagnoses",
)


def tree_hashes(path):
    path = Path(path)
    if path.is_file():
        return {str(path.resolve()): file_hash(path)}
    if not path.is_dir():
        raise ValueError("恢复所需路径不存在：" + str(path))
    return {
        str(p.resolve()): file_hash(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
        and p.suffix not in {".log", ".tmp"}
        and ".interrupted-" not in str(p)
    }


def snapshot(step):
    arguments = step["arguments"]
    inputs = {}
    for key in INPUTS:
        if arguments.get(key) and (key != "prepared" or arguments["task"] != "preprocess"):
            inputs.update(tree_hashes(arguments[key]))
    target = (
        Path(arguments["prepared"])
        if arguments["task"] == "preprocess"
        else Path(step["output"]) / arguments["task"]
    )
    outputs = tree_hashes(target) if target.exists() else {}
    input_roots = [
        str(arguments[k])
        for k in INPUTS
        if arguments.get(k) and (k != "prepared" or arguments["task"] != "preprocess")
    ]
    return dict(
        inputs=inputs,
        outputs=outputs,
        input_roots=input_roots,
        output_roots=[str(target)] if target.exists() else [],
    )


def verify_snapshot(record):
    for group, roots in (("inputs", "input_roots"), ("outputs", "output_roots")):
        if roots in record:
            actual = {}
            for root in record[roots]:
                actual.update(tree_hashes(root))
            if actual != record[group]:
                raise ValueError("已完成步骤的输入或输出改变（包含新增/缺失文件），不能直接续跑")
            continue
        for name, digest in record[group].items():
            path = Path(name)
            if not path.is_file() or file_hash(path) != digest:
                raise ValueError("已完成步骤的输入或输出已改变，不能直接续跑：" + name)


def live_commands(plan):
    from .jobs import identity

    live = {}
    for step in plan["steps"]:
        receipt = Path(step["output"]) / "active-command.json"
        if receipt.is_file():
            record = read_json(receipt)
            if (
                record.get("running")
                and record.get("identity")
                and identity(record["pid"]) == record["identity"]
            ):
                live[record["pid"]] = dict(record, process_group=True)
        for part in ("train", "preflight"):
            queue = Path(step["output"]) / part / "queue.json"
            if queue.is_file():
                for record in read_json(queue).get("stages", {}).values():
                    if (
                        record.get("pid")
                        and record.get("identity")
                        and identity(record["pid"]) == record["identity"]
                    ):
                        live[record["pid"]] = dict(record, process_group=False)
    return list(live.values())


def reject_live_commands(plan):
    live = live_commands(plan)
    if live:
        raise ValueError(
            "原任务子进程仍存活 PID=" + ",".join(str(r["pid"]) for r in live) + "; 请先 --stop，不重复运行"
        )


def build_resume_plan(args, profile, interactive):
    from .jobs import identity, job_owner

    if (
        args.task
        or args.tasks
        or getattr(args, "name", None)
        or args.reset_settings
        or any(getattr(args, k, None) for k in INPUTS)
    ):
        raise ValueError("--resume-job 使用原任务的名称、数据、配置和步骤；不要同时改动名称/输入/步骤")
    if args.resume_job == "latest":
        runtime = personal_path(choose(args.runtime, None, profile["runtime"], "环境与缓存目录", interactive))
        pointer = runtime / "last-job.json"
        if not pointer.is_file():
            raise ValueError("没有可恢复的历史任务")
        directory = Path(read_json(pointer)["directory"]).resolve()
    else:
        directory = Path(args.resume_job).expanduser().resolve()
    plan = copy.deepcopy(read_json(directory / "plan.json"))
    if plan.get("schema_version") != 1 or Path(plan["directory"]).resolve() != directory:
        raise ValueError("任务计划版本或路径不符")
    if args.runtime and Path(args.runtime).expanduser().resolve() != Path(plan["runtime"]).resolve():
        raise ValueError("恢复须使用原任务 runtime")
    state = read_json(directory / "job_state.json")
    owner = job_owner(directory, state)
    if owner.get("pid") and identity(owner["pid"]) is not None:
        if not owner.get("identity") or identity(owner["pid"]) == owner["identity"]:
            raise ValueError("原任务进程仍在，请先查看 --status 或 --stop")
    reject_live_commands(plan)
    for name, digest in plan.get("frozen_files", {}).items():
        if not Path(name).is_file() or file_hash(name) != digest:
            raise ValueError("任务配置文件已经改变：" + name)
    if state["status"] == "passed":
        raise ValueError("此流程已经完成；如需新实验请新建任务")
    if len(state["steps"]) != len(plan["steps"]):
        raise ValueError("任务状态和计划步骤数量不符")
    print("校验已完成步骤与输入来源，请等待……", flush=True)
    prompted = False
    for step, prior in zip(plan["steps"], state["steps"]):
        if step["name"] != prior["name"] or step["output"] != prior["output"]:
            raise ValueError("任务状态和计划步骤不符")
        step.pop("reuse", None)
        step.pop("restart", None)
        arguments = step["arguments"]
        if prior["status"] in {"passed", "reused"} and (
            prior.get("snapshot") or arguments["task"] == "environment"
        ):
            if prior.get("snapshot"):
                verify_snapshot(prior["snapshot"])
            step["reuse"] = prior
            print("保留已完成步骤：" + step["name"], flush=True)
            continue
        if arguments["task"] in LONG_TASKS:
            arguments["resume"] = True
        else:
            step["restart"] = True
        if arguments.get("device") == "cuda":
            if interactive and not prompted:
                show_gpus()
                args.gpu_count = int(
                    choose(
                        args.gpu_count,
                        arguments.get("gpu_count"),
                        1,
                        "恢复时最多使用几张 GPU（1—5；不足时等待或少卡运行）",
                        True,
                    )
                )
                args.gpu = choose(
                    args.gpu, arguments.get("gpu"), "auto", "恢复使用 GPU 编号/UUID，逗号分隔，或 auto", True
                )
                prompted = True
            if args.gpu is not None:
                arguments["gpu"] = args.gpu
            if args.gpu_count is not None and arguments["task"] in {"train", "environment"}:
                arguments["gpu_count"] = args.gpu_count
        if args.python:
            arguments["python"] = str(Path(args.python).expanduser().resolve())
    plan.update(resume_existing=True, reset_settings=False)
    return plan
