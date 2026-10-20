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
    from .gpu_hold import records
    from .inspection import model_runs
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
        for part in ("train", "preflight", "predict", "test/predictions"):
            run = Path(step["output"]) / part
            # Keep the legacy queue even when fold directories are already present.
            queues = dict.fromkeys(folder / "queue.json" for folder in [run, *model_runs(run)])
            for queue in queues:
                if not queue.is_file() or not queue.resolve().is_relative_to(run.resolve()):
                    continue
                for record in read_json(queue).get("stages", {}).values():
                    if (
                        record.get("pid")
                        and record.get("identity")
                        and identity(record["pid"]) == record["identity"]
                    ):
                        live[record["pid"]] = dict(record, process_group=False)
    for step in plan["steps"]:
        roots = [Path(step["output"])]
        if step.get("arguments", {}).get("task") == "preprocess" and step["arguments"].get("prepared"):
            roots.append(Path(step["arguments"]["prepared"]))
        for root in roots:
            for queue in root.rglob(".cpu-work/*/queue.json"):
                for record in read_json(queue).get("stages", {}).values():
                    if record.get("identity") and identity(record.get("pid")) == record["identity"]:
                        live[record["pid"]] = dict(record, process_group=False)
    for folder in (Path(plan["directory"]) / "gpu-holds").glob("*"):
        for record in records(folder):
            if record.get("identity") and identity(record["pid"]) == record["identity"]:
                live[record["pid"]] = dict(record, process_group=True)
    return list(live.values())


def reject_live_commands(plan):
    live = live_commands(plan)
    if live:
        raise ValueError(
            "原任务子进程仍存活 PID=" + ",".join(str(r["pid"]) for r in live) + "; 请先 --stop，不重复运行"
        )


def build_resume_plan(args, profile, interactive):
    from .extension import append_steps, choose_append, source_stamp, verify_source
    from .jobs import identity, job_owner

    if (
        args.task
        or args.tasks
        or getattr(args, "name", None)
        or getattr(args, "preprocess_cache", None)
        or args.reset_settings
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
    stamp = source_stamp(directory)
    plan = copy.deepcopy(read_json(directory / "plan.json"))
    plan["resume_source"] = stamp
    plan.setdefault("profile", profile)
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
    if len(state["steps"]) != len(plan["steps"]):
        raise ValueError("任务状态和计划步骤数量不符")
    names = choose_append(args, plan, state["status"] == "passed", interactive)
    if not names and any(getattr(args, key, None) for key in INPUTS):
        raise ValueError("续跑使用原任务输入；新增输入只能用于 --append-tasks 的新步骤")
    print("校验已完成步骤与输入来源，请等待……", flush=True)
    prompted = False
    for step, prior in zip(plan["steps"], state["steps"]):
        if step["name"] != prior["name"] or step["output"] != prior["output"]:
            raise ValueError("任务状态和计划步骤不符")
        step.pop("reuse", None)
        step.pop("restart", None)
        arguments = step["arguments"]
        if arguments["task"] == "environment" and any(s["name"] == "preprocess" for s in plan["steps"]):
            arguments["device"] = "cpu"
        if prior["status"] in {"passed", "reused"}:
            if not prior.get("snapshot") and arguments["task"] != "environment":
                raise ValueError("已完成步骤缺少校验记录，保留产物但不能自动重跑：" + step["name"])
            if prior.get("snapshot"):
                verify_snapshot(prior["snapshot"])
            step["reuse"] = prior
            print("保留已完成步骤：" + step["name"], flush=True)
            continue
        if arguments["task"] in LONG_TASKS:
            arguments["resume"] = True
        else:
            step["restart"] = True
        from .inference_phases import gpu_ready

        cpu_tail = arguments["task"] in {"test", "predict"} and gpu_ready(step)
        if arguments.get("device") == "cuda" and not cpu_tail:
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
            if args.gpu_count is not None and arguments["task"] in {"train", "environment", "test", "predict"}:
                arguments["gpu_count"] = args.gpu_count
        if args.python:
            arguments["python"] = str(Path(args.python).expanduser().resolve())
    plan.update(resume_existing=True, reset_settings=False)
    plan = append_steps(args, plan, names, interactive)
    verify_source(plan)
    return plan
