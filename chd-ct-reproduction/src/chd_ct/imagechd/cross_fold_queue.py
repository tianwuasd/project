"""One global GPU budget for independent fold/stage pairs."""

import os
import sys
from contextlib import ExitStack

from ..recovery import output_lock
from .common import write_json
from .config import STAGES
from .gpu_queue import options
from .stage_queue import run_queue
from .training_session import commit_stage, finish_session, training_session


def run_folds(
    directory,
    output,
    mode,
    device,
    gpu_ids,
    resume,
    gpu_limit,
    lock_fd,
    folds,
    report,
    check_completed,
    on_fold_complete,
):
    queue_options = options(gpu_ids, gpu_limit, device, mode)
    sessions, resuming, descriptors = {}, {}, [lock_fd]
    with ExitStack() as stack:
        for fold in folds:
            key = f"fold_{fold['index']}"
            child = output / key
            resuming[key] = resume and child.exists()
            child.mkdir(exist_ok=resume)
            descriptors.append(stack.enter_context(output_lock(child)))
            if check_completed(fold):
                continue
            sessions[key] = stack.enter_context(
                training_session(
                    directory,
                    child,
                    output / "effective-config.json",
                    mode,
                    device,
                    gpu_limit or len(gpu_ids),
                    resuming[key],
                    fold,
                )
            )
        tasks = [f"{key}/{stage}" for key in sessions for stage in STAGES]

        def command_for(task):
            key, stage = task.split("/")
            return [
                sys.executable,
                "-m",
                "chd_ct.imagechd.stage_worker",
                "--prepared",
                directory,
                "--output",
                output / key,
                "--stage",
                stage,
                "--mode",
                mode,
                *(["--resume"] if resuming[key] else []),
                *(["--check-environment"] if gpu_limit is not None else []),
            ]

        def completed(task, info):
            key, stage = task.split("/")
            session = sessions[key]
            commit_stage(session, stage, info)
            if len(session["collection"]["stages"]) == len(STAGES):
                finish_session(session)
                on_fold_complete(session["config"]["fold"])

        def publish(state):
            report["current_folds"] = sorted(
                {
                    task.split("/")[0]
                    for task, record in state["stages"].items()
                    if record["status"] == "running"
                }
            )
            report["current_fold"] = next(iter(report["current_folds"]), None)
            for key in sessions:
                stages = {stage: state["stages"][f"{key}/{stage}"] for stage in STAGES}
                statuses = {item["status"] for item in stages.values()}
                status = (
                    "complete"
                    if statuses == {"complete"}
                    else "failed"
                    if "failed" in statuses
                    else "interrupted"
                    if "interrupted" in statuses
                    else "running"
                    if statuses.intersection({"running", "complete"})
                    else "pending"
                )
                if report["folds"][key]["status"] != "complete":
                    report["folds"][key]["status"] = status
                child_state = dict(
                    state,
                    stages=stages,
                    fold=key,
                    status=status,
                    active_gpus=sum(r["status"] == "running" for r in stages.values()),
                    pending_stages=sum(r["status"] in {"pending", "not_started"} for r in stages.values()),
                )
                write_json(output / key / "queue.json", child_state)
            write_json(output / "report.json", report)

        run_queue(
            tasks,
            output=output,
            command_for=command_for,
            env=os.environ.copy(),
            on_complete=completed,
            lock_fds=tuple(descriptors),
            **queue_options,
            dependencies={f"{key}/blood_lstm": [f"{key}/blood2d"] for key in sessions},
            result_path_for=lambda task: output / (task + ".result.json"),
            log_path_for=lambda task: output
            / task.split("/")[0]
            / "stage_logs"
            / (task.split("/")[1] + ".log"),
            on_state=publish,
        )
