"""Three fold coordinator; reuses independent six-stage trainers and caches."""

from pathlib import Path

from ..models.chd_unet import ARCHITECTURE, GRID_ADAPTER
from ..recovery import file_hash, read_json
from .common import LABELS, NORMALIZATION, load_case, read_prepared, write_json
from .config import STAGES, UNAVAILABLE, effective_config, load_config
from .folds import CV_MODEL_FORMAT, make_plan


def train_threefold(prepared, output, config_path, mode, device, gpu_ids, resume, gpu_limit, lock_fd):
    from .checkpoints import read_collection
    from .train import train

    directory, data = read_prepared(prepared)
    source = load_config(config_path)
    source.pop("fold", None)
    if mode not in {"check", "smoke", "preflight", "train"}:
        raise ValueError("未知训练模式")
    if mode == "train" and (data.get("limit") is not None or source.get("profile") == "smoke"):
        raise ValueError("limit 缓存/smoke 配置不能冒充正式训练")
    plan = make_plan(data, source["seed"], source["cross_validation"]["development_scope"])
    if mode != "check":
        from .autotune import resolve
        from .folds import apply_fold
        from .train import training_rows

        rows = training_rows(apply_fold(data, plan["folds"][0]))["train"]
        source = resolve(source, directory, rows, output, gpu_limit or len(gpu_ids or []) or 1, resume)
    config = effective_config(source, mode, device)
    output = Path(output)
    contract = {
        "format": CV_MODEL_FORMAT,
        "recovery_version": 1,
        "mode": mode,
        "config": config,
        "prepared_sha256": file_hash(directory / "dataset.json"),
        "cache_hashes": {r["case_id"]: file_hash(directory / r["cache"]) for r in data["cases"]},
        "split_plan": plan,
    }
    contract_path = output / "training-contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("三折训练数据、配置或患者划分改变，不能续训")
    previous = read_json(output / "models.json") if (output / "models.json").is_file() else None
    if previous and (not contract_path.is_file() or previous.get("format") != CV_MODEL_FORMAT):
        raise ValueError("此目录不是当前三折训练记录，请使用新输出目录")
    if mode == "check" and previous:
        raise ValueError("缓存检查请使用独立输出目录")
    collection = {
        "format": CV_MODEL_FORMAT,
        "architecture": ARCHITECTURE,
        "grid_adapter": GRID_ADAPTER,
        "status": "running",
        "mode": mode,
        "labels": list(LABELS),
        "normalization": NORMALIZATION,
        "prepared_sha256": contract["prepared_sha256"],
        "config": config,
        "split_plan": plan,
        "ensemble": "per-stage arithmetic mean of probabilities; OOF for development patients",
        "folds": {},
        "unavailable": UNAVAILABLE,
    }
    report = {
        "task": "chd_threefold",
        "status": "running",
        "mode": mode,
        "prepared": str(directory),
        "test_cases_used_for_training": 0,
        "current_fold": None,
        "completed_folds": 0,
        "folds": {},
        "development_scope": plan["development_scope"],
        "external_cases": len(plan["external_case_ids"]),
    }
    for fold in plan["folds"]:
        key = f"fold_{fold['index']}"
        report["folds"][key] = {
            "status": "pending",
            **{part + "_cases": len(fold["cases"][part]) for part in ("train", "val", "test")},
            "patients": {part: len(ids) for part, ids in fold["patients"].items()},
        }
        collection["folds"][key] = {"index": fold["index"], "file": key + "/models.json"}
        prior_info = (previous or {}).get("folds", {}).get(key, {})
        if prior_info.get("sha256"):
            # Preserve every committed reference even if validation is interrupted
            # before reaching that fold during a second or later resume.
            collection["folds"][key]["sha256"] = prior_info["sha256"]
    # Do not rewrite any records until the prior run's contract has matched.
    write_json(contract_path, contract)
    write_json(output / "folds.json", plan)
    write_json(output / "effective-config.json", config)
    try:
        if mode == "check":
            for row in data["cases"]:
                load_case(directory, row)
            report["status"] = "checked"
            return report
        write_json(output / "models.json", collection)
        for fold in plan["folds"]:
            key = f"fold_{fold['index']}"
            child = output / key
            report["current_fold"] = key
            summary = report["folds"][key]
            summary["status"] = "running"
            write_json(output / "report.json", report)
            print(
                f"第 {fold['index'] + 1}/3 折：训练 {summary['train_cases']}，"
                f"验证 {summary['val_cases']}，测试 {summary['test_cases']} 例；测试折不参与更新或选模。",
                flush=True,
            )
            prior_info = (previous or {}).get("folds", {}).get(key, {})
            completed = resume and prior_info.get("sha256")
            if completed:
                if (
                    not (child / "models.json").is_file()
                    or file_hash(child / "models.json") != prior_info["sha256"]
                ):
                    raise ValueError("已完成折的模型清单改变，不能续训：" + key)
                _, saved = read_collection(child, allow_smoke=True)
                print("已核验，保留完成折：" + key, flush=True)
            else:
                train(
                    directory,
                    child,
                    output / "effective-config.json",
                    mode,
                    device,
                    gpu_ids,
                    resume=resume and child.exists(),
                    gpu_limit=gpu_limit,
                    fold_context=fold,
                    inherited_lock_fds=(lock_fd,),
                )
                _, saved = read_collection(child, allow_smoke=True)
            if (
                saved["config"] != {**config, "fold": fold}
                or saved["prepared_sha256"] != contract["prepared_sha256"]
                or saved["mode"] != mode
            ):
                raise ValueError("折模型来自不同配置或患者划分：" + key)
            collection["folds"][key]["sha256"] = file_hash(child / "models.json")
            summary.update(
                status="complete",
                stages={
                    stage: {
                        k: saved["stages"][stage].get(k)
                        for k in ("best_epoch", "best_val_loss", "epochs_completed")
                    }
                    for stage in STAGES
                },
            )
            report["completed_folds"] += 1
            write_json(output / "models.json", collection)
            write_json(output / "report.json", report)
            write_json(output / "training-summary.json", report)
        collection["status"] = "complete"
        report.update(status="passed", current_fold=None, models=str(output / "models.json"))
        write_json(output / "models.json", collection)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report.update(status=status, error=str(error))
        collection.update(status=status, error=str(error))
        if report["current_fold"]:
            report["folds"][report["current_fold"]]["status"] = status
        write_json(output / "models.json", collection)
        raise
    finally:
        write_json(output / "report.json", report)
        write_json(output / "training-summary.json", report)
