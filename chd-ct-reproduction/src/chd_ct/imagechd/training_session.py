"""Shared fold preparation, frozen contracts and model-manifest publication."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..models.chd_unet import ARCHITECTURE, GRID_ADAPTER
from ..recovery import file_hash, read_json
from .common import LABELS, MODEL_FORMAT, NORMALIZATION, load_case, read_prepared, write_json
from .config import STAGES, UNAVAILABLE, effective_config, load_config


@contextmanager
def training_session(prepared, output, config_path, mode, device, concurrency, resume, fold_context=None):
    from .train import training_rows

    directory, data = read_prepared(prepared)
    config = load_config(config_path)
    config.pop("fold", None)
    if fold_context is not None:
        from .folds import apply_fold

        data = apply_fold(data, fold_context)
        config["fold"] = fold_context
    rows = training_rows(data)
    if mode not in {"check", "smoke", "preflight", "train"}:
        raise ValueError("未知训练模式")
    if mode == "train" and (data.get("limit") is not None or config.get("profile") == "smoke"):
        raise ValueError("limit 缓存/smoke 配置不能冒充正式训练")
    if (
        device == "cpu"
        and mode not in {"check", "smoke"}
        and any(s.get("base", 0) * 2 ** (s.get("levels", 1) - 1) >= 768 for s in config["stages"].values())
    ):
        raise ValueError("论文规模配置需要 GPU 预检；CPU 请使用 --mode smoke 或独立的缩小配置")
    output = Path(output).expanduser().resolve()
    report = {
        "status": "running",
        "task": "chd_multistage",
        "mode": mode,
        "train_cases": len(rows["train"]),
        "val_cases": len(rows["val"]),
        "test_cases_used": 0,
        "prepared": str(directory),
        "cache_resolution": data["resolution"],
        "unavailable": UNAVAILABLE,
    }
    collection = {
        "status": "running",
        "format": MODEL_FORMAT,
        "architecture": ARCHITECTURE,
        "grid_adapter": GRID_ADAPTER,
        "labels": list(LABELS),
        "normalization": NORMALIZATION,
        "mode": mode,
        "prepared_sha256": file_hash(directory / "dataset.json"),
        "stages": {},
        "unavailable": UNAVAILABLE,
    }
    session = dict(
        directory=directory,
        data=data,
        config=config,
        rows=rows,
        output=output,
        report=report,
        collection=collection,
    )
    validated = False
    try:
        if mode == "check":
            if (output / "models.json").exists():
                raise ValueError("缓存检查请使用独立输出目录")
            validated = True
            for selected in rows.values():
                for row in selected:
                    load_case(directory, row)
            report["status"] = "checked"
            yield session
            return
        from .autotune import resolve

        config = resolve(config, directory, rows["train"], output, concurrency, resume)
        config = effective_config(config, mode, device)
        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        collection["config"] = config
        contract = dict(
            config=config,
            mode=mode,
            prepared_sha256=collection["prepared_sha256"],
            architecture=ARCHITECTURE,
            recovery_version=1,
            cache_hashes={r["case_id"]: file_hash(directory / r["cache"]) for r in data["cases"]},
        )
        contract_path = output / "training-contract.json"
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError("训练数据或配置已改变，不能续训")
        if resume and not contract_path.exists() and (output / "models.json").exists():
            raise ValueError("旧版训练未记录优化器，不能精确续训；请另建训练目录")
        validated = True
        write_json(output / "report.json", report)
        write_json(contract_path, contract)
        write_json(output / "effective-config.json", config)
        write_json(output / "models.json", collection)
        session["config"] = config
        yield session
    except BaseException as error:
        if not validated or collection.get("status") == "complete":
            raise
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        collection.update(status="failed", error=str(error))
        write_json(output / "models.json", collection)
        raise
    finally:
        if validated:
            write_json(output / "report.json", report)


def commit_stage(session, stage, info):
    session["collection"]["stages"][stage] = info
    write_json(session["output"] / "models.json", session["collection"])


def finish_session(session):
    collection, report, output = session["collection"], session["report"], session["output"]
    collection["stages"] = {stage: collection["stages"][stage] for stage in STAGES}
    collection["status"] = "complete"
    write_json(output / "models.json", collection)
    report.update(status="passed", models=str(output / "models.json"), stages=list(STAGES))
    write_json(output / "report.json", report)
