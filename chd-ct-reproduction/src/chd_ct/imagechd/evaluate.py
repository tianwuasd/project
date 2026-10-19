"""Separate evaluation in the prepared voxel grid; never changes model selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..cpu_runtime import allocation, iter_cases, volume_bytes
from ..recovery import output_lock
from .common import LABELS, load_case, read_prepared, safe_child, write_json
from .folds import held_out_index, select_rows, validate_plan
from .train import file_hash


def dice_metrics(prediction, target):
    if (
        prediction.shape != target.shape
        or not np.isin(prediction, range(8)).all()
        or not np.isin(target, [*range(8), 255]).all()
    ):
        raise ValueError("评估输入尺寸或标签无效")
    valid = target != 255
    if not valid.any():
        raise ValueError("没有可评估体素")
    dice = {}
    for label, name in enumerate(LABELS):
        p, t = (prediction == label) & valid, (target == label) & valid
        denominator = int(p.sum() + t.sum())
        dice[name] = 2 * int((p & t).sum()) / denominator if denominator else None
    values = [dice[name] for name in LABELS[1:] if dice[name] is not None]
    return {
        "dice": dice,
        "mean_foreground_dice": float(np.mean(values)) if values else None,
        "foreground_classes_in_mean": len(values),
        "ignored_voxels": int((~valid).sum()),
    }


def summarize_cases(rows):
    values = [r["mean_foreground_dice"] for r in rows if r["mean_foreground_dice"] is not None]
    return {
        "cases_evaluated": len(rows),
        "mean_case_foreground_dice": float(np.mean(values)) if values else None,
        "dice_by_class": {
            name: float(np.mean(present))
            if (present := [r["dice"][name] for r in rows if r["dice"][name] is not None])
            else None
            for name in LABELS
        },
    }


def evaluate(prepared, predictions, output, split="auto", inherited_lock_fds=()):
    cache, data = read_prepared(prepared)
    directory = Path(predictions).expanduser().resolve()
    report = json.loads((directory / "prediction-report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("prepared_sha256") != file_hash(cache / "dataset.json"):
        raise ValueError("预测未完成或不属于此数据清单")
    if data.get("for_prediction"):
        raise ValueError("无标签缓存不能评估")
    plan = report.get("split_plan")
    if plan:
        validate_plan(plan)
    rows, split = select_rows(data, split, plan=plan, fold=report.get("fold_context"))
    if not rows:
        raise ValueError("所选集合为空")
    available = {r["case_id"]: r for r in report["cases"]}
    result = {
        "status": "running",
        "space": "prepared_grid",
        "split": split,
        "physical_distance_metrics": None,
        "cases": [],
        "folds": {},
        "denominator": "per-case foreground Dice; ignore=255; both-absent classes are null",
    }
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        selected = allocation("evaluate", output, len(rows), volume_bytes(rows, 32))
        items = [
            dict(
                cache=str(cache),
                directory=str(directory),
                row=row,
                available={row["case_id"]: available[row["case_id"]]} if row["case_id"] in available else {},
                plan=plan,
            )
            for row in rows
        ]
        with output_lock(output) as lock_fd:
            result["cases"] = list(
                iter_cases(
                    "chd_ct.imagechd.evaluate",
                    "evaluate_case",
                    items,
                    output,
                    selected,
                    (*inherited_lock_fds, lock_fd),
                )
            )
        for index in range(3):
            subset = [
                r
                for r in result["cases"]
                if r["inference_protocol"] == "out_of_fold" and r["fold_indices"] == [index]
            ]
            if subset:
                result["folds"][f"fold_{index}"] = summarize_cases(subset)
        protocols = {r["inference_protocol"] for r in result["cases"]}
        result.update(status="passed", protocols=sorted(protocols), **summarize_cases(result["cases"]))
        return result
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "evaluation.json", result)


def evaluate_case(cache, directory, row, available, plan):
    cache, directory = Path(cache), Path(directory)
    if row["case_id"] not in available:
        raise ValueError("所选集合缺少预测，请先完成该集合的预测")
    info = available[row["case_id"]]
    if plan:
        index = held_out_index(plan, row)
        expected_protocol = "out_of_fold" if index is not None else "threefold_ensemble"
        expected_folds = [index] if index is not None else [0, 1, 2]
        if info.get("inference_protocol") != expected_protocol or info.get("fold_indices") != expected_folds:
            raise ValueError("预测折次不符合独立评估要求，开发集必须使用未见过该患者的测试折")
    path = safe_child(directory, info["grid_prediction"])
    if info.get("grid_prediction_sha256") and file_hash(path) != info["grid_prediction_sha256"]:
        raise ValueError("预测文件校验失败")
    with np.load(path, allow_pickle=False) as record:
        prediction = record["prediction"]
    _, target = load_case(cache, row)
    return {
        "case_id": row["case_id"],
        "inference_protocol": info.get("inference_protocol", "single_split"),
        "fold_indices": info.get("fold_indices", []),
        **dice_metrics(prediction, target),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="单独评估 CHD 分割；不训练，不读取毫米距离")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["auto", "cv", "train", "val", "test"], default="auto")
    args = parser.parse_args(argv)
    evaluate(args.prepared, args.predictions, args.output, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
