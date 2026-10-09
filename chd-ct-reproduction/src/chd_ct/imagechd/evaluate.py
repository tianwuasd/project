"""Separate evaluation in the prepared voxel grid; never changes model selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import LABELS, load_case, read_prepared, safe_child, write_json
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


def evaluate(prepared, predictions, output, split="test"):
    cache, data = read_prepared(prepared)
    directory = Path(predictions).expanduser().resolve()
    report = json.loads((directory / "prediction-report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("prepared_sha256") != file_hash(cache / "dataset.json"):
        raise ValueError("预测未完成或不属于此数据清单")
    if data.get("for_prediction"):
        raise ValueError("无标签缓存不能评估")
    rows = [r for r in data["cases"] if r.get("split") == split]
    if not rows:
        raise ValueError("所选集合为空")
    available = {r["case_id"]: r for r in report["cases"]}
    result = {
        "status": "running",
        "space": "prepared_grid",
        "split": split,
        "physical_distance_metrics": None,
        "cases": [],
        "denominator": "per-case foreground Dice; ignore=255; both-absent classes are null",
    }
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        for row in rows:
            if row["case_id"] not in available:
                raise ValueError("所选集合缺少预测，请先完成该集合的预测")
            with np.load(
                safe_child(directory, available[row["case_id"]]["grid_prediction"]), allow_pickle=False
            ) as record:
                prediction = record["prediction"]
            _, target = load_case(cache, row)
            result["cases"].append({"case_id": row["case_id"], **dice_metrics(prediction, target)})
        values = [r["mean_foreground_dice"] for r in result["cases"] if r["mean_foreground_dice"] is not None]
        result.update(
            status="passed",
            cases_evaluated=len(rows),
            mean_case_foreground_dice=float(np.mean(values)) if values else None,
        )
        return result
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "evaluation.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="单独评估 CHD 分割；不训练，不读取毫米距离")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args(argv)
    evaluate(args.prepared, args.predictions, args.output, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
