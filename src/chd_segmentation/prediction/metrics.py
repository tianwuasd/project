from pathlib import Path

import nibabel as nib
import numpy as np

from ..artifacts.store import digest, file_digest, read_json, require_result_output, write_json


def case_data_identity(prepared, row):
    return digest({"dataset": prepared["dataset"], "labels": prepared["dataset_json"]["labels"],
                   "image_sha256": row["sha256"]["image"], "spacing_xyz": row["spacing_xyz"],
                   "geometry_policy": row["geometry_policy"], "unknown_classes": row["unknown_classes"],
                   "ignore_label_ids": row.get("ignore_label_ids", []), "plan": prepared["plan"],
                   "technical": prepared["technical"]})


def evaluation_output(config, prepared, prediction_dir, output):
    protected = [prepared["manifest"]["data_root"], prediction_dir, config["run_dir"],
                 config["manifest"], config["plan"], config["pretrained"]]
    result = require_result_output(config, output, protected)
    if result.suffix.lower() != ".json":
        raise ValueError("评价输出必须是结果目录内的独立 JSON 文件")
    return result


def evaluate(prediction_dir, prepared, output, config):
    output = evaluation_output(config, prepared, prediction_dir, output)
    records = read_json(Path(prediction_dir) / "predictions.json")["cases"]
    selection = [(row, records[row["case_id"]]) for row in prepared["manifest"]["cases"] if row["case_id"] in records]
    if not selection:
        raise ValueError("没有与当前评价清单匹配的预测")
    identity = digest({"cases": [{"data": case_data_identity(prepared, row), "label": row["sha256"].get("label"),
                                 "prediction": receipt["sha256"], "checkpoint": receipt["checkpoint_sha256"]}
                                for row, receipt in selection], "metric": "supervised-dice-v2"})
    for row, receipt in selection:
        if receipt["data_identity"] != case_data_identity(prepared, row) or \
                receipt.get("reference_sha256") not in (None, row["sha256"]["label"]):
            raise ValueError("预测与评价的数据、类别或标签版本不同")
        if file_digest(row["label"]) != row["sha256"]["label"] or \
                file_digest(Path(prediction_dir) / receipt["file"]) != receipt["sha256"]:
            raise ValueError("评价来源标签或预测摘要不一致")
    if output.exists():
        previous = read_json(output)
        if previous.get("identity") != identity:
            raise FileExistsError("已有评价的输入或模型不同，请使用新的结果目录")
        return {**previous, "reused": True}
    results = []
    classes = len(prepared["dataset_json"]["labels"])
    for row in prepared["manifest"]["cases"]:
        receipt = records.get(row["case_id"])
        if receipt is None:
            continue
        path = Path(prediction_dir) / receipt["file"]
        roi = tuple(slice(begin, end) for begin, end in receipt["roi_xyz"])
        prediction_image, reference_image = nib.load(path), nib.load(row["label"])
        if prediction_image.shape != reference_image.shape:
            raise ValueError("评价标签与预测原图尺寸不同")
        prediction = np.asanyarray(prediction_image.dataobj[roi])
        truth = np.asanyarray(reference_image.dataobj[roi])
        if not np.isfinite(truth).all() or not np.equal(truth, np.rint(truth)).all():
            raise ValueError("评价标签不是有限整数")
        truth = truth.astype(np.int16)
        truth[np.isin(truth, row.get("ignore_label_ids", []))] = -1
        if not ((truth >= -1) & (truth < classes)).all():
            raise ValueError("评价标签超出协议范围")
        valid = (truth >= 0) & (truth < classes)
        by_class = {}
        for c in range(1, classes):
            if c in row["unknown_classes"]:
                continue
            target, found = (truth == c) & valid, (prediction == c) & valid
            reference_count, predicted_count = int(target.sum()), int(found.sum())
            by_class[str(c)] = {"reference_voxels": reference_count, "predicted_voxels": predicted_count,
                                "dice": float(2 * (target & found).sum() / (reference_count + predicted_count))
                                if reference_count + predicted_count else None}
        results.append({"case_id": row["case_id"], "scope": receipt["scope"], "roi_xyz": receipt["roi_xyz"], "classes": by_class})
    aggregates = {}
    for c in range(1, classes):
        values = [row["classes"][str(c)]["dice"] for row in results
                  if str(c) in row["classes"] and row["classes"][str(c)]["dice"] is not None]
        aggregates[str(c)] = {"cases": len(values), "mean_dice": float(np.mean(values)) if values else None}
    summary = {"identity": identity, "scope": prepared["scope"], "labels": prepared["dataset_json"]["labels"],
               "cases": results, "aggregate": aggregates,
               "reused": False, "interpretation": "技术子区域运行结果，不证明泛化效果" if prepared["scope"] == "technical" else "按有监督类别评价"}
    write_json(output, summary)
    return summary
