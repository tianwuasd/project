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
    if result.exists():
        raise FileExistsError("评价输出已存在，请指定新的报告文件")
    return result


def evaluate(prediction_dir, prepared, output, config):
    output = evaluation_output(config, prepared, prediction_dir, output)
    records = read_json(Path(prediction_dir) / "predictions.json")["cases"]
    results = []
    classes = len(prepared["dataset_json"]["labels"])
    for row in prepared["manifest"]["cases"]:
        receipt = records.get(row["case_id"])
        if receipt is None:
            continue
        if receipt["data_identity"] != case_data_identity(prepared, row):
            raise ValueError("预测与评价的数据、类别或空间身份不同")
        expected_reference = row["sha256"]["label"]
        if receipt.get("reference_sha256") not in (None, expected_reference):
            raise ValueError("评价标签版本与预测记录不同")
        if file_digest(row["label"]) != expected_reference:
            raise ValueError("评价标签摘要与已审计来源不同")
        path = Path(prediction_dir) / receipt["file"]
        if file_digest(path) != receipt["sha256"]:
            raise ValueError("待评价预测摘要不匹配")
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
                                if reference_count else None}
        results.append({"case_id": row["case_id"], "scope": receipt["scope"], "roi_xyz": receipt["roi_xyz"], "classes": by_class})
    summary = {"scope": prepared["scope"], "cases": results, "interpretation": "技术子区域运行结果，不证明泛化效果" if prepared["scope"] == "technical" else "按有监督类别评价"}
    write_json(output, summary)
    return summary
