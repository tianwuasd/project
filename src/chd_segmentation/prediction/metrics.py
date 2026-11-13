from pathlib import Path

import nibabel as nib
import numpy as np

from ..artifacts.store import file_digest, read_json, write_json


def evaluate(prediction_dir, prepared, output):
    records = read_json(Path(prediction_dir) / "predictions.json")["cases"]
    results = []
    classes = len(prepared["dataset_json"]["labels"])
    for row in prepared["manifest"]["cases"]:
        receipt = records.get(row["case_id"])
        if receipt is None:
            continue
        path = Path(prediction_dir) / receipt["file"]
        if file_digest(path) != receipt["sha256"]:
            raise ValueError("待评价预测摘要不匹配")
        roi = tuple(slice(begin, end) for begin, end in receipt["roi_xyz"])
        prediction = np.asanyarray(nib.load(path).dataobj[roi])
        truth = np.asanyarray(nib.load(row["label"]).dataobj[roi])
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
