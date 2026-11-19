from pathlib import Path
from time import perf_counter

import numpy as np

from ..artifacts.store import digest, read_json, require_result_output, write_json
from ..observability.events import event, logged, stage
from .evaluation_blocks import count_labels


def case_data_identity(prepared, row, *, include_generation=False):
    plan = prepared['plan'] if include_generation else {
        key:value for key,value in prepared['plan'].items() if key!='generation'}
    return digest({"dataset": prepared["dataset"], "labels": prepared["dataset_json"]["labels"],
                   "image_sha256": row.get('sha256', {}).get('image', row.get('image')), "spacing_xyz": row["spacing_xyz"],
                   "geometry_policy": row["geometry_policy"], "unknown_classes": row["unknown_classes"],
                   "ignore_label_ids": row.get("ignore_label_ids", []), "plan": plan,
                   "technical": prepared["technical"]})


def evaluation_output(config, prepared, prediction_dir, output):
    protected = [prepared["manifest"]["data_root"], prediction_dir, config["run_dir"],
                 config["manifest"], config["plan"], config["pretrained"]]
    result = require_result_output(config, output, protected)
    if result.suffix.lower() != ".json":
        raise ValueError("评价输出必须是结果目录内的独立 JSON 文件")
    return result


@logged("evaluation", config_argument=3)
def evaluate(prediction_dir, prepared, output, config):
    started = perf_counter()
    output = evaluation_output(config, prepared, prediction_dir, output)
    records = read_json(Path(prediction_dir) / "predictions.json")["cases"]
    selection = [(row, records[row["case_id"]]) for row in prepared["manifest"]["cases"] if row["case_id"] in records]
    if not selection:
        raise ValueError("没有与当前评价清单匹配的预测")
    for index, (row, receipt) in enumerate(selection, 1):
        if receipt["data_identity"] not in (case_data_identity(prepared, row),
                case_data_identity(prepared, row, include_generation=True)) or \
                receipt.get("reference_sha256") not in (None, row.get('sha256', {}).get('label')):
            raise ValueError("预测与评价的数据、类别或标签版本不同")
    identity = digest({"cases": [{"data": receipt['data_identity'], "label": row.get('label'),
                                 "prediction": receipt.get('identity', receipt['file']),
                                 "checkpoint": receipt.get('checkpoint', receipt.get('checkpoint_file',receipt.get('checkpoint_sha256')))}
                                for row, receipt in selection], "metric": "supervised-dice-v2"})
    if output.exists():
        previous = read_json(output)
        if previous.get("identity") != identity:
            raise FileExistsError("已有评价的输入或模型不同，请使用新的结果目录")
        event(config["run_dir"], "evaluation_reused", cases=len(selection), output=str(output),
              seconds=perf_counter() - started)
        return {**previous, "reused": True}
    results = []
    classes = len(prepared["dataset_json"]["labels"])
    for row in prepared["manifest"]["cases"]:
        receipt = records.get(row["case_id"])
        if receipt is None:
            continue
        mark = perf_counter()
        event(config["run_dir"], "evaluation_case_started", case_id=row["case_id"],
              completed=len(results), total=len(selection))
        path = Path(prediction_dir) / receipt["file"]
        roi = tuple(slice(begin, end) for begin, end in receipt["roi_xyz"])
        by_class = count_labels(path,row['label'],roi,classes,row['unknown_classes'],
                               row.get('ignore_label_ids',[]),config)
        results.append({"case_id": row["case_id"], "scope": receipt["scope"], "roi_xyz": receipt["roi_xyz"], "classes": by_class})
        event(config["run_dir"], "evaluation_case_completed", case_id=row["case_id"], completed=len(results),
              total=len(selection), seconds=perf_counter() - mark, supervised_classes=list(by_class),
              skipped_unknown_classes=row["unknown_classes"])
    aggregates = {}
    for c in range(1, classes):
        values = [row["classes"][str(c)]["dice"] for row in results
                  if str(c) in row["classes"] and row["classes"][str(c)]["dice"] is not None]
        aggregates[str(c)] = {"cases": len(values), "mean_dice": float(np.mean(values)) if values else None}
    summary = {"identity": identity, "scope": prepared["scope"], "labels": prepared["dataset_json"]["labels"],
               "cases": results, "aggregate": aggregates,
               "reused": False, "interpretation": "技术子区域运行结果，不证明泛化效果" if prepared["scope"] == "technical" else "按有监督类别评价"}
    write_json(output, summary)
    event(config["run_dir"], "evaluation_completed", cases=len(results), aggregate=aggregates,
          seconds=perf_counter() - started, output=str(output))
    return summary
