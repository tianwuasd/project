import csv
from copy import deepcopy
from pathlib import Path


TASKS = ("aaoca_present", "origin_left", "high_risk")
COLUMNS = ("patient_key", *TASKS, "reviewer", "reference", "review_status", "ct_matched")


def _text(value):
    return "" if value is None else str(value).strip()


def _table(headers, values):
    headers = [_text(value) for value in headers]
    if len(headers) != len(set(headers)) or not set(COLUMNS).issubset(headers):
        raise ValueError("诊断表需要唯一的完整canonical列名")
    result = []
    for values_row in values:
        if not any(_text(value) for value in values_row):
            continue
        if len(values_row) != len(headers):
            raise ValueError("诊断表行宽与列名数量不一致")
        result.append(dict(zip(headers, values_row)))
    return result


def read_label_rows(path, sheet=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        if sheet is not None:
            raise ValueError("CSV不支持工作表参数")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            return _table(next(reader), reader)
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("诊断表仅支持CSV、XLSX或XLSM")
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=False)
    if sheet is None and len(workbook.sheetnames) != 1:
        workbook.close()
        raise ValueError("多工作表诊断文件必须明确指定diagnosis_sheet")
    try:
        worksheet = workbook[sheet] if sheet is not None else workbook.active
        values = worksheet.iter_rows(values_only=True)
        return _table(next(values), values)
    finally:
        workbook.close()


def validate_manifest(manifest):
    cases = manifest["cases"]
    patients, aliases, case_ids = set(), {}, set()
    for case in cases:
        patient = _text(case["patient_key"])
        if not patient or patient in patients or case["case_id"] in case_ids:
            raise ValueError("冻结清单必须患者唯一且病例标识唯一")
        patients.add(patient)
        case_ids.add(case["case_id"])
        if case["role"] not in {"candidate_train", "candidate_validation", "predict"}:
            raise ValueError("冻结清单含未知角色")
        if case["dataset"] not in {"aaoca", "imagechd"}:
            raise ValueError("冻结清单含未知数据集")
        if case["dataset"] == "imagechd" and case["role"] != "predict":
            raise ValueError("ImageCHD仅允许预测角色")
        for alias in [patient, *case["aliases"]]:
            alias = _text(alias)
            if not alias or (alias in aliases and aliases[alias] != patient):
                raise ValueError("冻结清单存在空别名或患者别名冲突")
            aliases[alias] = patient
    return cases


def reviewed_labels(rows, manifest):
    eligible = {case["patient_key"] for case in validate_manifest(manifest) if case["dataset"] == "aaoca"}
    result = {}
    for row in rows:
        patient = _text(row["patient_key"])
        if patient not in eligible:
            continue
        if patient in result:
            raise ValueError("诊断表存在重复患者标识")
        status = _text(row["review_status"])
        matched = _text(row["ct_matched"]).lower() in {"true", "1"}
        reviewer, reference = _text(row["reviewer"]), _text(row["reference"])
        labels = dict.fromkeys(TASKS)
        if status == "confirmed" and matched:
            if not reviewer or not reference:
                raise ValueError("确认标签必须提供reviewer及reference")
            for task in TASKS:
                value = _text(row[task])
                if value and value not in {"0", "1"}:
                    raise ValueError("确认标签仅允许空白、0或1")
                labels[task] = int(value) if value else None
            if labels["aaoca_present"] != 1 and any(labels[task] is not None for task in TASKS[1:]):
                raise ValueError("起源及风险标签仅允许用于已确认AAOCA阳性病例")
        result[patient] = {"labels": labels, "reviewer": reviewer, "reference": reference,
                           "ct_matched": matched, "review_status": status}
    return result


def task_partitions(config, manifest, task):
    if task not in TASKS:
        raise ValueError("未知分类任务")
    if config.get("diagnoses") is None:
        raise ValueError("分类需要明确提供diagnoses诊断表")
    cases = validate_manifest(manifest)
    reviewed = reviewed_labels(read_label_rows(config["diagnoses"], config.get("diagnosis_sheet")), manifest)
    partitions = {"candidate_train": [], "candidate_validation": []}
    for case in cases:
        review = reviewed.get(case["patient_key"])
        if case["dataset"] != "aaoca" or case["role"] not in partitions or review is None:
            continue
        target = review["labels"][task]
        if review["review_status"] == "confirmed" and review["ct_matched"] and target in (0, 1):
            item = deepcopy(case)
            item["target"] = target
            partitions[case["role"]].append(item)
    for partition in partitions.values():
        if {item["target"] for item in partition} != {0, 1}:
            raise ValueError("训练集与验证集必须分别具备已确认的两类标签")
    return partitions["candidate_train"], partitions["candidate_validation"]


def export_confirmed_labels(config, manifest, output):
    source = config.get("diagnoses")
    if source is None:
        raise ValueError("需先指定含明确人工复核字段的canonical CSV或Excel表")
    reviewed = reviewed_labels(read_label_rows(source, config.get("diagnosis_sheet")), manifest)
    output = Path(output).resolve()
    private = (Path(config["project_root"]) / "private").resolve()
    if output == private or not output.is_relative_to(private):
        raise ValueError("确认标签表必须存于本任务private目录")
    if output.exists():
        raise FileExistsError("确认标签输出已经存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for patient, item in reviewed.items():
            writer.writerow({"patient_key": patient, **item["labels"], "reviewer": item["reviewer"],
                             "reference": item["reference"], "review_status": item["review_status"],
                             "ct_matched": str(item["ct_matched"]).lower()})
    return {"path": str(output), "patients": len(reviewed),
            "confirmed_by_task": {task: sum(v["labels"][task] is not None for v in reviewed.values()) for task in TASKS}}
