import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path, PurePosixPath


TASKS = ("aaoca_present", "origin_left", "high_risk")
COLUMNS = ("patient_key", *TASKS, "reviewer", "reference", "review_status", "ct_matched")
CERTAINTY_COLUMNS = dict(zip(TASKS, ("aaoca_certainty", "origin_certainty", "risk_certainty")))
POLICIES = ("confirmed", "document_explicit", "research_sources")
TASK_PREFIXES = dict(zip(TASKS, ("aaoca", "origin", "risk")))


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
        row = dict(zip(headers, values_row))
        if _text(row.get("source_type")) and not set(CERTAINTY_COLUMNS.values()).issubset(headers):
            raise ValueError("研究标签表须提供三个任务各自的确定程度")
        result.append(row)
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
    if sheet is None and {"data", "evidence", "definitions"}.issubset(workbook.sheetnames):
        sheet = "data"
    if sheet is None and len(workbook.sheetnames) != 1:
        workbook.close()
        raise ValueError("多工作表诊断文件必须明确指定diagnosis_sheet")
    try:
        worksheet = workbook[sheet] if sheet is not None else workbook.active
        values = worksheet.iter_rows(values_only=True)
        return _table(next(values), values)
    finally:
        workbook.close()


def imagechd_attestation(case):
    value = case.get("label_attestation")
    if value is None:
        return None
    if (case["dataset"] != "imagechd" or value["source_type"] != "user_attestation"
            or value["task"] != "aaoca_present" or value["value"] != 0
            or value["case_id"] != case["case_id"] or value["confirmed_by"] != "user"
            or value["patient_split_authorized"] is not True
            or value["image_bytes"] != case["source_stat"]["image"]["bytes"]
            or not _text(value["reference"]) or not _text(value["statement"]) or not _text(value["confirmed_on"])):
        raise ValueError("ImageCHD用户阴性确认须具备实际病例、输入大小及明确来源")
    return value


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
        if case["dataset"] == "imagechd" and case["role"] != "predict" and imagechd_attestation(case) is None:
            raise ValueError("ImageCHD监督角色须有显式用户阴性确认")
        for alias in [patient, *case["aliases"]]:
            alias = _text(alias)
            if not alias or (alias in aliases and aliases[alias] != patient):
                raise ValueError("冻结清单存在空别名或患者别名冲突")
            aliases[alias] = patient
    return cases


def ct_volume_binding(row, case, aaoca_root, task=None):
    prefix = TASK_PREFIXES[task] if task is not None else None
    task_columns = prefix is not None and (prefix + "_ct_matched" in row or prefix + "_matched_ct_images" in row)
    declared = _text(row[prefix + "_ct_matched"] if task_columns else row["ct_matched"]).lower() in {"true", "1"}
    raw = _text(row[prefix + "_matched_ct_images"] if task_columns else row.get("matched_ct_images"))
    images = json.loads(raw) if raw else []
    if not isinstance(images, list) or any(not isinstance(item, str) or not item for item in images):
        raise ValueError("matched_ct_images须为具体CT相对路径的JSON数组")
    normalized = []
    for item in images:
        path = PurePosixPath(item)
        if path.is_absolute() or ".." in path.parts or ":" in item or "\\" in item:
            raise ValueError("配对CT必须使用AAOCA数据根内的正斜杠相对路径")
        normalized.append(path.as_posix())
    selected = None
    available = set()
    if aaoca_root is not None:
        root = Path(aaoca_root).resolve()
        if case.get("image"):
            selected = Path(case["image"]).resolve().relative_to(root).as_posix()
            available.add(selected)
        for volume in case.get("volumes", []):
            available.add(Path(volume["image"]).resolve().relative_to(root).as_posix())
    reference = _text(row.get("ct_match_reference"))
    if task_columns:
        references = json.loads(reference)
        if not isinstance(references, dict) or prefix not in references:
            raise ValueError("逐任务ct_match_reference须包含aaoca/origin/risk键")
        value = references[prefix]
        if not isinstance(value, list):
            raise ValueError("逐任务CT参考须为JSON数组")
        reference = json.dumps(value, ensure_ascii=False) if value else ""
    legacy_schema = set(COLUMNS).issubset(row) and set(row).issubset({*COLUMNS, "source_type"})
    if legacy_schema and not _text(row.get("source_type")) and declared and _text(row["review_status"]) == "confirmed":
        if not _text(row["reviewer"]) or not _text(row["reference"]):
            raise ValueError("旧confirmed标签必须提供真实reviewer及reference")
        if case.get("image"):
            return {"ct_match_declared": True, "ct_matched": True, "available_ct_matched": True,
                    "selected_ct_image": selected if aaoca_root is not None else str(case["image"]),
                    "matched_ct_images": [selected] if selected is not None else [],
                    "ct_match_reference": "legacy_confirmed_attestation: " + _text(row["reference"]),
                    "ct_binding_state": "legacy_confirmed_attestation"}
    valid = declared and bool(reference) and bool(normalized) and aaoca_root is not None
    current = valid and selected is not None and selected in normalized
    state = ("not_declared_matched" if not declared else "missing_volume_binding" if not normalized
             else "missing_pairing_reference" if not reference else "missing_dataset_root" if aaoca_root is None
             else "no_selected_volume" if selected is None else "selected_volume_matched" if current
             else "selected_volume_not_matched")
    return {"ct_match_declared": declared, "ct_matched": current,
            "available_ct_matched": valid and bool(available.intersection(normalized)),
            "selected_ct_image": selected, "matched_ct_images": normalized,
            "ct_match_reference": reference, "ct_binding_state": state}


def reviewed_labels(rows, manifest, policy="confirmed", registry=None, aaoca_root=None):
    if policy not in POLICIES:
        raise ValueError("标签策略须为confirmed、document_explicit或research_sources")
    aliases = {}
    cases = validate_manifest(manifest) if registry is None else registry
    seen = set()
    for case in cases:
        if case["dataset"] == "aaoca":
            if not _text(case["patient_key"]) or case["patient_key"] in seen:
                raise ValueError("患者登记主编号为空或重复")
            seen.add(case["patient_key"])
            for alias in (case["patient_key"], *case["aliases"]):
                if _text(alias) in aliases and aliases[_text(alias)] != case["patient_key"]:
                    raise ValueError("患者登记别名冲突")
                aliases[_text(alias)] = case["patient_key"]
    result = {}
    patients = {case["patient_key"]: case for case in cases if case["dataset"] == "aaoca"}
    for row in rows:
        patient = aliases.get(_text(row["patient_key"]))
        if patient is None:
            continue
        if patient in result:
            raise ValueError("诊断表存在重复患者标识")
        status = _text(row["review_status"])
        binding = ct_volume_binding(row, patients[patient], aaoca_root)
        bindings = {task: ct_volume_binding(row, patients[patient], aaoca_root, task) for task in TASKS}
        reviewer, reference = _text(row["reviewer"]), _text(row["reference"])
        source_type = _text(row.get("source_type"))
        confirmed_raw = dict.fromkeys(TASKS)
        if status == "confirmed" and any(item["ct_matched"] for item in bindings.values()) and source_type in {"", "clinical_review"}:
            if not reviewer or not reference:
                raise ValueError("确认标签必须提供reviewer及reference")
            for task in TASKS:
                value = _text(row[task])
                if value and value not in {"0", "1"}:
                    raise ValueError("确认标签仅允许空白、0或1")
                confirmed_raw[task] = int(value) if value else None
            if confirmed_raw["aaoca_present"] != 1 and any(confirmed_raw[task] is not None for task in TASKS[1:]):
                raise ValueError("起源及风险标签仅允许用于已确认AAOCA阳性病例")
        research = dict.fromkeys(TASKS)
        certainty = {}
        if source_type:
            if source_type not in {"document_explicit", "clinical_review", "unknown"}:
                raise ValueError("未知标签来源类型")
            if _text(row["ct_matched"]).lower() not in {"true", "false", "1", "0", "unknown", ""}:
                raise ValueError("研究表ct_matched须为true/false/unknown或空白")
            for task, column in CERTAINTY_COLUMNS.items():
                level = _text(row[column])
                if level not in {"explicit", "uncertain", "unknown", "conflict"}:
                    raise ValueError("任务确定程度须为explicit/uncertain/unknown/conflict")
                certainty[task] = level
                value = _text(row[task])
                if value.lower() in {"", "unknown"}:
                    continue
                if value not in {"0", "1"} or level != "explicit" or source_type == "unknown" or not reference:
                    raise ValueError("任务0/1须有明确原文确定程度及可追溯来源")
                if source_type == "document_explicit":
                    research[task] = int(value)
            if research["aaoca_present"] != 1 and any(research[task] is not None for task in TASKS[1:]):
                raise ValueError("起源及风险研究标签仅适用于文书明确AAOCA阳性")
            if any(value is not None for value in research.values()):
                references = json.loads(reference)
                if not isinstance(references, list) or not references or any(
                        not isinstance(item, dict) or not item for item in references):
                    raise ValueError("文书来源reference须为非空JSON对象数组")
            if _text(row.get("coronary_artery")) in {"multiple", "conflict"} and research["origin_left"] is not None:
                raise ValueError("多支或冲突异常冠脉不能强行编码单侧起源")
        selected = research if policy in {"document_explicit", "research_sources"} and source_type == "document_explicit" else confirmed_raw
        labels = {task: selected[task] if bindings[task]["ct_matched"] else None for task in TASKS}
        confirmed = {task: confirmed_raw[task] if bindings[task]["ct_matched"] else None for task in TASKS}
        result[patient] = {"labels": labels, "confirmed_labels": confirmed, "confirmed_raw_labels": confirmed_raw, "research_labels": research,
                           "reviewer": reviewer, "reference": reference,
                           "ct_match_status": row.get("ct_match_status", row["ct_matched"]),
                           "review_status": status, "source_type": source_type, "certainty": certainty,
                           "label_policy": policy, "source_fields": dict(row), "ct_bindings": bindings,
                           "reference_positive_aaoca": selected["aaoca_present"] == 1,
                           "confirmed_positive_aaoca": confirmed_raw["aaoca_present"] == 1, **binding}
    if policy == "research_sources":
        for case in cases:
            if case["dataset"] != "imagechd":
                continue
            attestation = imagechd_attestation(case)
            if attestation is None:
                continue
            labels = {"aaoca_present": 0, "origin_left": None, "high_risk": None}
            reference = json.dumps(attestation, ensure_ascii=False)
            bindings = {task: {"ct_match_declared": task == "aaoca_present", "ct_matched": task == "aaoca_present",
                        "available_ct_matched": task == "aaoca_present", "selected_ct_image": case["image"],
                        "matched_ct_images": [case["image"]] if task == "aaoca_present" else [],
                        "ct_match_reference": reference,
                        "ct_binding_state": "user_attested_frozen_input" if task == "aaoca_present" else "not_applicable"}
                        for task in TASKS}
            result[case["patient_key"]] = {"labels": labels, "research_labels": labels,
                "confirmed_labels": dict.fromkeys(TASKS), "confirmed_raw_labels": dict.fromkeys(TASKS),
                "reviewer": "", "reference": reference, "ct_match_status": "user_attested_dataset_case",
                "review_status": "user_attested", "source_type": "user_attestation",
                "certainty": {"aaoca_present": "explicit", "origin_left": "not_applicable", "high_risk": "not_applicable"},
                "label_policy": policy, "source_fields": deepcopy(attestation), "ct_bindings": bindings,
                "reference_positive_aaoca": False, "confirmed_positive_aaoca": False, **bindings["aaoca_present"]}
    return result


def summarize_registry_labels(rows, registry, manifest, aaoca_root=None):
    selected = {case["patient_key"]: case["image"] for case in validate_manifest(manifest) if case["dataset"] == "aaoca"}
    candidates = [{**patient, **({"image": selected[patient["patient_key"]]} if patient["patient_key"] in selected else {})}
                  for patient in registry]
    reviews = reviewed_labels(rows, manifest, "document_explicit", registry=candidates, aaoca_root=aaoca_root)
    frozen_roles = {case["patient_key"]: case["role"] for case in validate_manifest(manifest)
                    if case["dataset"] == "aaoca"}
    groups = {}
    for patient in registry:
        key = patient["patient_key"]
        role = frozen_roles.get(key, "outside_existing_freeze")
        counts = groups.setdefault(role, {task: {"0": 0, "1": 0, "unknown": 0, "conflict": 0,
                                               "ct_matched_0": 0, "ct_matched_1": 0,
                                               "available_ct_labeled": 0,
                                               "available_ct_0": 0, "available_ct_1": 0,
                                               "ct_unmatched_labeled": 0} for task in TASKS})
        review = reviews.get(key)
        for task in TASKS:
            value = review["research_labels"][task] if review else None
            level = review["certainty"].get(task) if review else None
            counts[task][str(value) if value is not None else "conflict" if level == "conflict" else "unknown"] += 1
            if value is not None:
                task_binding = review["ct_bindings"][task]
                counts[task]["available_ct_labeled"] += int(task_binding["available_ct_matched"])
                counts[task]["available_ct_" + str(value)] += int(task_binding["available_ct_matched"])
                counts[task]["ct_matched_" + str(value) if task_binding["ct_matched"] else "ct_unmatched_labeled"] += 1
    return {"ct_patients": len(registry), "matched_label_patients": len(reviews),
            "ct_binding_states": dict(Counter(review["ct_binding_state"] for review in reviews.values())),
            "ct_binding_states_by_task": {task: dict(Counter(review["ct_bindings"][task]["ct_binding_state"]
                                                             for review in reviews.values())) for task in TASKS},
            "unmatched_registry_patients": len(registry) - len(reviews), "groups": groups,
            "scope": "all registered CT patients; existing freeze unchanged; no new split assigned"}


def task_partitions(config, manifest, task):
    if task not in TASKS:
        raise ValueError("未知分类任务")
    if config.get("diagnoses") is None:
        raise ValueError("分类需要明确提供diagnoses诊断表")
    cases = validate_manifest(manifest)
    policy = config.get("label_policy", "confirmed")
    reviewed = reviewed_labels(read_label_rows(config["diagnoses"], config.get("diagnosis_sheet")), manifest, policy,
                               aaoca_root=config["aaoca_root"])
    partitions = {"candidate_train": [], "candidate_validation": []}
    for case in cases:
        review = reviewed.get(case["patient_key"])
        if case["role"] not in partitions or review is None:
            continue
        target = review["labels"][task]
        if review["ct_bindings"][task]["ct_matched"] and target in (0, 1):
            item = deepcopy(case)
            item["target"] = target
            if policy in {"document_explicit", "research_sources"} or review["ct_binding_state"] == "legacy_confirmed_attestation":
                item["label_provenance"] = {key: review[key] for key in
                    ("label_policy", "source_type", "reference", "certainty", "review_status", "reviewer", "ct_matched", "ct_match_status",
                     "selected_ct_image", "matched_ct_images", "ct_match_reference", "ct_binding_state")}
                item["label_provenance"]["task_ct_binding"] = review["ct_bindings"][task]
            partitions[case["role"]].append(item)
    for partition in partitions.values():
        if {item["target"] for item in partition} != {0, 1}:
            raise ValueError("任务" + task + "按" + policy + "策略的训练集与验证集须各有CT匹配的0/1两类")
    return partitions["candidate_train"], partitions["candidate_validation"]


def export_confirmed_labels(config, manifest, output):
    source = config.get("diagnoses")
    if source is None:
        raise ValueError("需先指定含明确人工复核字段的canonical CSV或Excel表")
    reviewed = reviewed_labels(read_label_rows(source, config.get("diagnosis_sheet")), manifest,
                               aaoca_root=config["aaoca_root"])
    output = Path(output).resolve()
    private = (Path(config["project_root"]) / "private").resolve()
    if output == private or not output.is_relative_to(private):
        raise ValueError("确认标签表必须存于本任务private目录")
    if output.exists():
        raise FileExistsError("确认标签输出已经存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(COLUMNS)
    for item in reviewed.values():
        for field in item["source_fields"]:
            if field not in fields:
                fields.append(field)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for patient, item in reviewed.items():
            writer.writerow({**item["source_fields"], "patient_key": patient, **item["confirmed_raw_labels"]})
    return {"path": str(output), "patients": len(reviewed),
            "confirmed_by_task": {task: sum(v["confirmed_labels"][task] is not None for v in reviewed.values()) for task in TASKS}}
