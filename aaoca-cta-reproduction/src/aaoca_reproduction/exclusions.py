from pathlib import Path

from .records import event, identity, read_json, signature, write_json


SKIP_FILE = "localization-skipped.json"
SKIP_REASONS = {"missing_label_file", "missing_localization_structures", "empty_localization_structures"}


def exclusion_source(case, config):
    return {"source": {key: case.get(key) for key in (
        "case_id", "dataset", "image", "label", "source_stat", "spacing_xyz", "spacing_evidence", "localization")},
        "method": {key: config.get(key) for key in (
            "classification_recipe", "roi_mm", "offset_index_mm", "imagechd_orientation_policy")}}


def exclusion_directory(case, config):
    return Path(config.get("prepared_inputs", {}).get(case["case_id"],
                Path(config["data_root"]) / "aaoca-classification" / case["case_id"])).resolve()


def read_exclusion(case, config, directory=None):
    directory = Path(directory) if directory is not None else exclusion_directory(case, config)
    path = directory / SKIP_FILE
    if not path.exists():
        return None
    record = read_json(path)
    if (record["state"] != "skipped" or record["reason"] not in SKIP_REASONS
            or (config.get("prepared_input_policy") != "compatible" and record["source_identity"] != identity(exclusion_source(case, config)))
            or record["case_id"] != case["case_id"] or (directory / "classification.json").exists()):
        raise ValueError("定位排除记录与当前源、方法或完整结果冲突")
    if config.get("prepared_input_policy") == "compatible":
        return record
    for name in ("image", "label"):
        if name == "label" and record["reason"] == "missing_label_file":
            if Path(case[name]).exists():
                raise ValueError("被跳过的定位文件已出现，需要重新核对来源并预处理该例")
        elif signature(case[name]) != case["source_stat"][name]:
            raise ValueError("定位排除记录的源输入已变化，需要重新核对该例")
    return record


def save_exclusion(case, config, directory, reason, details, preprocessing_id):
    if reason not in SKIP_REASONS:
        raise ValueError("仅允许跳过明确缺失或为空的定位分割")
    record = {"schema": 1, "state": "skipped", "case_id": case["case_id"],
              "reason": reason, "details": details, "identity": preprocessing_id,
              "source_identity": identity(exclusion_source(case, config)), "reused": False,
              "training_updates": 0, "model_prediction": False}
    write_json(Path(directory) / SKIP_FILE, record)
    event("classification_localization_skipped", case_id=case["case_id"], reason=reason, details=details)
    return record


def usable_cases(cases, config):
    usable, skipped = [], []
    for case in cases:
        record = read_exclusion(case, config)
        if record is None:
            usable.append(case)
        else:
            skipped.append({"case_id": case["case_id"], "role": case["role"], "reason": record["reason"]})
    if skipped:
        event("classification_localization_exclusions", skipped=len(skipped), usable=len(usable))
    return usable, skipped
