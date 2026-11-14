import hashlib
import json
from pathlib import Path

import yaml


def apply_split(manifest, config):
    split_config = config["split"]
    path_value = split_config["path"]
    if path_value is None:
        return manifest
    path = Path(path_value).resolve(strict=True)
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(raw.decode("utf-8"))
    elif suffix in (".yaml", ".yml"):
        value = yaml.safe_load(raw.decode("utf-8"))
    else:
        raise ValueError("划分文件必须为 JSON 或 YAML")

    fmt = split_config["format"]
    if fmt == "auto":
        if isinstance(value, list):
            fmt = "folds"
        elif isinstance(value, dict) and value.get("protocol") == "dataset1282-scheme-a-preprocess-1":
            fmt = "chd1282_cohort"
        elif isinstance(value, dict) and "roles" in value and "folds" in value:
            fmt = "imagechd_roles"
        elif isinstance(value, dict) and "aux_train" in value:
            fmt = "imagechd"
        elif isinstance(value, dict) and "development" in value:
            fmt = "roles"
        elif isinstance(value, dict) and "train" in value and "val" in value:
            fmt = "folds"
        else:
            raise ValueError("无法识别划分文件格式")

    fold = config["fold"]
    metadata = {}
    if fmt == "folds":
        if isinstance(value, dict):
            folds = [value]
        elif isinstance(value, list):
            folds = value
        else:
            raise ValueError("折列表格式必须为对象列表")
        if fold >= len(folds):
            raise ValueError("指定折号超出划分文件范围")
        selected = folds[fold]
        roles_map = {"train": "train", "val": "validation", "test": "test", "aux_train": "train"}
    elif fmt == "chd1282_cohort":
        if manifest["dataset"] != "chd1282" or value["protocol"] != "dataset1282-scheme-a-preprocess-1" or fold != 0:
            raise ValueError("1282 Scheme A 冻结清单使用固定 fold 0；三折请显式选择折列表")
        metadata = value["cases"]
        selected = {role: [key for key, record in metadata.items()
                           if record["role"] == role and record.get("included", True) and record.get("index_scan", True)]
                    for role in ("train", "development", "source_test", "target_primary", "target_secondary")}
        roles_map = {"train": "train", "development": "validation", "source_test": "test",
                     "target_primary": "test", "target_secondary": "test"}
    elif fmt == "imagechd_roles":
        selected = {role: [key for key, roles in value["roles"].items() if roles[str(fold)] == role]
                    for role in ("train", "val", "test", "aux_train")}
        roles_map = {"train": "train", "val": "validation", "test": "test", "aux_train": "train"}
    elif fmt == "imagechd":
        selected = value
        roles_map = {"train": "train", "val": "validation", "test": "test", "aux_train": "train"}
        if manifest["dataset"] != "imagechd":
            raise ValueError("aux_train 划分只支持 ImageCHD")
    elif fmt == "roles":
        selected = value
        roles_map = {"train": "train", "development": "validation", "test": "test"}
    else:
        raise ValueError("不支持的划分格式：" + str(fmt))
    if selected.get("aux_train") and manifest["dataset"] != "imagechd":
        raise ValueError("辅助训练角色仅适用于 ImageCHD")

    id_field = split_config["id_field"]
    cases_by_id = {}
    for row in manifest["cases"]:
        if id_field not in row:
            raise ValueError("病例缺少划分映射字段：" + id_field)
        key = str(row[id_field])
        if key in cases_by_id:
            raise ValueError("清单中的划分映射字段重复：" + key)
        cases_by_id[key] = row

    assigned = {}
    source_roles = {}
    selected_ids = set()
    for source_role, role in roles_map.items():
        ids = selected.get(source_role, [])
        if not isinstance(ids, list):
            raise ValueError("划分角色必须对应病例编号列表：" + source_role)
        for value_id in ids:
            case_key = str(value_id)
            if case_key not in cases_by_id:
                raise ValueError("划分中存在清单未收录的病例：" + case_key)
            if case_key in assigned:
                raise ValueError("病例重复出现在多个划分角色：" + case_key)
            assigned[case_key] = role
            source_roles[case_key] = source_role
            selected_ids.add(case_key)

    result = dict(manifest)
    rows = []
    for case_key, row in cases_by_id.items():
        if case_key not in selected_ids:
            continue
        updated = dict(row)
        updated["roles"] = [assigned[case_key]]
        if fmt == "chd1282_cohort":
            source_role = source_roles[case_key]
            updated["source_role"] = source_role
            if assigned[case_key] == "test":
                updated["test_set"] = "internal" if source_role == "source_test" else "external"
                updated["test_group"] = source_role
            if "index_scan" in metadata[case_key]:
                updated["index_scan"] = metadata[case_key]["index_scan"]
        updated["cohort"] = "auxiliary" if case_key in set(map(str, selected.get("aux_train", []))) else "main"
        if metadata:
            patient = metadata[case_key]["patient"]
            if updated.get("patient") not in (None, patient):
                raise ValueError("原清单与冻结划分的患者身份不符")
            updated["patient"] = patient
        rows.append(updated)
    result["cases"] = rows
    if not rows:
        raise ValueError("划分未选择任何病例")
    result["split_source"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                              "fold": fold, "format": fmt, "id_field": id_field,
                              "unit": ("case" if manifest["dataset"] == "imagechd" else "patient")
                              if split_config.get("unit", "auto") == "auto" else split_config["unit"]}
    return result


def select_test(manifest, test_set, group=None, case_ids=None):
    if test_set not in ("internal", "external"):
        raise ValueError("测试入口必须为 internal 或 external")
    groups = {"source_test"} if test_set == "internal" else {"target_primary", "target_secondary"}
    if group is not None and group not in groups:
        raise ValueError("测试子组与所选测试入口不匹配")
    rows = [row for row in manifest["cases"] if "test" in row["roles"] and
            row.get("test_set", "internal") == test_set and (group is None or row.get("test_group") == group)]
    if case_ids is not None:
        requested = set(case_ids)
        if not requested or not requested <= {r["case_id"] for r in rows}:
            raise ValueError("手动选择的病例不属于当前测试组")
        rows = [row for row in rows if row["case_id"] in requested]
    if not rows:
        raise ValueError("当前清单没有所选测试组病例")
    return {**manifest, "cases": rows}
