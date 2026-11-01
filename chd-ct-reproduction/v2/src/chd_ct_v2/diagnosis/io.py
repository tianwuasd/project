"""Versioned JSON artifacts and case-level validation."""

import hashlib
import json
from pathlib import Path

from ..imagechd.common import write_json

DISEASES = (
    "ASD",
    "VSD",
    "AVSD",
    "ToF",
    "TGA",
    "DORV",
    "CAT",
    "CA",
    "AAH",
    "DAA",
    "IAA",
    "PA",
    "APVC",
    "DSVC",
    "PDA",
    "PAS",
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_artifact(path, filename, kind):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= filename
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != kind or data.get("status") != "passed":
        raise ValueError(f"不是完整的 {kind} 结果：{path}")
    return path, data


def new_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def case_index(rows, splits=False):
    result, patients = {}, {}
    for row in rows:
        case = row.get("case_id")
        if not isinstance(case, str) or not case or case in {".", ".."} or any(c in case for c in "/\\:"):
            raise ValueError("病例编号无效")
        if case in result:
            raise ValueError("病例编号重复：" + case)
        if splits:
            patient, split = row.get("patient_id"), row.get("split")
            if not isinstance(patient, str) or not patient or split not in {None, "train", "val", "test"}:
                raise ValueError("患者/划分信息无效")
            if patients.setdefault(patient, split) != split:
                raise ValueError("同一患者跨数据划分")
        result[case] = row
    if not result:
        raise ValueError("病例集合为空")
    return result


def save(path, value):
    write_json(path, value)
