"""Import diagnosis truth; explicitly record user-selected blank semantics."""

import csv
import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from .io import DISEASES, case_index, file_hash, new_output, read_artifact, save

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def xlsx_rows(path):
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(x.itertext())
                for x in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("m:sheets/m:sheet", NS)
        key = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in relationships if r.attrib["Id"] == key)
        name = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        root = ET.fromstring(archive.read(name))
        rows = []
        for row in root.findall("m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                if cell.find("m:f", NS) is not None:
                    raise ValueError("诊断表不接受公式单元格；请先导出已核实数值的 CSV")
                value = cell.findtext("m:v", default="", namespaces=NS)
                if cell.attrib.get("t") == "s" and value:
                    value = strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(cell.find("m:is", NS).itertext())
                values[col] = value
            rows.append(values)
        if not rows:
            raise ValueError("诊断工作表为空")
        headers = {col: str(value).strip() for col, value in rows[0].items()}
        if len(set(headers.values())) != len(headers):
            raise ValueError("诊断表存在重复列名")
        return [
            {header: row.get(col, "") for col, header in headers.items()}
            for row in rows[1:]
            if any(row.values())
        ]


def import_labels(source, output, blank_policy="negative"):
    source = Path(source).expanduser().resolve()
    if blank_policy not in {"negative", "unknown"}:
        raise ValueError("blank_policy 只能是 negative 或 unknown")
    if source.suffix.lower() == ".xlsx":
        rows = xlsx_rows(source)
    elif source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if len(set(reader.fieldnames or [])) != len(reader.fieldnames or []):
                raise ValueError("诊断表存在重复列名")
            rows = list(reader)
    else:
        raise ValueError("诊断标签支持 .xlsx 或 .csv")
    if not rows or not (set(rows[0]) & set(DISEASES)):
        raise ValueError("未找到 ImageCHD 疾病列")
    result = {
        "format": "chd-diagnosis-labels-v1",
        "status": "passed",
        "diseases": list(DISEASES),
        "blank_policy": blank_policy,
        "source_sha256": file_hash(source),
        "source_name": source.name,
        "id_policy": "case_id" if "case_id" in rows[0] else "ImageCHD index + 1000",
        "cases": [],
    }
    for row in rows:
        if "case_id" in row:
            case = row["case_id"].strip()
        elif "index" in row:
            try:
                index = float(row["index"])
                if not index.is_integer() or index < 0:
                    raise ValueError()
                case = "ct_" + str(1000 + int(index))
            except (TypeError, ValueError) as error:
                raise ValueError("ImageCHD index 必须是非负整数") from error
        else:
            raise ValueError("需要 case_id 或 ImageCHD 原始 index 列")
        labels, assumed = {}, []
        for disease in DISEASES:
            if disease not in row:
                labels[disease] = None  # A missing column is not a blank cell.
                continue
            text = str(row[disease] or "").strip()
            if text in {"1", "1.0"}:
                labels[disease] = 1
            elif text in {"0", "0.0"}:
                labels[disease] = 0
            elif not text:
                labels[disease] = 0 if blank_policy == "negative" else None
                if blank_policy == "negative":
                    assumed.append(disease)
            elif text.lower() in {"unknown", "?", "na", "nan"}:
                labels[disease] = None
            else:
                raise ValueError(f"{case}/{disease} 标签必须为 0、1、空白或 unknown")
        result["cases"].append({"case_id": case, "labels": labels, "assumed_negative": assumed})
    case_index(result["cases"])
    result["counts"] = {
        d: {
            "positive": sum(r["labels"][d] == 1 for r in result["cases"]),
            "negative": sum(r["labels"][d] == 0 for r in result["cases"]),
            "unknown": sum(r["labels"][d] is None for r in result["cases"]),
            "assumed_negative": sum(d in r["assumed_negative"] for r in result["cases"]),
        }
        for d in DISEASES
    }
    result["positive_cases"] = sum(any(v == 1 for v in r["labels"].values()) for r in result["cases"])
    output = new_output(output)
    save(output / "labels.json", result)
    with (output / "labels-review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", *DISEASES])
        writer.writeheader()
        for row in result["cases"]:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    **{d: "unknown" if v is None else v for d, v in row["labels"].items()},
                }
            )
    return result


def read_labels(path):
    path, data = read_artifact(path, "labels.json", "chd-diagnosis-labels-v1")
    if data.get("diseases") != list(DISEASES) or data.get("blank_policy") not in {"negative", "unknown"}:
        raise ValueError("诊断标签协议不匹配")
    for row in case_index(data["cases"]).values():
        if set(row["labels"]) != set(DISEASES) or any(
            v is not None and (type(v) is not int or v not in {0, 1}) for v in row["labels"].values()
        ):
            raise ValueError("无效诊断标签")
    return path, data
