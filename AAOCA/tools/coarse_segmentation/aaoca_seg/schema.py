"""Public label schema and private input manifest handling."""
import csv
import hashlib
import json
from pathlib import Path
import re

# id, upstream class, Chinese name, task. These are project IDs, not upstream IDs.
LABELS = (
    (1, "ventricle_left", "左心室", "heartchambers_highres"),
    (2, "ventricle_right", "右心室", "heartchambers_highres"),
    (3, "atrium_left", "左心房", "heartchambers_highres"),
    (4, "atrium_right", "右心房", "heartchambers_highres"),
    (5, "myocardium", "心肌", "heartchambers_highres"),
    (6, "aorta", "主动脉", "heartchambers_highres"),
    (7, "pulmonary_artery", "肺动脉", "heartchambers_highres"),
    (8, "pulmonary_vein", "肺静脉", "total"),
    (9, "superior_vena_cava", "上腔静脉", "total"),
    (10, "inferior_vena_cava", "下腔静脉", "total"),
    (11, "coronary_arteries", "冠状动脉", "coronary_arteries"),
)
TASKS = ("total", "heartchambers_highres", "coronary_arteries")
FIELDS = ("case_id", "image", "spacing_confirmed", "hu_confirmed")
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def read_manifest(path, case_ids=None, limit=None):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("清单需要 case_id,image,spacing_confirmed,hu_confirmed 四列")
        rows = list(reader)
    seen, images, result = set(), set(), []
    for row in rows:
        key = row["case_id"].strip()
        if not CASE_ID.fullmatch(key) or key.upper().split(".")[0] in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }:
            raise ValueError(f"非法研究代号: {key!r}")
        if key.casefold() in seen:
            raise ValueError(f"重复研究代号: {key}")
        seen.add(key.casefold())
        image = Path(row["image"].strip())
        image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
        if str(image).casefold() in images:
            raise ValueError(f"重复影像: {key}")
        images.add(str(image).casefold())
        if any(row[f].strip().lower() != "true" for f in FIELDS[2:]):
            raise ValueError(f"{key}: 必须根据来源确认 HU 和真实间距，不能只看头信息")
        result.append({"case_id": key, "image": str(image)})
    if case_ids is not None:
        wanted = set(case_ids)
        missing = wanted - {r["case_id"] for r in result}
        if not wanted or missing:
            raise ValueError(f"病例选择为空或不存在: {sorted(missing)}")
        result = [r for r in result if r["case_id"] in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit 必须大于零")
        result = result[:limit]
    if not result:
        raise ValueError("清单没有可用的 AAOCA 体积")
    return result


def export_inventory(inventory, manifest, data_root=None):
    """Export registered AAOCA native volumes; never recursively discover scans."""
    inventory = Path(inventory).resolve()
    records = json.loads(inventory.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("研究总清单应为 JSON 数组")
    rows = []
    for record in records:
        if record.get("type") != "AAOCA":
            continue
        if (record.get("copy_status") != "copied_verified"
                or record.get("intensity_units_from_conversion") != "HU"
                or not record.get("units") or record["units"][0] != "mm"
                or record.get("header_issues")):
            raise ValueError("AAOCA 清单含来源或空间信息未确认的记录，请先核查")
        original = record["destination"].replace("\\", "/")
        if data_root:
            marker = "/AAOCA/"
            if marker not in original:
                raise ValueError("重定位需要原路径包含 /AAOCA/")
            root = Path(data_root).resolve()
            image = (root / "AAOCA" / original.split(marker, 1)[1]).resolve()
            if not image.is_relative_to(root / "AAOCA"):
                raise ValueError("重定位路径越界")
        else:
            image = Path(original)
            if not image.is_absolute():
                image = inventory.parent / image
        rows.append({"case_id": "aaoca_" + hashlib.sha256(original.encode()).hexdigest()[:16],
                     "image": str(image.resolve()), "spacing_confirmed": "true", "hu_confirmed": "true"})
    if not rows:
        raise ValueError("没有已核验的 AAOCA 记录")
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
