from pathlib import Path

import numpy as np
import yaml


LABELS = {
    "chd1282": ["background", "LV", "RV", "LA", "RA", "AO", "PA", "Myo", "SVC", "IVC", "PV"],
    "imagechd": ["background", "LV", "RV", "LA", "RA", "Myo", "AO", "PA"],
}


def load_manifest(path, dataset, scope):
    path = Path(path).resolve(strict=True)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value["dataset"] != dataset or value["scope"] != scope:
        raise ValueError("数据清单与数据集或任务范围不匹配")
    root = Path(value["data_root"]).resolve(strict=True)
    records = value["cases"]
    if not records or len({row["case_id"] for row in records}) != len(records):
        raise ValueError("病例清单为空或病例编号重复")
    for row in records:
        if not row["case_id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("病例编号包含不允许的字符")
        if not set(row["roles"]) <= {"train", "validation", "test", "predict"} or not row["roles"]:
            raise ValueError("病例角色无效")
        for kind in ("image", "label"):
            if row.get(kind) is None:
                continue
            source = Path(row[kind])
            source = source.resolve() if source.is_absolute() else (root / source).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("病例文件不在数据根目录内或不存在")
            row[kind] = str(source)
            expected = row.get("sha256", {}).get(kind)
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError("清单必须提供已审计来源的 SHA256：" + kind)
            stat = source.stat()
            row.setdefault("source_stamp", {})[kind] = [stat.st_size, stat.st_mtime_ns]
        spacing = np.asarray(row["spacing_xyz"], dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("缺少可靠的逐例毫米间距")
        unknown = set(row["unknown_classes"])
        if not unknown <= set(range(1, len(LABELS[dataset]))):
            raise ValueError("未知类别编号无效")
        if dataset == "imagechd" and unknown not in (set(), {5}):
            raise ValueError("ImageCHD 仅接收完整或缺 MYO 的监督协议")
        if set(row.get("known_absent_classes", [])) & unknown:
            raise ValueError("真实缺如与未知类别不能重叠")
        if scope == "formal" and "predict" not in row["roles"] and not row.get("patient"):
            raise ValueError("正式划分缺少患者身份")
        expected = "shared_index_excel" if dataset == "chd1282" else "matching_affine_author_spacing"
        if row["geometry_policy"] != expected:
            raise ValueError("空间处理规则与数据集不匹配")
    return {**value, "data_root": str(root), "cases": records}
