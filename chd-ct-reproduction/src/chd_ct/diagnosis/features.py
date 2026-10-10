"""Named voxel-grid descriptors from predictions only; no physical calibration implied."""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..imagechd.common import LABELS, read_prepared, safe_child
from .io import case_index, file_hash, new_output, read_artifact, save

PAIRS = (("LV", "RV"), ("LA", "RA"), ("LV", "AO"), ("RV", "AO"), ("LV", "PA"), ("RV", "PA"), ("AO", "PA"))
ATTRS = ("present", "fraction", "components", "largest_fraction", "extent_x", "extent_y", "extent_z")
FEATURE_NAMES = tuple(f"{a}_{n}" for n in LABELS[1:] for a in ATTRS) + tuple(
    f"contact_{a}_{b}" for a, b in PAIRS
)
FEATURE_SCHEMA = "imagechd7-voxel-anatomy-v1"
RULE_KEYS = tuple(
    f"conn_{a}_{b}"
    for a, b in (*PAIRS, ("LV", "RA"), ("RV", "LA"), ("PV", "LA"), ("PV", "RA"), ("PV", "SVC"), ("PV", "IVC"))
) + (
    "conn_ao_pa",
    "n_island_AO",
    "n_island_SVC",
    "n_island_SVC_to_RA",
    "n_island_SVC_to_LA",
    "single_ventricle_confirmed",
    "single_atrium_confirmed",
    "ao_override_fraction",
    "stenosis_PA_trunk",
    "common_arterial_trunk_confirmed",
    "pa_atresia_confirmed",
    "stenosis_AO_arch",
    "stenosis_descending_AO",
    "posi_AO_to_spine_arch",
    "posi_AO_to_spine_middle",
    "posi_AO_to_spine_low",
    "stenosis_PA_valve_upper",
    "stenosis_PA_valve",
    "stenosis_PA_valve_down",
    "aortic_arch_interruption_confirmed",
)


def measure(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or min(mask.shape) < 2 or not np.isin(mask, range(8)).all():
        raise ValueError("需要三维七结构预测标签（0—7）")
    foreground = np.count_nonzero(mask)
    if not foreground:
        raise ValueError("分割没有前景，不能生成诊断特征")
    values, present, warnings = {}, {}, []
    structure = ndi.generate_binary_structure(3, 1)
    # Process masks individually to bound memory on native CT grids.
    for label, name in enumerate(LABELS[1:], 1):
        binary = mask == label
        count = int(binary.sum())
        present[name] = bool(count)
        values[f"present_{name}"] = int(bool(count))
        values[f"fraction_{name}"] = count / foreground
        cc, n = ndi.label(binary, structure)
        values[f"components_{name}"] = int(n) if count else None
        values[f"largest_fraction_{name}"] = (
            float(np.bincount(cc.ravel())[1:].max() / count) if count else None
        )
        for axis, key in enumerate("xyz"):
            support = np.flatnonzero(binary.any(axis=tuple(i for i in range(3) if i != axis)))
            values[f"extent_{key}_{name}"] = (
                float((support[-1] - support[0] + 1) / mask.shape[axis]) if count else None
            )
        if not count:
            warnings.append("missing_predicted_" + name)
    for a, b in PAIRS:
        a_id, b_id = LABELS.index(a), LABELS.index(b)
        values[f"contact_{a}_{b}"] = (
            int(bool((ndi.binary_dilation(mask == a_id, structure) & (mask == b_id)).any()))
            if present[a] and present[b]
            else None
        )
    return {"values": values, "rule_features": dict.fromkeys(RULE_KEYS), "warnings": warnings}


def extract(prepared, predictions, output):
    cache, manifest = read_prepared(prepared)
    directory = Path(predictions).expanduser().resolve()
    report_path = directory / "prediction-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("prepared_sha256") != file_hash(cache / "dataset.json"):
        raise ValueError("预测未完成或与预处理清单不匹配")
    predicted = case_index(report.get("cases", []))
    available = case_index(manifest["cases"])
    if set(predicted) - set(available):
        raise ValueError("预测包含清单外病例")
    result = {
        "format": "chd-diagnosis-features-v1",
        "status": "passed",
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "prepared_sha256": report["prepared_sha256"],
        "segmentation_sha256": report["models_sha256"],
        "segmentation_mode": report.get("model_mode", "unverified"),
        "segmentation_prepared_sha256": report.get("segmentation_prepared_sha256"),
        "prediction_sha256": file_hash(report_path),
        "cases": [],
        "measurement_note": "Grid-normalized descriptors; one-voxel contact is a proxy, not proven anatomical communication. Clinical rule features require separately reviewed evidence.",
    }
    for case, pred in predicted.items():
        row = available[case]
        path = safe_child(directory, pred["grid_prediction"])
        with np.load(path, allow_pickle=False) as record:
            mask = record["prediction"]
        if list(mask.shape) != row["cache_shape"]:
            raise ValueError("预测网格形状与清单不匹配")
        result["cases"].append(
            {
                "case_id": case,
                "patient_id": row.get("patient_id", case),
                "split": row.get("split"),
                "mask_sha256": file_hash(path),
                **measure(mask),
            }
        )
    case_index(result["cases"], splits=True)
    output = new_output(output)
    save(output / "features.json", result)
    return result


def read_features(path):
    path, data = read_artifact(path, "features.json", "chd-diagnosis-features-v1")
    if data.get("feature_schema") != FEATURE_SCHEMA or data.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("解剖特征协议不匹配")
    if not data.get("segmentation_sha256") or not data.get("prepared_sha256"):
        raise ValueError("特征缺少来源记录")
    for row in case_index(data["cases"], splits=True).values():
        if set(row["values"]) != set(FEATURE_NAMES) or any(
            v is not None and (not isinstance(v, (int, float)) or not np.isfinite(v))
            for v in row["values"].values()
        ):
            raise ValueError("特征缺失或数值无效")
    return path, data
