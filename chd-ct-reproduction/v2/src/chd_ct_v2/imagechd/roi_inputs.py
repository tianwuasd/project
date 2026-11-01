"""Image-only localizer ROIs, bound to this fold's frozen localization weights."""

from pathlib import Path

import numpy as np

from ..recovery import file_hash, read_json, write_json
from .common import load_case
from .geometry import bounding_box, resize

ROI_METHOD = "fold-localizer-predicted-roi-v1"


def locate(image, directory, collection, device):
    from .predict import anatomy

    crops = [
        resize(anatomy(image, stage, directory, collection, device), image.shape, labels=True).astype(
            np.uint8
        )
        for stage in ("crop64", "crop128")
    ]
    foreground = (crops[0] > 0) | (crops[1] > 0)
    roi = bounding_box(foreground, collection["config"]["roi_margin"])
    full = all(s.start == 0 and s.stop == n for s, n in zip(roi, image.shape))
    return (
        crops,
        roi,
        {
            "roi_method": ROI_METHOD,
            "roi_slices": [[s.start, s.stop] for s in roi],
            "empty_roi_fallback": not bool(foreground.any()),
            "full_volume_roi": full,
            "roi_volume_fraction": float(np.prod([s.stop - s.start for s in roi]) / image.size),
            "roi_warning": "定位框覆盖全图；未使用真值强行缩框" if full else None,
        },
    )


def row_roi(row, shape):
    box = row.get("predicted_roi")
    if (
        not isinstance(box, list)
        or len(box) != 3
        or any(not isinstance(p, list) or len(p) != 2 for p in box)
        or any(type(v) is not int for p in box for v in p)
        or any(not 0 <= p[0] < p[1] <= n for p, n in zip(box, shape))
    ):
        raise ValueError("缺少有效预测 ROI；先完成本折定位与 ROI 阶段，禁止退回真值框")
    return tuple(slice(*p) for p in box)


def roi_contract(directory, rows, config, output):
    return {
        "method": ROI_METHOD,
        "prepared_sha256": file_hash(Path(directory) / "dataset.json"),
        "localizers": {s: file_hash(Path(output) / (s + ".pt")) for s in ("crop64", "crop128")},
        "roi_margin": config["roi_margin"],
        "fold": config.get("fold"),
        "cases": {
            part: {r["case_id"]: file_hash(Path(directory) / r["cache"]) for r in selected}
            for part, selected in rows.items()
        },
    }


def prepare_rois(directory, rows, config, output, resume=False):
    from .gpu_memory import gpu_budget

    output = Path(output)
    path = output / "roi.json"
    contract = roi_contract(directory, rows, config, output)
    record = read_json(path) if resume and path.is_file() else {"contract": contract, "cases": {}}
    if record["contract"] != contract:
        raise ValueError("预测 ROI 的定位权重/数据/患者划分已改变，不能续用")
    collection = read_json(output / "models.json")
    ref = collection.get("roi")
    if resume and ref:
        if ref.get("file") != path.name or file_hash(path) != ref.get("sha256"):
            raise ValueError("已完成 ROI 清单改变")
    else:
        # No durable completion receipt: regenerate, do not bless partial bytes.
        record = {"contract": contract, "cases": {}}
    with gpu_budget(config["device"], output / "roi.gpu-memory.json", config) as guard:
        for part, selected in rows.items():
            for row in selected:
                case = row["case_id"]
                if case in record["cases"]:
                    row_roi({"predicted_roi": record["cases"][case]["roi_slices"]}, row["cache_shape"])
                    continue
                guard.check()
                image, _ = load_case(Path(directory), row, labels=False)
                _, _, details = locate(image, output, collection, config["device"])
                record["cases"][case] = {**details, "split": part, "shape": list(image.shape)}
                write_json(path, record)
                print(
                    f"[roi] {part} {len(record['cases'])}: {case}; {details['roi_warning'] or 'ready'}",
                    flush=True,
                )
    return {
        "file": path.name,
        "sha256": file_hash(path),
        "method": ROI_METHOD,
        "training_source": "in_sample_localizer",
        "test_cases_used": 0,
    }


def attach_rois(directory, rows, config, output):
    output = Path(output)
    collection = read_json(output / "models.json")
    ref = collection.get("roi", {})
    path = output / "roi.json"
    if ref.get("file") != path.name or not path.is_file() or file_hash(path) != ref.get("sha256"):
        raise ValueError("本折预测 ROI 清单缺失或被修改")
    record = read_json(path)
    if record.get("contract") != roi_contract(directory, rows, config, output):
        raise ValueError("预测 ROI 不属于当前定位权重/训练验证划分")
    expected = {r["case_id"] for selected in rows.values() for r in selected}
    if set(record.get("cases", {})) != expected:
        raise ValueError("ROI 病例集合不一致")
    result = {}
    for part, selected in rows.items():
        result[part] = []
        for row in selected:
            info = record["cases"][row["case_id"]]
            if info["split"] != part or info["shape"] != row["cache_shape"]:
                raise ValueError("ROI 划分或网格不一致")
            bound = {**row, "predicted_roi": info["roi_slices"], "roi_sha256": ref["sha256"]}
            row_roi(bound, row["cache_shape"])
            result[part].append(bound)
    return result, ref["sha256"]
