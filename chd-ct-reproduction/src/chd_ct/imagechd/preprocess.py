"""Raw NIfTI -> reusable caches and split manifest; never trains a model."""

import argparse
import csv
import hashlib
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .common import FORMAT, IGNORE, LABELS, NORMALIZATION, normalize, write_json
from .geometry import resize


def discover(folder, for_prediction=False):
    folder = Path(folder).expanduser().resolve()
    if folder.is_file() and for_prediction and folder.name.endswith((".nii", ".nii.gz")):
        return [(folder.name.removesuffix(".gz").removesuffix(".nii"), folder, None)]
    if (folder / "ImageCHD_dataset").is_dir():
        folder = folder / "ImageCHD_dataset"
    images = sorted(folder.glob("ct_*_image.nii.gz"))
    if not images:
        raise ValueError("请选择已解压的 ImageCHD_dataset 文件夹；预测预处理也可指定单个 NIfTI。")
    pairs = []
    for image in images:
        case = image.name.removesuffix("_image.nii.gz")
        label = image.with_name(case + "_label.nii.gz")
        if not for_prediction and not label.is_file():
            raise ValueError(f"缺少配对标签：{label.name}")
        pairs.append((case, image, None if for_prediction else label))
    if not for_prediction and len(list(folder.glob("ct_*_label.nii.gz"))) != len(pairs):
        raise ValueError("存在没有配对影像的标签")
    return pairs


def map_labels(array):
    if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all() or array.min() < 0:
        raise ValueError("分割标签必须为非负整数")
    present = set(int(x) for x in np.unique(array))
    missing = sorted(set(range(1, 8)) - present)
    extra = sorted(present - set(range(8)))
    # The seven foreground IDs retain the dataset author's definitions.
    target = np.where(array <= 7, array, IGNORE).astype(np.uint8)
    return target, missing, extra


def assign_splits(cases, seed, split_file=None):
    if split_file:
        with Path(split_file).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"case_id", "patient_id", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("划分 CSV 需要 case_id,patient_id,split")
            rows = list(reader)
        mapping, patients = {}, {}
        for row in rows:
            case, patient, split = row["case_id"], row["patient_id"], row["split"]
            if not patient or case in mapping or split not in {"train", "val", "test"}:
                raise ValueError("划分含重复病例、空患者编号或非法 split")
            if patients.setdefault(patient, split) != split:
                raise ValueError("同一患者跨数据划分")
            mapping[case] = row
        if set(cases) - set(mapping):
            raise ValueError("指定划分未覆盖全部保留病例")
        result = {
            case: {"patient_id": mapping[case]["patient_id"], "split": mapping[case]["split"]}
            for case in cases
        }
    else:
        if len(cases) < 3:
            raise ValueError("至少需要 3 例完整标注，才能划分 train/val/test")
        order = sorted(cases)
        random.Random(seed).shuffle(order)
        n = max(1, round(len(order) * 0.15))
        result = {
            case: {"patient_id": case, "split": "test" if i < n else "val" if i < 2 * n else "train"}
            for i, case in enumerate(order)
        }
    if not all(any(r["split"] == split for r in result.values()) for split in ("train", "val", "test")):
        raise ValueError("保留病例必须覆盖 train/val/test 三个非空集合")
    return result


def prepare(source, output, size=0, seed=42, for_prediction=False, split_file=None, limit=None):
    if (size != 0 and size < 8) or (limit is not None and limit < 3):
        raise ValueError("size=0 保留原始网格，或至少为8；测试用 limit 至少为3")
    torch.set_num_threads(2)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": FORMAT,
        "status": "running",
        "size": size,
        "resolution": "resized" if size else "native",
        "seed": seed,
        "for_prediction": for_prediction,
        "normalization": NORMALIZATION,
        "labels": list(LABELS),
        "ignore_index": IGNORE,
        "cases": [],
        "excluded": [],
        "limit": limit,
        "id_policy": "explicit_patient_split" if split_file else "filename_as_patient",
        "geometry": "native header retained; no claim that spacing or intensity units are calibrated",
    }
    write_json(output / "preprocess-report.json", report)
    try:
        pairs = discover(source, for_prediction)
        report["discovered"] = len(pairs)
        hashes = set()
        for index, (case, image_path, label_path) in enumerate(pairs):
            if limit and len(report["cases"]) >= limit:
                break
            print(f"预处理 {index + 1}/{len(pairs)}：{case}", flush=True)
            original = nib.load(image_path)
            if len(original.shape) != 3 or not np.isfinite(original.affine).all():
                raise ValueError(f"{case}：无效的三维影像/affine")
            if abs(np.linalg.det(original.affine[:3, :3])) < 1e-8:
                raise ValueError(f"{case}：affine 不可逆")
            canonical = nib.as_closest_canonical(original)
            targets = {}
            if label_path:
                label = nib.load(label_path)
                if label.shape != original.shape or not np.allclose(label.affine, original.affine, atol=1e-4):
                    raise ValueError(f"{case}：影像和标签网格不同")
                target, missing, extra = map_labels(np.asanyarray(nib.as_closest_canonical(label).dataobj))
                if missing:
                    report["excluded"].append(
                        {"case_id": case, "reason": "missing_foreground_annotation", "missing_ids": missing}
                    )
                    continue
                targets["target"] = (resize(target, (size,) * 3, labels=True) if size else target).astype(
                    np.uint8
                )
            else:
                extra = []
            image, intensity = normalize(canonical.get_fdata(dtype=np.float32))
            # Detect exact duplicate compressed images before assigning patient splits.
            with image_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest in hashes:
                raise ValueError("检测到内容完全重复的影像；请先确认患者身份，避免跨集合泄漏")
            hashes.add(digest)
            cache = f"{case}.npz"
            np.savez_compressed(
                output / cache,
                image=(resize(image, (size,) * 3) if size else image).astype(np.float32),
                **targets,
            )
            row = {
                "case_id": case,
                "cache": cache,
                "cache_shape": [size] * 3 if size else list(image.shape),
                "extra_ids_ignored": extra,
                "intensity": intensity,
                "source_sha256": digest,
                "source_name": image_path.name,
                "native_shape": list(original.shape),
                "native_affine": original.affine.tolist(),
                "native_orientation": nib.orientations.io_orientation(original.affine).tolist(),
                "canonical_shape": list(canonical.shape),
                "canonical_affine": canonical.affine.tolist(),
                "units": list(original.header.get_xyzt_units()),
            }
            report["cases"].append(row)
        if not report["cases"]:
            raise ValueError("没有可用病例")
        if not for_prediction:
            splits = assign_splits([r["case_id"] for r in report["cases"]], seed, split_file)
            for row in report["cases"]:
                row.update(splits[row["case_id"]])
        report["status"] = "prepared"
        write_json(output / "dataset.json", report)  # Only a completed preparation is consumable.
        print(f"已准备 {len(report['cases'])} 例；排除 {len(report['excluded'])} 例：{output}")
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "preprocess-report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅预处理 ImageCHD；生成可复用缓存，不训练、不预测")
    parser.add_argument("--input", required=True, help="已解压的 ImageCHD 目录；预测模式也支持单个 NIfTI")
    parser.add_argument("--output", required=True, help="新的预处理结果目录")
    parser.add_argument("--size", type=int, default=0, help="0 保留原始分辨率；正数仅用于生成缩小缓存")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", help="显式患者划分 CSV；case_id 示例 ct_1001")
    parser.add_argument("--for-prediction", action="store_true", help="无标签预处理，不生成训练划分")
    parser.add_argument("--limit", type=int, help="只用于软件试跑，保留前 N 个合格病例")
    args = parser.parse_args(argv)
    prepare(args.input, args.output, args.size, args.seed, args.for_prediction, args.split_file, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
