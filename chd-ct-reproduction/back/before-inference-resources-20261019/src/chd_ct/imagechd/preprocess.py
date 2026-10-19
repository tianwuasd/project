"""Raw NIfTI -> reusable caches and split manifest; never trains a model."""

import argparse
import csv
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..recovery import atomic_path, file_hash, output_lock, read_json, write_json
from .cache_reuse import CacheReuseError, changed_fields, inspect_sources, read_cache
from .common import FORMAT, IGNORE, LABELS, NORMALIZATION, normalize
from .geometry import resize
from .spacing import (
    RESAMPLED_FORMAT,
    find_spacing_excel,
    read_spacing_excel,
    resampling_grid,
    sample_grid,
    validate_spacing,
)


def discover(folder, for_prediction=False):
    folder = Path(folder).expanduser().resolve()
    if folder.is_file() and for_prediction and folder.name.endswith((".nii", ".nii.gz")):
        case = folder.name.removesuffix(".gz").removesuffix(".nii").removesuffix("_image")
        return [(case, folder, None)]
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


def prepare_case(case, image_path, label_path, size, spacing_record=None, target_spacing=(1, 1, 1)):
    print("  读取影像与标签：" + case, flush=True)
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
            return None, {"case_id": case, "reason": "missing_foreground_annotation", "missing_ids": missing}
        targets["target"] = target
    else:
        extra = []
    print("  归一化：" + case, flush=True)
    image, intensity = normalize(canonical.get_fdata(dtype=np.float32))
    resampling = None
    if spacing_record:
        native_spacing = validate_spacing(spacing_record["spacing_mm"])
        # Excel axes describe the supplied array, before canonical reorientation.
        order = np.argsort(nib.orientations.io_orientation(original.affine)[:, 0].astype(int))
        source_spacing = native_spacing[order]
        shape, transform = resampling_grid(image.shape, source_spacing, target_spacing)
        print(f"  按 Excel 重采样：{case} -> {shape}，间距 {list(target_spacing)} mm", flush=True)
        image = sample_grid(image, shape, transform)
        if targets:
            targets["target"] = sample_grid(targets["target"], shape, transform, labels=True)
        resampling = {
            "source": "imagechd_excel",
            "excel_row": spacing_record["excel_row"],
            "native_spacing_mm": native_spacing.tolist(),
            "canonical_spacing_mm": source_spacing.tolist(),
            "target_spacing_mm": list(map(float, target_spacing)),
            "resampled_shape": shape,
            "output_to_canonical": transform.tolist(),
            "image_interpolation": "linear",
            "label_interpolation": "nearest",
            "grid_alignment": "volume_center",
        }
    if size and targets:
        targets["target"] = resize(targets["target"], (size,) * 3, labels=True).astype(np.uint8)
    cache = f"{case}.npz"
    arrays = dict(image=(resize(image, (size,) * 3) if size else image).astype(np.float32), **targets)
    row = {
        "case_id": case,
        "cache": cache,
        "cache_shape": [size] * 3 if size else list(image.shape),
        "extra_ids_ignored": extra,
        "intensity": intensity,
        "source_sha256": file_hash(image_path),
        "source_name": image_path.name,
        "native_shape": list(original.shape),
        "native_affine": original.affine.tolist(),
        "native_orientation": nib.orientations.io_orientation(original.affine).tolist(),
        "canonical_shape": list(canonical.shape),
        "canonical_affine": canonical.affine.tolist(),
        "units": list(original.header.get_xyzt_units()),
    }
    if resampling:
        row["resampling"] = resampling
    return arrays, row


def prepare(
    source,
    output,
    size=0,
    seed=42,
    for_prediction=False,
    split_file=None,
    limit=None,
    resume=False,
    spacing_excel=None,
    target_spacing=(1, 1, 1),
    no_resample=False,
):
    if (size != 0 and size < 8) or (limit is not None and limit < 3):
        raise ValueError("size=0 保留原始网格，或至少为8；测试用 limit 至少为3")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output):
        return prepare_locked(
            source,
            output,
            size,
            seed,
            for_prediction,
            split_file,
            limit,
            resume,
            spacing_excel,
            target_spacing,
            no_resample,
        )


def prepare_locked(
    source,
    output,
    size,
    seed,
    for_prediction,
    split_file,
    limit,
    resume,
    spacing_excel=None,
    target_spacing=(1, 1, 1),
    no_resample=False,
):
    torch.set_num_threads(2)
    options = dict(
        format=FORMAT,
        size=size,
        seed=seed,
        for_prediction=for_prediction,
        normalization=NORMALIZATION,
        labels=list(LABELS),
        ignore_index=IGNORE,
        limit=limit,
        id_policy="explicit_patient_split" if split_file else "filename_as_patient",
    )
    pairs = discover(source, for_prediction)
    excel = find_spacing_excel(pairs[0][1], spacing_excel, no_resample)
    spacings = read_spacing_excel(excel) if excel else {}
    if excel:
        target_spacing = validate_spacing(target_spacing).tolist()
        missing = sorted({case for case, _, _ in pairs} - set(spacings))
        if missing:
            raise ValueError("Excel 缺少病例间距：" + ", ".join(missing))
        options.update(
            format=RESAMPLED_FORMAT,
            resampling={
                "source": "imagechd_excel",
                "excel_name": excel.name,
                "excel_sha256": file_hash(excel),
                "target_spacing_mm": target_spacing,
                "fields": ["PixelSpacing", "calculate_z_thick"],
                "id_policy": "idx + 1000",
                "algorithm": "center_aligned_linear_nearest_v1",
            },
        )
    contract = dict(
        options=options,
        sources=[[case, str(image), str(label) if label else None] for case, image, label in pairs],
        split_sha256=file_hash(split_file) if split_file else None,
    )
    contract_file = output / "preprocess-contract.json"
    report_file = output / "preprocess-report.json"
    if contract_file.exists():
        differences = changed_fields(read_cache(contract_file), contract)
        if differences:
            raise CacheReuseError("预处理来源/参数改变：" + ", ".join(differences))
    elif resume and report_file.exists():
        prior = read_cache(report_file)
        if any(prior.get(k) != v for k, v in options.items()):
            differences = changed_fields({k: prior.get(k) for k in options}, options)
            raise CacheReuseError("旧缓存预处理参数不匹配：" + ", ".join(differences))
        print("旧版缓存：逐例重新读取原始数据并校验，完整缓存保留，不重新压缩。", flush=True)
    elif resume and any(p.name != ".writer.lock" for p in output.iterdir()):
        raise CacheReuseError("无法识别已有预处理目录，缺少来源记录")
    complete_file = output / "dataset.json"
    complete = read_cache(complete_file) if complete_file.exists() else None
    checked_sources, verified = inspect_sources(output, pairs, complete, options)
    if complete is None:
        write_json(contract_file, contract)
    report = dict(
        options,
        status="running",
        resolution="resized" if size else "resampled" if excel else "native",
        cases=[],
        excluded=[],
        discovered=len(pairs),
        geometry=(
            "Excel spacing used for resampling; original grid retained for prediction restoration"
            if excel
            else "native header retained; no claim that spacing or intensity units are calibrated"
        ),
    )
    records = output / ".resume"
    records.mkdir(exist_ok=True)
    hashes = set()
    try:
        for index, (case, image_path, label_path) in enumerate(pairs):
            if limit and len(report["cases"]) >= limit:
                break
            print(f"预处理 {index + 1}/{len(pairs)}：{case}（校验/处理）", flush=True)
            sources = checked_sources.get(case) or dict(
                image=file_hash(image_path), label=file_hash(label_path) if label_path else None
            )
            sidecar = records / (case + ".json")
            cached = read_json(sidecar) if sidecar.exists() else None
            if cached and cached["sources"] != sources:
                raise ValueError(f"{case} 原始影像或标签已改变；不能混用已有实验")
            cache_file = output / (case + ".npz")
            valid = cached and (
                cached.get("excluded")
                or case in verified
                or (cache_file.is_file() and file_hash(cache_file) == cached.get("cache_sha256"))
            )
            if valid:
                row = cached["row"]
                excluded = cached["excluded"]
                print("  已校验，保留：" + case, flush=True)
            else:
                arrays, row = prepare_case(
                    case, image_path, label_path, size, spacings.get(case), target_spacing
                )
                excluded = arrays is None
                if not excluded:
                    same = False
                    if cache_file.exists():
                        try:
                            with np.load(cache_file, allow_pickle=False) as previous:
                                same = set(previous.files) == set(arrays) and all(
                                    np.array_equal(previous[k], v) for k, v in arrays.items()
                                )
                        except (ValueError, EOFError):
                            pass
                        except Exception as error:
                            # zipfile.BadZipFile and decompression errors are damaged caches.
                            import zipfile
                            import zlib

                            if not isinstance(error, (zipfile.BadZipFile, zlib.error)):
                                raise
                    if not same:
                        if complete:
                            raise CacheReuseError("已完成缓存损坏；请用新目录重建，避免改变已有实验")
                        print("  写入缓存：" + case, flush=True)
                        with atomic_path(cache_file) as temporary:
                            np.savez_compressed(temporary, **arrays)
                    print("  " + ("已校验旧缓存，保留：" if same else "已保存：") + case, flush=True)
                cached = dict(
                    sources=sources,
                    row=row,
                    excluded=excluded,
                    cache_sha256=None if excluded else file_hash(cache_file),
                )
                if complete is None:
                    write_json(sidecar, cached)
                del arrays
            if excluded:
                report["excluded"].append(row)
            else:
                digest = row["source_sha256"]
                if digest in hashes:
                    raise ValueError("检测到内容完全重复的影像；请先确认患者身份，避免跨集合泄漏")
                hashes.add(digest)
                report["cases"].append(dict(row))
            if not complete:
                write_json(report_file, report)
        if not report["cases"]:
            raise ValueError("没有可用病例")
        if not for_prediction:
            splits = assign_splits([r["case_id"] for r in report["cases"]], seed, split_file)
            for row in report["cases"]:
                row.update(splits[row["case_id"]])
        report["status"] = "prepared"
        if complete:
            if complete["cases"] != report["cases"] or complete["excluded"] != report["excluded"]:
                raise CacheReuseError("已完成清单与原始数据不符；请使用新目录")
            print(f"预处理缓存沿用成功：{output}（{len(complete['cases'])} 例，未重复生成缓存）", flush=True)
            return complete  # Preserve its bytes/hash for downstream checkpoints.
        write_json(complete_file, report)
        print(f"已准备 {len(report['cases'])} 例；排除 {len(report['excluded'])} 例：{output}", flush=True)
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        raise
    finally:
        if not complete:
            write_json(report_file, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅预处理 ImageCHD；生成可复用缓存，不训练、不预测")
    parser.add_argument("--input", required=True, help="已解压的 ImageCHD 目录；预测模式也支持单个 NIfTI")
    parser.add_argument("--output", required=True, help="新的预处理结果目录")
    parser.add_argument("--size", type=int, default=0, help="0 保留重采样后的网格；正数再缩放为立方体")
    parser.add_argument(
        "--spacing-excel", help="影像间距表；默认自动读取影像同目录的 imagechd_dataset_image_info.xlsx"
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1, 1, 1),
        metavar=("X", "Y", "Z"),
        help="重采样目标间距，默认 1 1 1 mm",
    )
    parser.add_argument("--no-resample", action="store_true", help="保留旧预处理行为，忽略同目录间距表")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", help="显式患者划分 CSV；case_id 示例 ct_1001")
    parser.add_argument("--for-prediction", action="store_true", help="无标签预处理，不生成训练划分")
    parser.add_argument("--limit", type=int, help="只用于软件试跑，保留前 N 个合格病例")
    parser.add_argument("--resume", action="store_true", help="校验已有缓存并接续，参数和来源必须一致")
    parser.add_argument("--reuse-failure-report", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.reuse_failure_report:
        Path(args.reuse_failure_report).unlink(missing_ok=True)
    try:
        prepare(
            args.input,
            args.output,
            args.size,
            args.seed,
            args.for_prediction,
            args.split_file,
            args.limit,
            args.resume,
            args.spacing_excel,
            args.target_spacing,
            args.no_resample,
        )
    except CacheReuseError as error:
        if not args.reuse_failure_report:
            raise
        write_json(
            Path(args.reuse_failure_report),
            dict(
                kind="cache_reuse_failed",
                prepared=str(Path(args.output).expanduser().resolve()),
                reason=str(error),
            ),
        )
        print("预处理缓存不能沿用：" + str(error), file=sys.stderr, flush=True)
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
