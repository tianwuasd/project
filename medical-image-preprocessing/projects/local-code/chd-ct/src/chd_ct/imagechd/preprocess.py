"""Raw NIfTI -> reusable caches and split manifest; never trains a model."""

import argparse
import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..recovery import atomic_path, file_hash, output_lock, read_json, write_json
from .common import FORMAT, IGNORE, LABELS, NORMALIZATION, normalize
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


def prepare_case(case, image_path, label_path, size):
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
        targets["target"] = (resize(target, (size,) * 3, labels=True) if size else target).astype(np.uint8)
    else:
        extra = []
    print("  归一化：" + case, flush=True)
    image, intensity = normalize(canonical.get_fdata(dtype=np.float32))
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
    return arrays, row


def prepare(source, output, size=0, seed=42, for_prediction=False, split_file=None, limit=None, resume=False):
    if (size != 0 and size < 8) or (limit is not None and limit < 3):
        raise ValueError("size=0 保留原始网格，或至少为8；测试用 limit 至少为3")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output):
        return prepare_locked(source, output, size, seed, for_prediction, split_file, limit, resume)


def prepare_locked(source, output, size, seed, for_prediction, split_file, limit, resume):
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
    contract = dict(
        options=options,
        sources=[[case, str(image), str(label) if label else None] for case, image, label in pairs],
        split_sha256=file_hash(split_file) if split_file else None,
    )
    contract_file = output / "preprocess-contract.json"
    report_file = output / "preprocess-report.json"
    if contract_file.exists():
        if read_json(contract_file) != contract:
            raise ValueError("续跑的数据来源、划分或预处理参数改变；请使用新目录")
    elif resume and report_file.exists():
        prior = read_json(report_file)
        if any(prior.get(k) != v for k, v in options.items()):
            raise ValueError("旧缓存预处理参数不匹配；请使用原参数续跑")
        print("旧版缓存：逐例重新读取原始数据并校验，完整缓存保留，不重新压缩。", flush=True)
    elif resume and any(p.name != ".writer.lock" for p in output.iterdir()):
        raise ValueError("无法识别已有预处理目录，缺少来源记录")
    write_json(contract_file, contract)
    complete_file = output / "dataset.json"
    complete = read_json(complete_file) if complete_file.exists() else None
    report = dict(
        options,
        status="running",
        resolution="resized" if size else "native",
        cases=[],
        excluded=[],
        discovered=len(pairs),
        geometry="native header retained; no claim that spacing or intensity units are calibrated",
    )
    records = output / ".resume"
    records.mkdir(exist_ok=True)
    hashes = set()
    try:
        for index, (case, image_path, label_path) in enumerate(pairs):
            if limit and len(report["cases"]) >= limit:
                break
            print(f"预处理 {index + 1}/{len(pairs)}：{case}（校验/处理）", flush=True)
            sources = dict(image=file_hash(image_path), label=file_hash(label_path) if label_path else None)
            sidecar = records / (case + ".json")
            cached = read_json(sidecar) if sidecar.exists() else None
            if cached and cached["sources"] != sources:
                raise ValueError(f"{case} 原始影像或标签已改变；不能混用已有实验")
            cache_file = output / (case + ".npz")
            valid = cached and (
                cached.get("excluded")
                or (cache_file.is_file() and file_hash(cache_file) == cached.get("cache_sha256"))
            )
            if valid:
                row = cached["row"]
                excluded = cached["excluded"]
                print("  已校验，保留：" + case, flush=True)
            else:
                arrays, row = prepare_case(case, image_path, label_path, size)
                excluded = arrays is None
                if not excluded:
                    same = False
                    if cache_file.exists():
                        try:
                            with np.load(cache_file, allow_pickle=False) as previous:
                                same = set(previous.files) == set(arrays) and all(
                                    np.array_equal(previous[k], v) for k, v in arrays.items()
                                )
                        except (OSError, ValueError, EOFError):
                            pass
                        except Exception as error:
                            # zipfile.BadZipFile and decompression errors are damaged caches.
                            import zipfile
                            import zlib

                            if not isinstance(error, (zipfile.BadZipFile, zlib.error)):
                                raise
                    if not same:
                        if complete:
                            raise ValueError("已完成缓存损坏；请用新目录重建，避免改变已有实验")
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
                raise ValueError("已完成清单与原始数据不符；请使用新目录")
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
    parser.add_argument("--size", type=int, default=0, help="0 保留原始分辨率；正数仅用于生成缩小缓存")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", help="显式患者划分 CSV；case_id 示例 ct_1001")
    parser.add_argument("--for-prediction", action="store_true", help="无标签预处理，不生成训练划分")
    parser.add_argument("--limit", type=int, help="只用于软件试跑，保留前 N 个合格病例")
    parser.add_argument("--resume", action="store_true", help="校验已有缓存并接续，参数和来源必须一致")
    args = parser.parse_args(argv)
    prepare(
        args.input,
        args.output,
        args.size,
        args.seed,
        args.for_prediction,
        args.split_file,
        args.limit,
        args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
