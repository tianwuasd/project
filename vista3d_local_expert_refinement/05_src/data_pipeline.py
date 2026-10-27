"""病例划分与 nnU-Net 数据转换。原始 NIfTI 永不覆写。

每个外层目标组完全排除在训练目录之外；来源开发集只用来选模型。
训练标签使用 0..7，提交/分析前必须调用 inverse_labels 恢复官方编号。
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
# 服务器写入独立工作区，本机不设置该变量时维持原来的目录布局。
ROOT = Path(os.environ.get("WHOLE_HEART_WORKSPACE", str(CODE_ROOT))).expanduser().resolve()
LABELS = {"background": 0, "Myo": 1, "LA": 2, "LV": 3, "RA": 4, "RV": 5, "AO": 6, "PA": 7}
OFFICIAL = np.array([0, 205, 420, 500, 550, 600, 820, 850], dtype=np.int16)
from cohort import GROUPS, FOLDS, make_splits, validate_split


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode_extra(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"不能序列化 {type(obj)}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode_extra), encoding="utf-8")


def map_labels(array):
    """遇到 421、非整数或非有限值立即报错，不猜测类别。"""
    values = np.unique(array)
    if not np.isin(values, OFFICIAL).all():
        raise ValueError(f"未定义标签: {values[~np.isin(values, OFFICIAL)].tolist()}")
    result = np.zeros(array.shape, dtype=np.uint8)
    for train_id, raw_id in enumerate(OFFICIAL):
        result[array == raw_id] = train_id
    return result


def inverse_labels(array):
    if not np.isin(np.unique(array), np.arange(8)).all():
        raise ValueError("模型标签必须为整数 0..7")
    return OFFICIAL[array.astype(np.intp)]


def validate_geometry(case):
    """仅验几何和标签；原始图像阵列的强度统计由来源指纹阶段计算。"""
    image, label = nib.load(case["image"]), nib.load(case["label"])
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError(f"形状异常 {case['case_id']}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError(f"空间矩阵异常 {case['case_id']}")
    if not np.allclose(image.affine, label.affine, atol=1e-4, rtol=0):
        raise ValueError(f"影像/标注空间不匹配 {case['case_id']}")
    raw = np.asanyarray(label.dataobj)
    mapped = map_labels(raw)
    if not np.array_equal(inverse_labels(mapped), raw):
        raise ValueError("标签往返转换失败")
    return image, label, mapped


def export_case(root,fold,split,cases,case_id):
    if case_id not in split['train']+split['val']:raise ValueError('export case outside frozen source scope')
    out=Path(root)/"04_data/derived/nnUNet_raw"/f"Dataset{split['dataset_id']:03d}_{fold}"
    case = next(c for c in cases if c["case_id"]==case_id)
    from read_admission import read_slot
    with read_slot():image, label, mapped = validate_geometry(case)
    image_out = out / "imagesTr" / f"{case_id}_0000.nii.gz"
    copied_ok = False
    if image_out.exists():
        with read_slot(),image_out.open("rb") as stream:
            copied_ok = hashlib.file_digest(stream, "sha256").hexdigest() == case["image_sha256"]
    if not copied_ok:
        temporary = image_out.with_name(image_out.name + ".partial")
        with read_slot():shutil.copyfile(case["image"], temporary)
        os.replace(temporary, image_out)
    header = label.header.copy()
    header.set_data_dtype(np.uint8)
    converted = nib.Nifti1Image(mapped, label.affine, header)
    path = out / "labelsTr" / f"{case_id}.nii.gz"
    temporary = path.with_name(case_id + ".partial.nii.gz")
    nib.save(converted, temporary)
    os.replace(temporary, path)
    with read_slot():
        restored = nib.load(path);restored_array=np.asanyarray(restored.dataobj)
    if not np.array_equal(restored_array, mapped) or not np.allclose(restored.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError(f"写入后验证失败: {case_id}")
    record={"case_id": case_id, "shape": list(image.shape), "spacing_mm": list(map(float, image.header.get_zooms())), "geometry_and_label_roundtrip": "passed"}
    print(f"exported {fold}/{case_id}", flush=True)
    from server_data import write_json
    write_json(out/"export_case_records"/f"{case_id}.json",record)
    return record


def export_fold(root, fold, split, cases, resume=False):
    """为每个外层实验建立独立数据集；只复制来源病例。"""
    validate_split(split, cases)
    out = root / "04_data/derived/nnUNet_raw" / f"Dataset{split['dataset_id']:03d}_{fold}"
    if not out.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("派生数据路径超出工作区")
    if out.exists() and not resume:
        raise FileExistsError(f"禁止覆盖已有数据集: {out}")
    saved_split = out / "source_only_split.json"
    if saved_split.exists() and load_json(saved_split) != split:
        raise ValueError("现有数据集的划分不同，不能继续")
    (out / "imagesTr").mkdir(parents=True, exist_ok=resume)
    (out / "labelsTr").mkdir(exist_ok=resume)
    save_json(saved_split, split)
    lookup = {c["case_id"]: c for c in cases}
    from task_resources import case_budget
    from cpu_case_queue import run
    import sys
    ids=split['train']+split['val']
    paths=[lookup[cid]['image'] for cid in ids]
    refresh=lambda:case_budget(root,paths,phase='source_export',supported_threads=1)
    budget=refresh()
    commands=[(cid,[sys.executable,'-B',str(Path(__file__).resolve()),'--export-case',str(root),fold,cid]) for cid in ids]
    run(commands,budget['workers'],Path(root)/f'00_admin/export_logs/{fold}',lambda _:None,refresh=refresh)
    records=[load_json(out/'export_case_records'/f'{cid}.json') for cid in ids]
    save_json(out / "dataset.json", {"channel_names": {"0": "CT" if split["modality"] == "CT" else "MRI"}, "labels": LABELS, "numTraining": len(records), "file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIO"})
    save_json(out / "source_only_split.json", split)
    save_json(root / f"09_reports/{fold}_conversion.json", records)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", nargs="*", choices=list(FOLDS), default=[])
    parser.add_argument("--export-case",nargs=3,metavar=("ROOT","FOLD","CASE"))
    args = parser.parse_args()
    if args.export_case:
        root,fold,cid=args.export_case;root=Path(root)
        export_case(root,fold,load_json(root/"04_data/manifests/splits_v1.json")["folds"][fold],
            load_json(root/"04_data/manifests/case_inventory.json")["cases"],cid)
        return
    inventory_path = ROOT / "04_data/manifests/case_inventory.json"
    cases = load_json(inventory_path)["cases"]
    splits = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"]
    for split in splits.values():
        validate_split(split, cases)
    print(json.dumps({k: {p: len(v[p]) for p in ["train", "val", "target"]} for k, v in splits.items()}, indent=2), flush=True)
    for fold in args.export:
        from server_data import freeze_source_labels
        freeze_source_labels(ROOT, fold)
        export_fold(ROOT, fold, splits[fold], cases)


if __name__ == "__main__":
    main()
