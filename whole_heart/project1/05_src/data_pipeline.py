"""病例划分与 nnU-Net 数据转换。原始 NIfTI 永不覆写。

每个外层目标组完全排除在训练目录之外；来源开发集只用来选模型。
训练标签使用 0..7，提交/分析前必须调用 inverse_labels 恢复官方编号。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"background": 0, "Myo": 1, "LA": 2, "LV": 3, "RA": 4, "RV": 5, "AO": 6, "PA": 7}
OFFICIAL = np.array([0, 205, 420, 500, 550, 600, 820, 850], dtype=np.int16)
GROUPS = {"A": "A ct_train", "B": "B ct_train", "G": "G ct_train", "CD": "C and D mr_train", "E": "E mr_train"}
FOLDS = {"ct_holdA": (701, "CT", "A"), "ct_holdB": (702, "CT", "B"), "ct_holdG": (703, "CT", "G"), "mr_holdCD": (704, "MRI", "CD"), "mr_holdE": (705, "MRI", "E")}


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


def make_splits(cases, seed=20260919):
    """使用种子和病例 ID 的哈希排序，避免依赖库随机数版本。

    同一来源组在各轮保留同一个开发子集，减少不必要的划分变化。
    SHA 相同病例不允许被分开；病例身份缺失仍是已知局限。
    """
    eligible = [c for c in cases if c["status"] == "eligible"]
    hashes = [c["image_sha256"] for c in eligible]
    if len(hashes) != len(set(hashes)):
        raise ValueError("合格清单仍有重复影像，必须先核实或按患者组划分")
    result = {}
    for fold, (dataset_id, modality, holdout) in FOLDS.items():
        source_groups = [g for g in GROUPS if g != holdout and any(c["group"] == GROUPS[g] and c["modality"] == modality for c in eligible)]
        train, val = [], []
        for group in source_groups:
            ids = [c["case_id"] for c in eligible if c["group"] == GROUPS[group]]
            ids.sort(key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
            n_val = 5 if group == "E" else 4
            val.extend(ids[:n_val])
            train.extend(ids[n_val:])
        target = sorted(c["case_id"] for c in eligible if c["group"] == GROUPS[holdout])
        result[fold] = {"dataset_id": dataset_id, "modality": modality, "holdout": holdout, "split_seed": seed, "train": sorted(train), "val": sorted(val), "target": target, "fingerprint_fit_ids": sorted(train)}
    return result


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


def export_fold(root, fold, split, cases):
    """为每个外层实验建立独立数据集；只复制来源病例。"""
    out = root / "04_data/derived/nnUNet_raw" / f"Dataset{split['dataset_id']:03d}_{fold}"
    if out.exists():
        raise FileExistsError(f"禁止覆盖已有数据集: {out}")
    (out / "imagesTr").mkdir(parents=True)
    (out / "labelsTr").mkdir()
    lookup = {c["case_id"]: c for c in cases}
    records = []
    for case_id in split["train"] + split["val"]:
        case = lookup[case_id]
        image, label, mapped = validate_geometry(case)
        shutil.copyfile(case["image"], out / "imagesTr" / f"{case_id}_0000.nii.gz")
        header = label.header.copy()
        header.set_data_dtype(np.uint8)
        converted = nib.Nifti1Image(mapped, label.affine, header)
        path = out / "labelsTr" / f"{case_id}.nii.gz"
        nib.save(converted, path)
        restored = nib.load(path)
        if not np.array_equal(np.asanyarray(restored.dataobj), mapped) or not np.allclose(restored.affine, image.affine, atol=1e-4, rtol=0):
            raise ValueError(f"写入后验证失败: {case_id}")
        records.append({"case_id": case_id, "shape": list(image.shape), "spacing_mm": list(map(float, image.header.get_zooms())), "geometry_and_label_roundtrip": "passed"})
        print(f"exported {fold}/{case_id}", flush=True)
    save_json(out / "dataset.json", {"channel_names": {"0": "CT" if split["modality"] == "CT" else "MRI"}, "labels": LABELS, "numTraining": len(records), "file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIO"})
    save_json(out / "source_only_split.json", split)
    save_json(root / f"09_reports/{fold}_conversion.json", records)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", nargs="*", choices=list(FOLDS), default=[])
    args = parser.parse_args()
    inventory_path = ROOT / "04_data/manifests/case_inventory.json"
    cases = load_json(inventory_path)["cases"]
    splits = make_splits(cases)
    path = ROOT / "04_data/manifests/splits_v1.json"
    payload = {"inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(), "folds": splits, "patient_id_available": False}
    if path.exists() and load_json(path) != payload:
        raise ValueError("划分已冻结且与当前输入不一致，禁止覆盖")
    save_json(path, payload)
    print(json.dumps({k: {p: len(v[p]) for p in ["train", "val", "target"]} for k, v in splits.items()}, indent=2), flush=True)
    for fold in args.export:
        export_fold(ROOT, fold, splits[fold], cases)


if __name__ == "__main__":
    main()
