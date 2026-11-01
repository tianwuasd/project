"""Dataset selection and isolated downsampled copies for the smoke run."""

import csv
from pathlib import Path


def find_manifests(selection):
    path = Path(str(selection).strip().strip('"')).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        # A conventional child data/ directory is also supported without a recursive disk scan.
        files.extend(sorted((path / "data").glob("*.csv")))
        valid = []
        for candidate in files:
            try:
                with candidate.open(encoding="utf-8-sig", newline="") as stream:
                    columns = next(csv.reader(stream), [])
                if {"case_id", "patient_id", "split", "image", "label"}.issubset(columns):
                    valid.append(candidate.resolve())
            except (OSError, UnicodeError, csv.Error):
                continue
        if valid:
            return valid
    raise ValueError(
        "未找到 manifest.csv 数据清单。请选择 CSV，或包含它的数据目录；参阅 docs/data-guide.md。"
    )


def validate_dataset(manifest):
    import numpy as np

    from ..data import aligned_label, load_volume, read_manifest

    rows = read_manifest(manifest)
    splits = {key: sum(r["split"] == key for r in rows) for key in ("train", "val", "test")}
    if not splits["train"] or not splits["val"]:
        raise ValueError("数据清单至少需要 train 和 val，且患者不能跨集合。")
    foreground = set()
    for i, row in enumerate(rows):
        print(f"校验影像 {i + 1}/{len(rows)}", flush=True)
        volume = load_volume(row["image"])
        axes = volume.affine[:3, :3] / np.asarray(volume.spacing)
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
            raise ValueError("影像含 shear；距离特征需要先重采样到正交网格。")
        labels = aligned_label(row["label"], volume)
        foreground.update(int(x) for x in np.unique(labels) if x)
        if row.get("initial_label"):
            aligned_label(row["initial_label"], volume, 7)
    complete = all(r.get("initial_label") for r in rows if r["split"] in ("train", "val"))
    return {
        "cases": len(rows),
        "splits": splits,
        "initial_complete": complete,
        "foreground_ids": sorted(foreground),
        "note": "只校验格式、网格和编号范围；解剖语义与标签映射仍须按数据规范确认。",
    }


def prepare_sample(manifest, destination, edge=32):
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to

    from ..data import aligned_label, load_volume, read_manifest

    rows = read_manifest(manifest)
    selected = []
    for split, limit in (("train", 2), ("val", 1), ("test", 1)):
        selected.extend([r for r in rows if r["split"] == split][:limit])
    if not any(r["split"] == "train" for r in selected) or not any(r["split"] == "val" for r in selected):
        raise ValueError("需要 train 和 val 才能试跑")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    output_rows = []
    for i, row in enumerate(selected):
        volume = load_volume(row["image"])
        scale = min(1.0, edge / max(volume.data.shape))
        shape = tuple(max(2, round(n * scale)) for n in volume.data.shape)
        factors = np.asarray(volume.data.shape) / np.asarray(shape)
        transform = np.diag([*factors, 1.0])
        transform[:3, 3] = (factors - 1) / 2
        target_affine = volume.affine @ transform
        copied = {k: row[k] for k in ("case_id", "patient_id", "split")}
        copied["initial_label"] = ""
        for key, classes in (("image", None), ("label", 11), ("initial_label", 7)):
            if not row.get(key):
                continue
            array = volume.data if classes is None else aligned_label(row[key], volume, classes)
            resampled = resample_from_to(
                nib.Nifti1Image(array, volume.affine),
                (shape, target_affine),
                order=1 if classes is None else 0,
            )
            path = destination / f"case_{i:03d}_{key}.nii.gz"
            nib.save(resampled, path)
            copied[key] = str(path)
        output_rows.append(copied)
    output = destination / "manifest.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case_id", "patient_id", "split", "image", "label", "initial_label"]
        )
        writer.writeheader()
        writer.writerows(output_rows)
    inference = next(
        (r for r in output_rows if r["split"] == "test"), next(r for r in output_rows if r["split"] == "val")
    )
    return {
        "manifest": str(output),
        "image": inference["image"],
        "inference_split": inference["split"],
        "cases": len(output_rows),
        "max_dimension": edge,
        "downsampled": True,
        "note": "缩小副本只用于检查流程；未修改源数据，也不评估医学效果。",
    }
