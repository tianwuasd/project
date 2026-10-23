"""Geometry validation and explicit, lossless-of-source label combination."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .schema import LABELS


def inspect_image(path, full=False):
    path = Path(path)
    if not path.is_file() or not str(path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError(f"需要存在的 NIfTI 文件: {path}")
    image = nib.load(str(path), keep_file_open=True)
    if b'normalized [-1,1]; NOT HU' in bytes(image.header['descrip']):
        raise ValueError('不能把[-1,1]直接作为HU输入，请选择预处理任务的image_hu.nii.gz配套影像')
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError("需要三维单通道体积，每轴至少两个体素")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("空间单位不是 mm；不能猜测真实间距或自动改头")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError("无效空间变换")
    qform, qc = image.get_qform(coded=True)
    sform, sc = image.get_sform(coded=True)
    if not qc and not sc:
        raise ValueError("缺少有效 qform/sform")
    if qc and sc and not np.allclose(qform, sform, atol=1e-3, rtol=0):
        raise ValueError("qform 与 sform 不一致，请核查来源")
    spacing = nib.affines.voxel_sizes(affine)
    if not np.allclose(spacing, image.header.get_zooms()[:3], atol=1e-4, rtol=1e-4):
        raise ValueError("头间距与 affine 不一致")
    warnings = []
    if max(spacing) > 1.5:
        warnings.append("存在大于 1.5 mm 的间距，细小冠脉可能难以分辨；这不是排除标准")
    result = {"shape": list(image.shape), "spacing_mm": spacing.tolist(),
              "affine": affine.tolist(), "warnings": warnings}
    if full:
        # Read the entire payload without keeping a second full-volume float array.
        low, high = float("inf"), float("-inf")
        for z in range(0, image.shape[2], 32):
            slab = np.asarray(image.dataobj[:, :, z:z + 32])
            if not np.isfinite(slab).all():
                raise ValueError("影像包含 NaN 或 Inf")
            low, high = min(low, float(slab.min())), max(high, float(slab.max()))
        if low == high:
            raise ValueError("影像为常数体积")
        result["intensity_range"] = [low, high]
    return result


def combine(image_path, attempt, mask_root=None):
    source = nib.load(str(image_path))
    combined = np.zeros(source.shape, dtype=np.uint8)
    # Later writes win: veins < chambers/myocardium/arteries < coronaries.
    order = [8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 11]
    stats, overlaps, files = {}, [], []
    voxel_mm3 = abs(float(np.linalg.det(source.affine[:3, :3])))
    for label_id in order:
        _, name, chinese, task = LABELS[label_id - 1]
        path = Path(mask_root or attempt) / task / f"{name}.nii.gz"
        mask = nib.load(str(path))
        if mask.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"{name}: 输出空间单位不是 mm")
        if mask.shape != source.shape or not np.allclose(mask.affine, source.affine, atol=1e-3, rtol=0):
            raise ValueError(f"{name}: 输出与输入网格不一致，禁止只复制头信息")
        data = np.asanyarray(mask.dataobj)
        if not np.isin(data, [0, 1]).all():
            raise ValueError(f"{name}: 期望二值掩膜 0/1")
        selected = data == 1
        previous, counts = np.unique(combined[selected], return_counts=True)
        for old, count in zip(previous, counts):
            if old:
                overlaps.append({"overwritten_label": int(old), "winning_label": label_id,
                                 "voxels": int(count)})
        count = int(selected.sum())
        stats[name] = {"id": label_id, "name_zh": chinese, "raw_voxels": count,
                       "raw_volume_mm3": count * voxel_mm3}
        combined[selected] = label_id
        files.append(path)
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    header["cal_min"], header["cal_max"] = 0, 11
    output = nib.Nifti1Image(combined, source.affine, header)
    output.set_sform(source.get_sform(), int(source.header["sform_code"]))
    output.set_qform(source.get_qform(), int(source.header["qform_code"]))
    merged = Path(attempt) / "aaoca_11_labels.nii.gz"
    nib.save(output, str(merged))
    reread = nib.load(str(merged))
    if (not np.array_equal(np.asanyarray(reread.dataobj), combined)
            or not np.allclose(reread.affine, source.affine, atol=1e-3, rtol=0)):
        raise ValueError("合并标签回读不一致")
    for label_id, name, _, _ in LABELS:
        stats[name]["merged_voxels"] = int(np.count_nonzero(combined == label_id))
    report = {"review_required": True, "classes": stats, "overlaps": overlaps,
              "priority_low_to_high": order,
              "empty_masks": [name for name, s in stats.items() if not s["raw_voxels"]],
              "notes": ["冠脉未分支命名；不自动判断开口异常或壁内走行",
                        "复核冠脉开口、近端走行、细小分支及血管连接，预测不能直接当作真值"]}
    qc_path = Path(attempt) / "qc.json"
    qc_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    colors = [(230, 80, 80), (80, 140, 240), (250, 150, 80), (110, 100, 230),
              (190, 100, 150), (255, 190, 50), (60, 200, 180), (170, 100, 220),
              (90, 160, 240), (70, 110, 210), (255, 240, 70)]
    color_path = Path(attempt) / "labels.ctbl"
    lines = ["# AAOCA coarse labels: load as a color table in 3D Slicer", "0 Background 0 0 0 0"]
    for (label_id, name, _, _), rgb in zip(LABELS, colors):
        lines.append(f"{label_id} {name} {' '.join(map(str, rgb))} 255")
    color_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return files + [merged, qc_path, color_path], report
