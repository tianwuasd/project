"""医学影像读取和统计：只读原始文件，不替用户修复标签或空间文件头。"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np

# 官方编号只用于分类；205 比 500 小并不意味着器官更小。
LABELS = {0: "背景", 205: "左心室心肌 Myo", 420: "左心房血池 LA",
          500: "左心室血池 LV", 550: "右心房血池 RA", 600: "右心室血池 RV",
          820: "升主动脉 AO", 850: "肺动脉 PA"}
COLORS = {0: "#000000", 205: "#ef6575", 420: "#ffd166", 500: "#40c9a2",
          550: "#55a6ed", 600: "#b194e5", 820: "#ff8c61", 850: "#ed99c6"}


def issue(code, detail, severity="需核实"):
    """问题分级是技术检查结果，不是对患者健康状况的评价。"""
    return {"code": code, "severity": severity, "detail": detail}


def discover(root: Path):
    """按相对路径配对，而非仅按病例号，避免不同中心同名病例互相覆盖。"""
    pairs, problems = [], []
    images = sorted(root.rglob("*_image.nii.gz"))
    for image in images:
        label = image.with_name(image.name.replace("_image.nii.gz", "_label.nii.gz"))
        group = str(image.parent.relative_to(root))
        group = group if group != "." else "未分组"
        lower = group.lower()
        # 模态只从明确的目录名识别，不根据灰度猜测。
        modality = "CT" if "ct" in lower else "MRI" if "mr" in lower else "未知"
        pairs.append({"case": image.name[:-len("_image.nii.gz")], "group": group,
                      "modality": modality, "image": str(image), "label": str(label)})
        if not label.exists():
            problems.append({"path": str(image), **issue("missing_label", "缺少对应标签", "错误")})
    for label in sorted(root.rglob("*_label.nii.gz")):
        if not label.with_name(label.name.replace("_label.nii.gz", "_image.nii.gz")).exists():
            problems.append({"path": str(label), **issue("orphan_label", "标签没有对应影像", "错误")})
    return pairs, problems


def file_hash(path):
    """检查完全相同的文件字节；不能据此证明不同文件来自不同患者。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_affine(matrix):
    # 先归一化列向量再判断退化，避免同一网格用“米”表示时被误判为奇异。
    if matrix is None or np.shape(matrix) != (4, 4) or not np.isfinite(matrix).all():
        return False
    linear = np.asarray(matrix)[:3, :3]
    lengths = np.linalg.norm(linear, axis=0)
    return bool(np.all(lengths > 0) and abs(np.linalg.det(linear / lengths)) > 1e-8)


def describe_header(img):
    """记录 NIfTI 的实际缩放和物理空间；空间单位未知时不冒称 mm/mL。"""
    h = img.header
    space_unit = h.get_xyzt_units()[0]
    factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(space_unit)
    spacing = np.asarray(h.get_zooms()[:3], dtype=float)
    qform, qcode = img.get_qform(coded=True)
    sform, scode = img.get_sform(coded=True)
    preferred = np.array(img.affine, dtype=float)
    affine_valid = valid_affine(preferred)
    nominal_voxel_ml = float(np.prod(spacing * factor) / 1000) if factor else None
    # sform 可以描述不同于 pixdim 的缩放甚至剪切。行列式给出真实网格单元体积，
    # 不能只乘三个边长；矩阵无效时仅给出文件头间距推算的名义值并标注依据。
    voxel_ml = (float(abs(np.linalg.det(preferred[:3, :3])) * factor ** 3 / 1000)
                if factor and affine_valid else nominal_voxel_ml)
    volume_basis = ("首选空间矩阵行列式" if affine_valid else "间距名义值（空间矩阵无效，需核实）") if factor else "空间单位未知，未计算"
    # 显示回退仅用于定位预览，不修改原件，也不抹去 preferred affine 的错误。
    display = preferred if valid_affine(preferred) else qform if valid_affine(qform) else None
    return {
        "shape": list(img.shape), "spacing_native": spacing.tolist(), "space_unit": space_unit,
        "spacing_mm": (spacing * factor).tolist() if factor else None,
        "voxel_ml": voxel_ml, "nominal_voxel_ml": nominal_voxel_ml, "volume_basis": volume_basis,
        "affine_spacing_mm": (np.linalg.norm(preferred[:3, :3], axis=0) * factor).tolist() if factor and affine_valid else None,
        "fov_mm": (spacing * factor * np.asarray(img.shape[:3])).tolist() if factor else None,
        "storage_dtype": str(h.get_data_dtype()),
        # NiBabel 把文件中的 slope/intercept 放到 dataobj；header 中可能已重置。
        "slope": float(getattr(img.dataobj, "slope", 1)),
        "intercept": float(getattr(img.dataobj, "inter", 0)),
        "qform_code": int(qcode), "sform_code": int(scode),
        "qform": qform.tolist() if qform is not None else None,
        "sform": sform.tolist() if sform is not None else None,
        "affine": preferred.tolist(), "affine_valid": valid_affine(preferred),
        "orientation": "".join(nib.aff2axcodes(preferred)) if valid_affine(preferred) else "无效",
        "display_affine": display.tolist() if display is not None else None,
        "display_fallback": not valid_affine(preferred) and display is not None,
    }


def geometry_check(image_meta, label_meta):
    """比较物理坐标，而非逐字节比较 qform/sform；容忍浮点舍入误差。"""
    problems = []
    for role, m in [("影像", image_meta), ("标签", label_meta)]:
        if not m["affine_valid"]:
            problems.append(issue("invalid_affine", f"{role}首选空间矩阵无效/奇异；不自动修复", "错误"))
        if not m["qform_code"] and not m["sform_code"]:
            problems.append(issue("uncoded_geometry", f"{role}没有启用 qform/sform，位置依赖默认解释"))
        if m["spacing_mm"] is None:
            problems.append(issue("unknown_units", f"{role}空间单位 {m['space_unit']} 无法换算为毫米"))
        if m["affine_spacing_mm"] is not None:
            if not np.allclose(m["spacing_mm"], m["affine_spacing_mm"], rtol=1e-4, atol=1e-5):
                problems.append(issue("spacing_affine_mismatch", f"{role}文件头间距与首选矩阵缩放不一致；体积采用矩阵，间距图采用文件头，需核实"))
            grid = np.array(m["affine"])[:3, :3]
            unit_columns = grid / np.linalg.norm(grid, axis=0)
            if not np.allclose(unit_columns.T @ unit_columns, np.eye(3), atol=1e-4):
                problems.append(issue("nonorthogonal_grid", f"{role}空间网格存在剪切；体积采用行列式，预览不能完整表达剪切"))
    if image_meta["shape"] != label_meta["shape"]:
        problems.append(issue("shape_mismatch", "影像与标签尺寸不一致", "错误"))
        return problems
    # 用八个角点的最大距离检查整幅影像的空间一致性，单位为 mm。
    units = {"mm": 1., "meter": 1000., "micron": .001}
    if (image_meta["affine_valid"] and label_meta["affine_valid"]
            and image_meta["space_unit"] in units and label_meta["space_unit"] in units):
        a, b = np.array(image_meta["affine"]), np.array(label_meta["affine"])
        a[:3] *= units[image_meta["space_unit"]]
        b[:3] *= units[label_meta["space_unit"]]
        corners = np.array([(*c, 1) for c in itertools.product(*[(0, n - 1) for n in image_meta["shape"]])]).T
        distance = float(np.linalg.norm(((a - b) @ corners)[:3], axis=0).max())
        if distance > .01:
            problems.append(issue("geometry_mismatch", f"角点最大物理偏差 {distance:.4f} mm", "错误"))
    return problems


def label_statistics(array, voxel_ml):
    """完整统计每个取值；不将 421 合并为 420，也不把小数标签四舍五入。"""
    counts = {}
    nonfinite = 0
    # 沿第一轴分块，控制 unique 排序临时内存，统计仍覆盖全部体素。
    for start in range(0, array.shape[0], 16):
        slab = array[start:start + 16]
        finite = np.isfinite(slab)
        nonfinite += int(slab.size - np.count_nonzero(finite))
        values, nums = np.unique(slab[finite], return_counts=True)
        for value, number in zip(values, nums):
            value = float(value)
            counts[value] = counts.get(value, 0) + int(number)
        if len(counts) > 256:
            raise ValueError("标签取值超过 256 种，可能误把连续影像当作标签；停止标签统计")
    total = int(array.size)
    official_total = sum(v for k, v in counts.items() if k in LABELS and k != 0)
    rows = []
    for value in sorted(set(counts) | set(LABELS)):
        number = counts.get(value, 0)
        rows.append({"value": int(value) if float(value).is_integer() else value,
                     "name": LABELS.get(value, "未定义标签"), "voxels": number,
                     "volume_ml": number * voxel_ml if voxel_ml is not None else None,
                     "image_fraction": number / total,
                     "target_fraction": number / official_total if value in LABELS and value != 0 and official_total else None})
    problems = []
    unknown = [r for r in rows if r["name"] == "未定义标签" and r["voxels"]]
    if unknown:
        problems.append(issue("unexpected_label", "非官方取值：" + ", ".join(f"{r['value']} ({r['voxels']} 体素)" for r in unknown)))
    missing = [v for v in LABELS if v and counts.get(v, 0) == 0]
    if missing:
        problems.append(issue("missing_class", f"没有出现的目标标签：{missing}"))
    if nonfinite:
        problems.append(issue("nonfinite_label", f"标签含 {nonfinite} 个 NaN/Inf", "错误"))
    if any(not float(v).is_integer() for v in counts):
        problems.append(issue("fractional_label", "标签含非整数值，可能有不合适的插值或缩放", "错误"))
    return rows, problems, nonfinite


def intensity_statistics(array, max_samples):
    """min/max 与有限值检查覆盖全图；分位数只基于确定性均匀抽样。"""
    low, high, finite_count = float("inf"), -float("inf"), 0
    for start in range(0, array.shape[0], 16):
        slab = array[start:start + 16]
        values = slab[np.isfinite(slab)]
        finite_count += int(values.size)
        if values.size:
            low, high = min(low, float(values.min())), max(high, float(values.max()))
    # order='K' 尽可能沿现有内存顺序取视图，避免三维数组整体复制。
    flat = array.ravel(order="K")
    step = max(1, int(np.ceil(flat.size / max_samples)))
    sample = flat[::step]
    sample = sample[np.isfinite(sample)]
    q = np.percentile(sample, [1, 50, 99]).tolist() if sample.size else [None] * 3
    return {"min": low if finite_count else None, "max": high if finite_count else None,
            "finite_voxels": finite_count, "nonfinite_voxels": int(array.size - finite_count),
            "sample_count": int(sample.size), "p01": q[0], "median": q[1], "p99": q[2]}, sample.copy()


def inspect_case(pair, max_samples=200000, hashes=True):
    """逐例完整读取，返回 JSON 兼容记录以及小规模抽样，用完即释放大数组。"""
    r = dict(pair, status="ok", issues=[])
    image, label = nib.load(pair["image"]), nib.load(pair["label"])
    if len(image.shape) != 3 or len(label.shape) != 3:
        raise ValueError("当前脚本只处理三维标量影像和标签；不把 4D 时间序列悄悄压成 3D")
    im, lm = describe_header(image), describe_header(label)
    r.update(image_meta=im, label_meta=lm)
    r["issues"].extend(geometry_check(im, lm))
    # dataobj 会应用 NIfTI slope/intercept；不能再手动减一次 1024。
    a = np.asarray(image.dataobj, dtype=np.float32)
    r["intensity"], sample = intensity_statistics(a, max_samples)
    del a
    labels = np.asanyarray(label.dataobj)
    r["labels"], problems, r["label_nonfinite"] = label_statistics(labels, lm["voxel_ml"])
    del labels
    r["issues"].extend(problems)
    if r["intensity"]["nonfinite_voxels"]:
        r["issues"].append(issue("nonfinite_image", f"影像含 {r['intensity']['nonfinite_voxels']} 个 NaN/Inf", "错误"))
    if r["intensity"]["min"] == r["intensity"]["max"]:
        r["issues"].append(issue("constant_image", "影像为常数或没有有限值", "错误"))
    r["image_bytes"] = Path(pair["image"]).stat().st_size
    r["label_bytes"] = Path(pair["label"]).stat().st_size
    if hashes:
        r["image_sha256"] = file_hash(pair["image"])
        r["label_sha256"] = file_hash(pair["label"])
    return r, sample
