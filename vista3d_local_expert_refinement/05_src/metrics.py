"""七结构的本地指标：完整标注范围，不冒充官方血管截断评价。

表面定义为六邻域腐蚀后剩余的边界体素中心。通过完整 affine 变换到
毫米物理空间，兼容旋转和斜切。HD95 定义为两个有向 95 分位数的较大值。
"""
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree


def boundary_points(mask, affine):
    surface = mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)
    points = np.argwhere(surface)
    return points @ affine[:3, :3].T + affine[:3, 3]


def binary_metrics(prediction, reference, affine):
    pred, ref = np.asarray(prediction, dtype=bool), np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape or pred.ndim != 3:
        raise ValueError("指标输入必须在同一个三维网格上")
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("无效物理空间矩阵")
    p, r = int(pred.sum()), int(ref.sum())
    if p + r == 0:
        return {"dice": None, "hd_mm": None, "hd95_mm": None, "status": "both_empty"}
    dice = 2 * int((pred & ref).sum()) / (p + r)
    if not p or not r:
        return {"dice": dice, "hd_mm": float("inf"), "hd95_mm": float("inf"), "status": "one_empty"}
    a, b = boundary_points(pred, affine), boundary_points(ref, affine)
    ab = cKDTree(b).query(a, workers=1)[0]
    ba = cKDTree(a).query(b, workers=1)[0]
    return {"dice": dice, "hd_mm": float(max(ab.max(), ba.max())), "hd95_mm": float(max(np.percentile(ab, 95), np.percentile(ba, 95))), "status": "ok"}
