"""中文统计图和三方向预览。图像变换仅发生在内存，不保存修改后的医学影像。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 批处理环境不弹出窗口，图保存为 PNG。
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

from medical import LABELS, COLORS, describe_header


def setup_font():
    """优先使用 Windows 中文字体；缺字时明确提示用户安装中文字体。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"] if n in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return chosen


def save(fig, output, filename):
    path = Path(output) / "figures" / filename
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path.relative_to(output)).replace("\\", "/")


def group_figures(records, output):
    """每个点表示一例病例，不把切片数量当成样本数。"""
    records = [r for r in records if r["status"] == "ok"]
    groups = sorted({r["group"] for r in records})
    figures = []
    if not groups:
        return figures
    counts = [sum(r["group"] == g for r in records) for g in groups]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(groups, counts, color=["#4682b4" if "ct" in g.lower() else "#9265b4" for g in groups])
    ax.bar_label(bars)
    ax.set(ylabel="成功读取的影像—标签组数", title="数据构成：一组对应一个三维病例记录")
    figures.append(("数据构成", save(fig, output, "01_groups.png")))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for axis, ax in enumerate(axs):
        for i, g in enumerate(groups):
            values = [r["image_meta"]["spacing_mm"][axis] for r in records if r["group"] == g and r["image_meta"]["spacing_mm"]]
            ax.scatter(np.full(len(values), i), values, alpha=.5, s=20)
        ax.set(xticks=range(len(groups)), xticklabels=groups, ylabel="体素间距（mm）", title=f"原始数组第 {axis + 1} 轴")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("采样密度比较：轴号不一定对应相同身体方向；间距不等于设备真实分辨能力")
    fig.tight_layout()
    figures.append(("体素间距", save(fig, output, "02_spacing.png")))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for index, key in enumerate(["shape", "fov_mm"]):
        for i, g in enumerate(groups):
            vals = [r["image_meta"][key] for r in records if r["group"] == g and r["image_meta"][key]]
            # 展示三个原始轴的范围，解释数组大小和物理覆盖并非同一回事。
            for axis in range(3):
                axs[index].scatter(np.full(len(vals), i) + (axis - 1) * .16,
                                   [v[axis] for v in vals], s=12, alpha=.5,
                                   color=["#4682b4", "#ed9850", "#60ad89"][axis],
                                   label=f"第{axis + 1}轴" if i == 0 else None)
        axs[index].set(xticks=range(len(groups)), xticklabels=groups,
                       ylabel="体素数" if key == "shape" else "覆盖长度（mm）",
                       title="数组尺寸" if key == "shape" else "名义物理覆盖（尺寸 × 间距）")
        axs[index].tick_params(axis="x", rotation=25)
        axs[index].legend()
    fig.tight_layout()
    figures.append(("尺寸与覆盖范围", save(fig, output, "03_size_fov.png")))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ax, modality in zip(axs, ["CT", "MRI"]):
        subgroups = [g for g in groups if any(r["group"] == g and r["modality"] == modality for r in records)]
        for i, g in enumerate(subgroups):
            rs = [r for r in records if r["group"] == g and r["intensity"]["median"] is not None]
            if rs:
                x = i + np.linspace(-.15, .15, len(rs))
                med = np.array([r["intensity"]["median"] for r in rs])
                lo = med - np.array([r["intensity"]["p01"] for r in rs])
                hi = np.array([r["intensity"]["p99"] for r in rs]) - med
                ax.errorbar(x, med, yerr=[lo, hi], fmt="o", alpha=.45, markersize=3, linewidth=.8)
        ax.set(xticks=range(len(subgroups)), xticklabels=subgroups,
               ylabel="缩放后的影像强度", title=f"{modality}：每例中位数与 P1—P99 范围")
    fig.suptitle("确定性抽样，包含背景；CT 和 MRI 使用不同纵轴，不能直接比较绝对数值")
    fig.tight_layout()
    figures.append(("影像灰度分布", save(fig, output, "04_intensity.png")))

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for ax, value in zip(axs.flat, [v for v in LABELS if v]):
        for i, g in enumerate(groups):
            vals = [l["volume_ml"] for r in records if r["group"] == g for l in r["labels"]
                    if l["value"] == value and l["volume_ml"] is not None]
            ax.scatter(i + np.linspace(-.15, .15, len(vals)), vals, color=COLORS[value], edgecolors="#555555", linewidths=.3, s=20)
        ax.set(title=LABELS[value], xticks=range(len(groups)), xticklabels=groups, ylabel="标注体积（mL）")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    axs.flat[-1].axis("off")
    axs.flat[-1].text(0, .8, "每个点是一例病例。\n体积按标签体素数 × 体素体积计算。\n异常值应核实，不能直接推断疾病。\n血管标注长度会影响体积。", va="top", fontsize=10)
    fig.tight_layout()
    figures.append(("七种结构体积", save(fig, output, "05_label_volumes.png")))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for i, g in enumerate(groups):
        rs = [r for r in records if r["group"] == g]
        background = [next(l["image_fraction"] for l in r["labels"] if l["value"] == 0) * 100 for r in rs]
        axs[0].scatter(np.full(len(rs), i), background, alpha=.5)
        bottom = 0
        for value in LABELS:
            if not value:
                continue
            vals = [l["target_fraction"] for r in rs for l in r["labels"] if l["value"] == value and l["target_fraction"] is not None]
            mean = float(np.mean(vals) * 100) if vals else 0
            axs[1].bar(i, mean, bottom=bottom, color=COLORS[value], label=LABELS[value] if i == 0 else None)
            bottom += mean
    for ax in axs:
        ax.set_xticks(range(len(groups)), groups)
        ax.tick_params(axis="x", rotation=20)
    axs[0].set(title="背景占整幅影像的比例", ylabel="%")
    axs[1].set(title="七类目标内部的平均比例（每例等权）", ylabel="%")
    axs[1].legend(bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    figures.append(("背景与类别比例", save(fig, output, "06_class_balance.png")))
    return figures


def canonical_array(path, dtype):
    """只做轴交换/翻转到接近 RAS 的方向，不重采样，不消除扫描倾斜。"""
    img = nib.load(path)
    meta = describe_header(img)
    if meta["display_affine"] is None or meta["spacing_mm"] is None:
        raise ValueError("没有可用于毫米预览的有效空间信息")
    matrix = np.asarray(meta["display_affine"])
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(matrix), nib.orientations.axcodes2ornt(("R", "A", "S")))
    arr = np.asarray(img.dataobj, dtype=dtype)
    oriented = nib.orientations.apply_orientation(arr, transform)
    affine = matrix @ nib.orientations.inv_ornt_aff(transform, img.shape)
    factor = {"mm": 1., "meter": 1000., "micron": .001}[meta["space_unit"]]
    affine[:3] *= factor
    return oriented, affine, meta


def case_preview(record, output, index):
    """三列是三个近似解剖平面；三行分别为原图、标签、叠加。"""
    a, am, meta = canonical_array(record["image"], np.float32)
    label, lm, _ = canonical_array(record["label"], np.float32)
    aligned = a.shape == label.shape and np.allclose(am, lm, atol=.01, rtol=0)
    errors = {v["code"] for v in record["issues"]}
    # 即使诊断性 qform 回退看起来重合，也不把空间异常病例画成已验证的叠加图。
    overlay_ok = aligned and not errors.intersection({"invalid_affine", "geometry_mismatch", "shape_mismatch", "uncoded_geometry"})
    fig, axs = plt.subplots(3, 3, figsize=(12, 11))
    unknown_values = [l["value"] for l in record["labels"] if l["name"] == "未定义标签" and l["voxels"]]
    centers = []
    foreground = np.isfinite(label) & (label != 0)
    for axis in range(3):
        occupancy = np.any(foreground, axis=tuple(i for i in range(3) if i != axis))
        indices = np.flatnonzero(occupancy)
        centers.append(int((indices[0] + indices[-1]) // 2) if indices.size else label.shape[axis] // 2)
    del foreground
    # 有非预期标签时，额外预览切在其范围中点，方便用户定位。
    if unknown_values:
        unknown = np.isin(label, unknown_values)
        for axis in range(3):
            indices = np.flatnonzero(np.any(unknown, axis=tuple(i for i in range(3) if i != axis)))
            if indices.size:
                centers[axis] = int((indices[0] + indices[-1]) // 2)
        del unknown
    for axis, title in enumerate(["近似矢状面", "近似冠状面", "近似横断面"]):
        z = min(centers[axis], a.shape[axis] - 1) if aligned else a.shape[axis] // 2
        sl = np.take(a, z, axis=axis).T
        ls = np.take(label, centers[axis], axis=axis).T
        dims = [i for i in range(3) if i != axis]
        asp = np.linalg.norm(am[:3, dims[1]]) / np.linalg.norm(am[:3, dims[0]])
        lasp = np.linalg.norm(lm[:3, dims[1]]) / np.linalg.norm(lm[:3, dims[0]])
        finite = sl[np.isfinite(sl)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
        hi = hi if hi > lo else lo + 1
        rgb = np.zeros((*ls.shape, 4), dtype=np.float32)
        for value in np.unique(ls[np.isfinite(ls)]):
            if value == 0:
                continue
            color = matplotlib.colors.to_rgba(COLORS.get(value, "#00e5ff"))
            rgb[ls == value] = color
        axs[0, axis].imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect=asp)
        axs[1, axis].set_facecolor("#101820")
        axs[1, axis].imshow(rgb, origin="lower", aspect=lasp)
        if overlay_ok:
            axs[2, axis].imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect=asp)
            axs[2, axis].imshow(rgb, origin="lower", aspect=asp, alpha=.55)
        else:
            axs[2, axis].text(.5, .5, "空间信息需核实\n不生成叠加图", ha="center", va="center", transform=axs[2, axis].transAxes)
        axs[0, axis].set_title(f"{title}｜切片索引 {z}")
        for row in range(3):
            axs[row, axis].set_xticks([])
            axs[row, axis].set_yticks([])
    for ax, text in zip(axs[:, 0], ["原始影像", "人工标签", "标签叠加"]):
        ax.set_ylabel(text, fontsize=12)
    legend = [Patch(color=COLORS[v], label=f"{v} {LABELS[v]}") for v in LABELS if v]
    if unknown_values:
        legend.append(Patch(color="#00e5ff", label="未定义 " + ",".join(map(str, unknown_values))))
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8)
    detail = "仅显示缩放：各切片 P1—P99；按体素间距等比例；RAS 轴视图保留原始倾斜"
    if meta["display_fallback"]:
        detail += "\n原影像首选矩阵无效，单独显示临时采用 qform；原件未修复"
    fig.suptitle(f"{record['group']} / {record['case']}\n{detail}", fontsize=11)
    fig.tight_layout(rect=[0, .08, 1, .91])
    return save(fig, output, f"case_{index:03d}_{record['case']}.png")
