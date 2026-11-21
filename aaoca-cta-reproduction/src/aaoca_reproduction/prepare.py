from contextlib import contextmanager
import os
from pathlib import Path
import re
from time import perf_counter

import nibabel as nib
import numpy as np
import psutil
from scipy import ndimage
import SimpleITK as sitk

from .records import RELEASE, check_work, event, identity, measure, read_json, signature, wait_memory, write_json


def effective_affine(image, case):
    affine = image.affine.copy()
    if case["dataset"] == "imagechd":
        direction = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        affine[:3, :3] = direction * np.asarray(case["spacing_xyz"])
    return affine


@contextmanager
def ascii_label(path, directory):
    if os.name != "nt":
        yield Path(path)
        return
    if Path(path).stat().st_dev != directory.stat().st_dev:
        raise ValueError("Windows标签ASCII入口要求源文件与工作目录在同一文件系统")
    link = directory / "source-label.seg.nrrd"
    if link.exists():
        raise FileExistsError("本例存在未清理的标签入口，请先核对本任务状态")
    os.link(path, link)
    try:
        yield link
    finally:
        if not os.path.samefile(path, link):
            raise RuntimeError("标签临时入口已变化，拒绝删除")
        link.unlink()


def load_localization(case, image, directory):
    if case["dataset"] == "imagechd":
        label = nib.load(case["label"])
        if label.shape != image.shape or not np.allclose(label.affine, image.affine, atol=1e-5, rtol=0):
            raise ValueError("ImageCHD影像与标签网格不匹配")
        data = np.asanyarray(label.dataobj)
        return data == 1, data == 6, {"LV": {"name": "LV", "value": 1, "layer": 0},
                                     "AO": {"name": "AO", "value": 6, "layer": 0}}
    with ascii_label(case["label"], directory) as link:
        label = sitk.ReadImage(str(link))
    if label.GetDimension() != 3:
        raise ValueError("定位输入需要三维标量或三维多分量Slicer分割")
    matrix = np.asarray(label.GetDirection()).reshape(3, 3) @ np.diag(label.GetSpacing())
    affine = np.eye(4)
    affine[:3, :3] = np.diag([-1, -1, 1]) @ matrix
    affine[:3, 3] = np.diag([-1, -1, 1]) @ label.GetOrigin()
    if tuple(label.GetSize()) != image.shape or not np.allclose(affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("AAOCA粗分割与原CT的数组或物理网格不一致")
    structures = {}
    accepted = {"LV": {"heart_ventricle_left", "ventricle_left", "左心室"},
                "AO": {"aorta", "主动脉"}}
    data = sitk.GetArrayViewFromImage(label)
    for key in label.GetMetaDataKeys():
        match = re.fullmatch(r"Segment(\d+)_Name", key)
        if match is None:
            continue
        prefix = "Segment" + match.group(1)
        name = label.GetMetaData(key)
        for kind, names in accepted.items():
            if name in names:
                if kind in structures:
                    raise ValueError("定位结构名称重复，需要明确指定标签来源")
                value = int(label.GetMetaData(prefix + "_LabelValue"))
                layer = int(label.GetMetaData(prefix + "_Layer"))
                values = data[..., layer] if label.GetNumberOfComponentsPerPixel() > 1 else data
                structures[kind] = (np.transpose(values == value, (2, 1, 0)),
                                    {"name": name, "value": value, "layer": layer})
    if set(structures) != {"LV", "AO"}:
        raise ValueError("既有粗分割缺少明确命名的左室腔或主动脉")
    return structures["LV"][0], structures["AO"][0], {k: v[1] for k, v in structures.items()}


def center_of_interest(lv, aorta):
    if lv.shape != aorta.shape or not lv.any() or not aorta.any():
        raise ValueError("定位需要同网格且非空的LV和主动脉")
    # 只在两个结构并集外扩两层的区域内计算，保留作者原有定位定义。
    coordinates = np.nonzero(lv | aorta)
    start = np.asarray([max(int(axis.min()) - 2, 0) for axis in coordinates])
    stop = [min(int(axis.max()) + 3, lv.shape[i]) for i, axis in enumerate(coordinates)]
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    dilated_lv = ndimage.binary_dilation(lv[slices])
    dilated_ao = ndimage.binary_dilation(aorta[slices])
    overlap = dilated_lv & dilated_ao
    if overlap.any():
        # 原作者在全局坐标求均值后取整；局部坐标须先加回起点。
        center = np.round(np.asarray(np.where(overlap)).mean(axis=1) + start).astype(int)
        mode = "dilated_intersection_mean"
    else:
        distance = ndimage.distance_transform_edt(~dilated_ao)
        distance[~dilated_lv] = np.inf
        center = np.asarray(np.unravel_index(np.argmin(distance), distance.shape)) + start
        mode = "nearest_point_on_dilated_lv"
    return center, mode


def author_reference_center(lv, aorta):
    left, right = ndimage.binary_dilation(lv), ndimage.binary_dilation(aorta)
    overlap = left & right
    if overlap.any():
        return np.round(np.asarray(np.where(overlap)).mean(axis=1)).astype(int)
    distance = ndimage.distance_transform_edt(~right)
    distance[~left] = np.inf
    return np.asarray(np.unravel_index(np.argmin(distance), distance.shape))


def prepare_case(case, config, directory, verify_center=False):
    directory = Path(directory).resolve()
    work = (config["project_root"] / "work").resolve()
    if not directory.is_relative_to(work) or directory == work:
        raise ValueError("裁剪输出必须位于本工程work的专用子目录")
    for key in ("image", "label"):
        if signature(case[key]) != case["source_stat"][key]:
            raise ValueError("冻结后输入影像或标签的文件状态已变化")
    method = {"author_release": RELEASE, "roi_mm": config["roi_mm"],
              "offset_index_mm": config["offset_index_mm"],
              "imagechd_orientation_policy": config["imagechd_orientation_policy"]}
    key = identity({"case": case, "method": method})
    receipt, output = directory / "prepared.json", directory / "crop.nii.gz"
    if receipt.exists():
        previous = read_json(receipt)
        if previous["identity"] != key or signature(output) != previous["output_stat"]:
            raise ValueError("已有裁剪与当前输入、方法或输出文件状态不一致")
        return {**previous, "reused": True}
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("裁剪目录已有未完成内容，请核对后清理本任务副产物")
    image = nib.load(case["image"])
    estimate = int(np.prod(image.shape)) * 12 + 512 * 2**20
    wait_memory(config, int(estimate * 1.5))
    directory.mkdir(parents=True, exist_ok=True)
    check_work(config)
    timings = {}
    event("prepare_started", case_id=case["case_id"], localization=case["localization"])
    with measure() as measured:
        tic = perf_counter()
        lv, aorta, structures = load_localization(case, image, directory)
        timings["label_load_seconds"] = perf_counter() - tic
        tic = perf_counter()
        center, center_mode = center_of_interest(lv, aorta)
        timings["localization_seconds"] = perf_counter() - tic
        center_check = None
        if verify_center:
            bounds = tuple(slice(max(int(x) - 32, 0), min(int(x) + 32, lv.shape[i]))
                           for i, x in enumerate(center))
            local_lv, local_ao = lv[bounds], aorta[bounds]
            actual, _ = center_of_interest(local_lv, local_ao)
            expected = author_reference_center(local_lv, local_ao)
            if not np.array_equal(actual, expected):
                raise AssertionError("真实局部标签上的定位与作者直接算法不一致")
            center_check = {"shape": list(local_lv.shape), "equal": True,
                            "center": actual.tolist(), "scope": "real_local_label_region"}
        affine = effective_affine(image, case)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        offset = np.asarray([int(mm / step) for mm, step in zip(config["offset_index_mm"], spacing)])
        size = np.asarray([min(int(mm / step), image.shape[i])
                           for i, (mm, step) in enumerate(zip(config["roi_mm"], spacing))])
        shifted = center + offset
        start = np.asarray([max(0, min(int(shifted[i] - size[i] / 2), image.shape[i] - int(size[i])))
                            for i in range(3)])
        slices = tuple(slice(int(a), int(a + b)) for a, b in zip(start, size))
        tic = perf_counter()
        crop = np.asarray(image.dataobj[slices], dtype=np.float32)
        timings["image_crop_read_seconds"] = perf_counter() - tic
        if not np.isfinite(crop).all() or tuple(crop.shape) != tuple(size):
            raise ValueError("裁剪数组包含非有限值或尺寸不一致")
        transform = np.eye(4)
        transform[:3, 3] = start
        cropped_affine = affine @ transform
        if not np.allclose(nib.affines.apply_affine(cropped_affine, [0, 0, 0]),
                           nib.affines.apply_affine(affine, start), atol=1e-6, rtol=0):
            raise AssertionError("裁剪起点的物理坐标不一致")
        pending = directory / "crop.pending.nii.gz"
        if crop.nbytes * 2 > config["max_work_bytes"] or psutil.disk_usage(str(directory)).free < crop.nbytes * 2:
            raise OSError("裁剪文件磁盘预算不足")
        tic = perf_counter()
        result = nib.Nifti1Image(crop, cropped_affine)
        result.header.set_xyzt_units("mm")
        nib.save(result, pending)
        pending.replace(output)
        timings["save_seconds"] = perf_counter() - tic
        record = {"identity": key, "case_id": case["case_id"], "localization": case["localization"],
                  "author_release": RELEASE, "reused": False, "structures": structures,
                  "source_shape": list(image.shape), "source_header_spacing": list(map(float, image.header.get_zooms())),
                  "effective_spacing_xyz": spacing.tolist(), "orientation": list(nib.aff2axcodes(affine)),
                  "spacing_evidence": case.get("spacing_evidence", "native_CT_geometry"),
                  "orientation_policy": method["imagechd_orientation_policy"] if case["dataset"] == "imagechd" else "native_CT_header",
                  "center_index": center.tolist(), "center_mode": center_mode,
                  "offset_index_voxels": offset.tolist(), "offset_RAS_mm": (affine[:3, :3] @ offset).tolist(),
                  "crop_start": start.tolist(), "crop_shape": size.tolist(),
                  "requested_roi_mm": config["roi_mm"], "actual_roi_mm": (size * spacing).tolist(),
                  "source_FOV_smaller_than_ROI": bool(np.any(np.asarray(image.shape) * spacing < config["roi_mm"])),
                  "crop_affine": cropped_affine.tolist(), "center_check": center_check,
                  "timing": timings, "training_updates": 0, "model_prediction": False,
                  "output_stat": signature(output)}
    record["measurement"] = measured
    record["work_inventory"] = check_work(config)
    write_json(receipt, record)
    event("prepare_completed", case_id=case["case_id"], seconds=measured["seconds"],
          crop_shape=size.tolist(), actual_roi_mm=record["actual_roi_mm"])
    return record
