from contextlib import contextmanager
import os
from pathlib import Path
import re
import threading
from time import perf_counter

import nibabel as nib
import numpy as np
import psutil
from scipy import ndimage
import SimpleITK as sitk

from .records import RELEASE, admit_stage, check_work, event, identity, measure, read_json, signature, write_json


LABEL_PATH_LOCK = threading.RLock()


def effective_affine(image, case):
    affine = image.affine.copy()
    if case["dataset"] == "chd1282":
        binding = case.get("spatial_binding")
        if not binding or binding["array_alignment"] != "shared_index_research_policy" or not binding["reference"]:
            raise ValueError("CHD1282缺少已有方向/数组配对依据，须先绑定整理成果")
        direction = np.asarray(binding["direction_matrix_RAS"], dtype=float)
        spacing = np.asarray(case["spacing_xyz"], dtype=float)
        if (direction.shape != (3, 3) or not np.isfinite(direction).all()
                or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)
                or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
            raise ValueError("CHD1282方向矩阵和作者间距无效")
        if binding["origin_policy"] != "local_zero" or binding["intensity_policy"] not in {"source_values_assumed_HU", "verified_HU"}:
            raise ValueError("CHD1282须明确局部原点与真实强度依据范围")
        affine = np.eye(4)
        affine[:3, :3] = direction * spacing
    if case["dataset"] == "imagechd":
        direction = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        affine[:3, :3] = direction * np.asarray(case["spacing_xyz"])
    return affine


@contextmanager
def ascii_label(path, directory):
    if os.name != "nt":
        yield Path(path)
        return
    path = Path(path).resolve(strict=True)
    if not path.name.isascii():
        raise ValueError("Windows原标签文件名须为ASCII；父目录可含中文")
    # SITK的Windows文件接口无法处理中文父目录；只在同步读取期间改变工作目录。
    with LABEL_PATH_LOCK:
        previous = Path.cwd()
        os.chdir(path.parent)
        try:
            yield Path(path.name)
        finally:
            os.chdir(previous)


def load_localization(case, image, directory, allow_missing=False):
    if case["dataset"] in {"imagechd", "chd1282"}:
        label = nib.load(case["label"])
        if label.shape != image.shape:
            raise ValueError("影像与解剖标签数组尺寸不匹配")
        if case["dataset"] == "imagechd" and not np.allclose(label.affine, image.affine, atol=1e-5, rtol=0):
            raise ValueError("ImageCHD影像与标签网格不匹配")
        if case["dataset"] == "chd1282":
            effective_affine(image, case)
        data = np.asanyarray(label.dataobj)
        ao_value = 5 if case["dataset"] == "chd1282" else 6
        return data == 1, data == ao_value, {"LV": {"name": "LV", "value": 1, "layer": 0},
                                             "AO": {"name": "AO", "value": ao_value, "layer": 0}}
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
        if allow_missing:
            return (structures.get("LV", (None,))[0], structures.get("AO", (None,))[0],
                    {k: v[1] for k, v in structures.items()})
        raise ValueError("既有粗分割缺少明确命名的左室腔或主动脉")
    return structures["LV"][0], structures["AO"][0], {k: v[1] for k, v in structures.items()}


def center_of_interest(lv, aorta, config=None, memory_plan=None):
    if memory_plan is None:
        memory_plan = []
    if lv.shape != aorta.shape or not lv.any() or not aorta.any():
        raise ValueError("定位需要同网格且非空的LV和主动脉")
    # 只在两个结构并集外扩两层的区域内计算，保留作者原有定位定义。
    if config is not None:
        memory_plan.append(admit_stage(config, "localization_bbox", {"union_bool": int(lv.size)}))
    union = lv | aorta
    bounds = ndimage.find_objects(union, max_label=1)[0]
    del union
    start = np.asarray([max(axis.start - 2, 0) for axis in bounds])
    stop = [min(axis.stop + 2, lv.shape[i]) for i, axis in enumerate(bounds)]
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    local_voxels = int(np.prod(np.asarray(stop) - start))
    if config is not None:
        memory_plan.append(admit_stage(config, "localization_dilation", {
            "two_dilated_masks_and_overlap": 3 * local_voxels,
            "morphology_workspace_bound": 16 * local_voxels}))
    dilated_lv = ndimage.binary_dilation(lv[slices])
    dilated_ao = ndimage.binary_dilation(aorta[slices])
    overlap = dilated_lv & dilated_ao
    if overlap.any():
        if config is not None:
            memory_plan.append(admit_stage(config, "intersection_coordinates", {
                "nonzero_and_coordinate_stack": int(np.count_nonzero(overlap)) * 6 * np.dtype(np.intp).itemsize}))
        # 原作者在全局坐标求均值后取整；局部坐标须先加回起点。
        center = np.round(np.asarray(np.where(overlap)).mean(axis=1) + start).astype(int)
        mode = "dilated_intersection_mean"
    else:
        if config is not None:
            memory_plan.append(admit_stage(config, "local_distance_transform", {
                "edt_indices_float64_vectors_and_reduction": 64 * local_voxels}))
        distance = ndimage.distance_transform_edt(~dilated_ao)
        distance[~dilated_lv] = np.inf
        center = np.asarray(np.unravel_index(np.argmin(distance), distance.shape)) + start
        mode = "nearest_point_on_dilated_lv"
    return center, mode


def localization_memory_plan(case, image, directory):
    voxels = int(np.prod(image.shape))
    if case["dataset"] in {"imagechd", "chd1282"}:
        label = nib.load(case["label"])
        itemsize = label.get_data_dtype().itemsize
        if label.dataobj.slope != 1 or label.dataobj.inter != 0:
            itemsize = max(itemsize, 8)
        label_bytes = int(np.prod(label.shape)) * itemsize
        components = 1
    else:
        with ascii_label(case["label"], directory) as path:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.ReadImageInformation()
            pixel_sizes = {sitk.sitkUInt8: 1, sitk.sitkInt8: 1, sitk.sitkVectorUInt8: 1, sitk.sitkVectorInt8: 1,
                           sitk.sitkUInt16: 2, sitk.sitkInt16: 2, sitk.sitkVectorUInt16: 2, sitk.sitkVectorInt16: 2,
                           sitk.sitkUInt32: 4, sitk.sitkInt32: 4, sitk.sitkVectorUInt32: 4, sitk.sitkVectorInt32: 4,
                           sitk.sitkFloat32: 4, sitk.sitkVectorFloat32: 4, sitk.sitkFloat64: 8, sitk.sitkVectorFloat64: 8}
            if reader.GetPixelID() not in pixel_sizes:
                raise ValueError("标签像素类型尚无明确内存核算")
            components = reader.GetNumberOfComponents()
            label_bytes = int(np.prod(reader.GetSize())) * components * pixel_sizes[reader.GetPixelID()]
    return {"label_buffer": label_bytes, "decoder_buffer_bound": label_bytes,
            "two_boolean_masks": 2 * voxels}, components


def author_reference_center(lv, aorta):
    left, right = ndimage.binary_dilation(lv), ndimage.binary_dilation(aorta)
    overlap = left & right
    if overlap.any():
        return np.round(np.asarray(np.where(overlap)).mean(axis=1)).astype(int)
    distance = ndimage.distance_transform_edt(~right)
    distance[~left] = np.inf
    return np.asarray(np.unravel_index(np.argmin(distance), distance.shape))


def localization_status(lv, aorta):
    if lv is not None and aorta is not None and lv.shape != aorta.shape:
        raise ValueError("左心室与主动脉定位网格不一致")
    counts = {"LV": None if lv is None else int(np.count_nonzero(lv)),
              "AO": None if aorta is None else int(np.count_nonzero(aorta))}
    reason = ("missing_localization_structures" if None in counts.values() else
              "empty_localization_structures" if 0 in counts.values() else None)
    return {"reason": reason, "voxel_counts": counts}


def crop_in_memory(case, config, directory, verify_center=False, allow_missing=False):
    directory = Path(directory).resolve()
    for key in ("image", "label"):
        if signature(case[key]) != case["source_stat"][key]:
            raise ValueError("冻结后输入影像或标签的文件状态已变化")
    image = nib.load(case["image"])
    # 在读取标签和CT像素前核对新增来源的方向依据。
    effective_affine(image, case)
    directory.mkdir(parents=True, exist_ok=True)
    check_work(config)
    allocations, components = localization_memory_plan(case, image, directory)
    memory_plan = [admit_stage(config, "label_read_and_masks", allocations)]
    timings = {}
    event("prepare_started", case_id=case["case_id"], localization=case["localization"])
    with measure() as measured:
        tic = perf_counter()
        lv, aorta, structures = load_localization(case, image, directory, allow_missing=allow_missing)
        timings["label_load_seconds"] = perf_counter() - tic
        status = localization_status(lv, aorta)
        if status["reason"] is not None:
            if allow_missing:
                return None, {**status, "structures": structures}
            raise ValueError("定位结构缺失或为空：" + str(status["voxel_counts"]))
        tic = perf_counter()
        center, center_mode = center_of_interest(lv, aorta, config, memory_plan)
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
        physical = config.get("classification_recipe") == "canonical_ras_v1"
        target_affine = None
        if physical:
            from .records import CLASSIFICATION_TARGET
            offset_world = np.asarray(config.get("offset_ras_mm", [10, 0, 10]), dtype=float)
            if offset_world.shape != (3,) or not np.isfinite(offset_world).all():
                raise ValueError("统一方向偏移需为三个有限RAS毫米值")
            roi = np.asarray(config["roi_mm"], dtype=float)
            target_spacing = roi / np.asarray(CLASSIFICATION_TARGET)
            target_affine = np.diag([*target_spacing, 1.0])
            target_affine[:3, 3] = nib.affines.apply_affine(affine, center) + offset_world - roi / 2 + target_spacing / 2
            from itertools import product
            corners = np.asarray(list(product(*[(0, n - 1) for n in CLASSIFICATION_TARGET])))
            source_corners = nib.affines.apply_affine(np.linalg.inv(affine) @ target_affine, corners)
            start = np.maximum(np.floor(source_corners.min(axis=0)).astype(int) - 3, 0)
            stop = np.minimum(np.ceil(source_corners.max(axis=0)).astype(int) + 4, image.shape)
            size = stop - start
            if np.any(size < 2):
                raise ValueError("统一物理ROI与原CT没有足够交集")
            offset = np.linalg.solve(affine[:3, :3], offset_world)
        else:
            offset = np.asarray([int(mm / step) for mm, step in zip(config["offset_index_mm"], spacing)])
            size = np.asarray([min(int(mm / step), image.shape[i])
                               for i, (mm, step) in enumerate(zip(config["roi_mm"], spacing))])
            shifted = center + offset
            start = np.asarray([max(0, min(int(shifted[i] - size[i] / 2), image.shape[i] - int(size[i])))
                                for i in range(3)])
        slices = tuple(slice(int(a), int(a + b)) for a, b in zip(start, size))
        memory_plan.append(admit_stage(config, "ct_roi_read", {
            "source_xy_slab_buffer_bound": int(np.prod(image.shape[:2])) * int(size[2]) * image.get_data_dtype().itemsize,
            "roi_float32_and_conversion": int(np.prod(size)) * 8}))
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
        record = {"case_id": case["case_id"], "localization": case["localization"],
                  "author_release": RELEASE, "reused": False, "structures": structures,
                  "source_shape": list(image.shape), "source_header_spacing": list(map(float, image.header.get_zooms())),
                  "effective_spacing_xyz": spacing.tolist(), "orientation": list(nib.aff2axcodes(affine)),
                  "spacing_evidence": case.get("spacing_evidence", "native_CT_geometry"),
                  "orientation_policy": config["imagechd_orientation_policy"] if case["dataset"] == "imagechd" else "native_CT_header",
                  "source_header_affine": image.affine.tolist(), "effective_source_affine": affine.tolist(),
                  "spatial_binding": case.get("spatial_binding"),
                  "coordinate_origin_scope": "arbitrary_local_origin" if case["dataset"] == "chd1282" else "source_header_origin",
                  "center_index": center.tolist(), "center_mode": center_mode,
                  "offset_index_voxels": offset.tolist(), "offset_RAS_mm": (affine[:3, :3] @ offset).tolist(),
                  "target_affine": target_affine.tolist() if physical else None,
                  "physical_ROI_partly_outside_source": bool(np.any(source_corners < 0) or np.any(source_corners > np.asarray(image.shape) - 1)) if physical else None,
                  "crop_start": start.tolist(), "crop_shape": size.tolist(),
                  "requested_roi_mm": config["roi_mm"], "actual_roi_mm": (size * spacing).tolist(),
                  "source_FOV_smaller_than_ROI": bool(np.any(np.asarray(image.shape) * spacing < config["roi_mm"])),
                  "crop_affine": cropped_affine.tolist(), "center_check": center_check,
                  "timing": timings, "memory_estimates": memory_plan, "label_components": components,
                  "training_updates": 0, "model_prediction": False}
    record["measurement"] = measured
    return crop, record


def prepare_case(case, config, directory, verify_center=False):
    directory = Path(directory).resolve()
    work = (config["project_root"] / "work").resolve()
    if not directory.is_relative_to(work) or directory == work:
        raise ValueError("裁剪输出必须位于本工程work的专用子目录")
    method = {"author_release": RELEASE, "roi_mm": config["roi_mm"],
              "offset_index_mm": config["offset_index_mm"],
              "imagechd_orientation_policy": config["imagechd_orientation_policy"]}
    key = identity({"case": case, "method": method})
    receipt, output = directory / "prepared.json", directory / "crop.nii.gz"
    for name in ("image", "label"):
        if signature(case[name]) != case["source_stat"][name]:
            raise ValueError("冻结后输入影像或标签的文件状态已变化")
    if receipt.exists():
        previous = read_json(receipt)
        if previous["identity"] != key or signature(output) != previous["output_stat"]:
            raise ValueError("已有裁剪与当前输入、方法或输出文件状态不一致")
        return {**previous, "reused": True}
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("裁剪目录已有未完成内容，请核对后清理本任务副产物")
    crop, record = crop_in_memory(case, config, directory, verify_center)
    if crop.nbytes * 2 > config["max_work_bytes"] or psutil.disk_usage(str(directory)).free < crop.nbytes * 2:
        raise OSError("裁剪文件磁盘预算不足")
    tic = perf_counter()
    result = nib.Nifti1Image(crop, np.asarray(record["crop_affine"]))
    result.header.set_xyzt_units("mm")
    pending = directory / "crop.pending.nii.gz"
    nib.save(result, pending)
    pending.replace(output)
    record["timing"]["save_seconds"] = perf_counter() - tic
    record.update(identity=key, output_stat=signature(output))
    record["work_inventory"] = check_work(config)
    write_json(receipt, record)
    event("prepare_completed", case_id=case["case_id"], seconds=record["measurement"]["seconds"],
          crop_shape=record["crop_shape"], actual_roi_mm=record["actual_roi_mm"])
    return record
