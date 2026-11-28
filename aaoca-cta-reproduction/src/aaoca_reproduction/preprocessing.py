from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import psutil
import torch
import torchio as tio

from .prepare import crop_in_memory
from .exclusions import read_exclusion, save_exclusion
from .records import CLASSIFICATION_TARGET, RELEASE, admit_stage, event, identity, measure, read_json, signature, write_json
from .resources import configure_cpu, memory_guard


TARGET = CLASSIFICATION_TARGET
RECIPE = "author_native_index"
CANONICAL_RECIPE = "canonical_ras_v1"


def method(config):
    if version("torchio") != "0.19.6":
        raise RuntimeError("作者重采样要求真实TorchIO 0.19.6")
    if config.get("classification_recipe") not in {RECIPE, CANONICAL_RECIPE}:
        raise ValueError("需要明确author_native_index或canonical_ras_v1配方")
    if config["roi_mm"] != [80, 80, 60] or config["offset_index_mm"] != [-10, 10, 10]:
        raise ValueError("作者配方要求固定裁剪毫米尺寸与源索引偏移")
    result = {"schema": 1, "author_release": RELEASE, "recipe": RECIPE, "target_xyz": list(TARGET),
            "roi_mm": config["roi_mm"], "offset_index_mm": config["offset_index_mm"],
            "imagechd_orientation_policy": config["imagechd_orientation_policy"],
            "torchio": version("torchio"), "interpolation": "bspline", "resample_input_grid": "unit_identity",
            "normalization": "floor((clip(HU,-1024,1024)+1024)/2048*255)/255",
            "network_axes": "BCZXY", "training_storage": "float16_XYZ", "inference_storage": "float32_XYZ",
            "author_reference_direction_confirmed": False, "formal_mixed_classification_ready": False}
    if config["classification_recipe"] == CANONICAL_RECIPE:
        result.update(recipe=CANONICAL_RECIPE, resample_input_grid="physical_RAS",
                      offset_ras_mm=config.get("offset_ras_mm", [10, 0, 10]),
                      output_direction="RAS_identity", outside_HU=-1024,
                      reproduction_scope="local_adaptation_author_world_direction_unknown")
    return result


def network_tensor(array):
    if tuple(array.shape) != TARGET or not np.isfinite(array).all():
        raise ValueError("分类输入必须为有限215×215×85数组")
    if array.min() < 0 or array.max() > 1:
        raise ValueError("分类输入需要已完成HU量化，数值范围0到1")
    return torch.as_tensor(np.asarray(array, dtype=np.float32)).permute(2, 0, 1)[None, None]


def preprocessing_identity(case, recipe):
    source = {name: case.get(name) for name in ("case_id", "dataset", "image", "label", "source_stat",
                                                "spacing_xyz", "spacing_evidence", "localization")}
    if "spatial_binding" in case:
        source["spatial_binding"] = case["spatial_binding"]
    return identity({"source": source, "method": recipe})


def array_statistics(array):
    return {"mean": float(array.mean()), "std": float(array.std()),
            "q05_q50_q95": np.quantile(array, [0.05, 0.5, 0.95]).tolist(),
            "zero_fraction": float(np.count_nonzero(array == 0) / array.size)}


def resample_crop(crop, crop_affine, verify_author=False, reference_affine=None):
    if version("torchio") != "0.19.6":
        raise RuntimeError("作者重采样要求真实TorchIO 0.19.6")
    if crop.ndim != 3 or min(crop.shape) < 2 or not np.isfinite(crop).all():
        raise ValueError("分类裁剪需要三维、各轴至少两体素且数值有限")
    if reference_affine is not None:
        if verify_author:
            raise ValueError("物理RAS适配不能登记为作者裸数组逐值复现")
        target_affine = np.asarray(reference_affine, dtype=float)
        image = tio.ScalarImage(tensor=(crop + 1024)[None], affine=crop_affine)
        sampled = tio.Resample(target=(TARGET, target_affine), image_interpolation="bspline")(image)
        data = sampled.data[0].numpy() - 1024
        if tuple(data.shape) != TARGET or not np.isfinite(data).all() or not np.allclose(sampled.affine, target_affine):
            raise ValueError("实际物理重采样网格或数组不符合RAS合同")
        normalized = np.asarray(np.floor((np.clip(data, -1024, 1024) + 1024) / 2048 * 255) / 255, dtype=np.float32)
        direction = target_affine[:3, :3] / np.linalg.norm(target_affine[:3, :3], axis=0)
        if not np.allclose(direction, np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("物理重采样输出必须为正交RAS方向")
        permutation = np.eye(4)[:, [2, 0, 1, 3]]
        corners = np.asarray(list(product(*[(0, x - 1) for x in TARGET])))
        world = nib.affines.apply_affine(target_affine, corners)
        recovered = nib.affines.apply_affine(np.linalg.inv(target_affine), world)
        probe = np.asarray([7, 13, 19])
        probe_zxy = probe[[2, 0, 1]]
        tensor = network_tensor(normalized)
        if not np.allclose(recovered, corners, atol=1e-6, rtol=0):
            raise AssertionError("RAS角点世界坐标逆变换不一致")
        if tensor[0, 0, *probe_zxy].item() != float(normalized[tuple(probe)]):
            raise AssertionError("RAS真实数组的ZXY转置不一致")
        if not np.allclose(nib.affines.apply_affine(target_affine, probe),
                           nib.affines.apply_affine(target_affine @ permutation, probe_zxy), atol=1e-6, rtol=0):
            raise AssertionError("RAS网络坐标到世界坐标不一致")
        return normalized, {"target_affine": target_affine.tolist(), "network_affine": (target_affine @ permutation).tolist(),
            "target_axis_codes": list(nib.aff2axcodes(target_affine)), "target_shape_xyz": list(TARGET),
            "network_shape": list(network_tensor(normalized).shape), "round_trip_checked": True,
            "direction_RAS": direction.tolist(), "author_check": None, "corner_voxel_centers_RAS_mm": world.tolist(),
            "intensity_range": [float(normalized.min()), float(normalized.max())]}
    ratio = np.asarray(crop.shape, dtype=float) / np.asarray(TARGET)
    unit_image = tio.ScalarImage(tensor=np.asarray(crop, dtype=np.float32)[None], affine=np.eye(4))
    sampled = tio.Resample(target=tuple(ratio), image_interpolation="bspline")(unit_image)
    expected_grid = np.eye(4)
    expected_grid[:3, :3] = np.diag(ratio)
    expected_grid[:3, 3] = (ratio - 1) / 2
    if not np.allclose(sampled.affine, expected_grid, atol=1e-6, rtol=0):
        raise AssertionError("TorchIO实际单位采样格与固定作者约定不一致")
    data = sampled.data[0].numpy()
    actual_shape = list(data.shape)
    data = data[:TARGET[0], :TARGET[1], :TARGET[2]]
    if tuple(data.shape) != TARGET or not np.isfinite(data).all():
        raise ValueError("重采样不足目标尺寸或包含非有限值")
    normalized = np.floor((np.clip(data, -1024, 1024) + 1024) / 2048 * 255) / 255
    normalized = np.asarray(normalized, dtype=np.float32)
    author_check = None
    if verify_author:
        # 固定作者utils.py的裸数组重采样和强度步骤，复用本例内存裁剪。
        original_pixel = crop.shape
        target_ratio = [(original_pixel[i] / TARGET[i]) for i in range(len(original_pixel))]
        transform = tio.Resample(target=target_ratio, image_interpolation="bspline")
        reference = transform(np.expand_dims(crop, axis=0))[0]
        reference = reference[:int(TARGET[0]), :int(TARGET[1]), :int(TARGET[2])]
        reference = np.clip(reference, a_min=-1024, a_max=1024)
        reference = np.array((reference + 1024) / 2048 * 255).astype(int) / 255
        equal = np.array_equal(normalized, reference.astype(np.float32))
        if not equal:
            raise AssertionError("真实裁剪上的分类数组与固定作者计算不一致")
        author_check = {"equal_after_float32": equal, "scope": "same_real_in_memory_crop",
                        "function": "utils.py torch_io_resize and normalize_and_resample intensity steps",
                        "author_release": RELEASE}
    target_affine = np.asarray(crop_affine) @ sampled.affine
    permutation = np.eye(4)[:, [2, 0, 1, 3]]
    network_affine = target_affine @ permutation
    corners = np.asarray(list(product(*[(0, x - 1) for x in TARGET])))
    world = nib.affines.apply_affine(target_affine, corners)
    crop_corners = np.asarray(list(product(*[(0, x - 1) for x in crop.shape])))
    recovered = nib.affines.apply_affine(np.linalg.inv(target_affine), world)
    if not np.allclose(recovered, corners, atol=1e-6, rtol=0):
        raise AssertionError("分类角点世界坐标逆变换不一致")
    tensor = network_tensor(normalized)
    probe = np.asarray([7, 13, 19])
    probe_zxy = probe[[2, 0, 1]]
    if tensor[0, 0, *probe_zxy].item() != float(normalized[tuple(probe)]):
        raise AssertionError("真实数组非对称点的ZXY转置不一致")
    if not np.allclose(nib.affines.apply_affine(target_affine, probe),
                       nib.affines.apply_affine(network_affine, probe_zxy), atol=1e-6, rtol=0):
        raise AssertionError("ZXY到世界坐标的映射不一致")
    levels = normalized.astype(np.float64) * 255
    if not np.allclose(levels, np.rint(levels), atol=2e-5, rtol=0):
        raise AssertionError("HU量化没有保持256级网格")
    geometry = {"ratio_xyz": ratio.tolist(), "torchio_unit_affine": sampled.affine.tolist(),
                "torchio_output_shape": actual_shape, "slice_start_xyz": [0, 0, 0],
                "target_affine": target_affine.tolist(), "network_ZXY_to_XYZ": permutation.tolist(),
                "network_affine": network_affine.tolist(), "target_axis_codes": list(nib.aff2axcodes(target_affine)),
                "target_shape_xyz": list(TARGET), "network_shape": list(tensor.shape),
                "corner_voxel_centers_RAS_mm": world.tolist(), "round_trip_checked": True,
                "crop_corner_voxel_centers_RAS_mm": nib.affines.apply_affine(crop_affine, crop_corners).tolist(),
                "probe_xyz": probe.tolist(), "probe_zxy": probe_zxy.tolist(),
                "probe_RAS_mm": nib.affines.apply_affine(target_affine, probe).tolist(),
                "intensity_range": [float(normalized.min()), float(normalized.max())], "author_check": author_check}
    return normalized, geometry


def prepare_classification(case, config, directory, verify_author=False):
    directory = Path(directory).resolve()
    root = Path(config["data_root"]).resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ValueError("正式分类预处理必须位于配置data_root的专用子目录")
    recipe = method(config)
    key = preprocessing_identity(case, recipe)
    excluded = read_exclusion(case, config, directory)
    if excluded is not None:
        if excluded["identity"] != key:
            raise ValueError("定位排除记录的预处理方法身份已变化，需要重新核对该例")
        return {**excluded, "reused": True}
    receipt = directory / "classification.json"
    if not Path(case["label"]).exists() and not receipt.exists():
        if signature(case["image"]) != case["source_stat"]["image"]:
            raise ValueError("冻结后选中影像状态已变化")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("缺失分割的病例目录包含未知或未完成文件")
        return save_exclusion(case, config, directory, "missing_label_file", {}, key)
    for name in ("image", "label"):
        if signature(case[name]) != case["source_stat"][name]:
            raise ValueError("冻结后选中输入文件状态已变化")
    if receipt.exists():
        previous = read_json(receipt)
        if previous["identity"] != key:
            raise ValueError("已有预处理身份与当前输入或方法不一致")
        for name, stamp in previous["outputs"].items():
            if signature(directory / name) != stamp:
                raise ValueError("已有预处理输出文件状态已变化")
        return {**previous, "reused": True}
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("分类目录存在未完成文件，须核对本任务状态")
    directory.mkdir(parents=True, exist_ok=True)
    timings = {}
    applied_resources = configure_cpu(config)
    with measure() as measured, memory_guard(config) as guarded:
        tic = perf_counter()
        crop, crop_record = crop_in_memory(case, config, directory, verify_center=verify_author, allow_missing=True)
        timings["crop_seconds"] = perf_counter() - tic
        if crop is None:
            return save_exclusion(case, config, directory, crop_record["reason"], crop_record, key)
        memory_estimate = admit_stage(config, "torchio_resample_and_quantization", {
            "crop_and_bspline_working_buffers": int(crop.nbytes) * 8,
            "target_buffers_and_float64_geometry_checks": int(np.prod(TARGET)) * 80,
            "author_reference_buffers": int(np.prod(TARGET)) * 24 if verify_author else 0})
        tic = perf_counter()
        normalized, geometry = resample_crop(crop, np.asarray(crop_record["crop_affine"]), verify_author,
                                             reference_affine=crop_record.get("target_affine"))
        statistics = array_statistics(normalized)
        timings["resample_normalize_seconds"] = perf_counter() - tic
        output_bytes = normalized.nbytes * 3
        if output_bytes > config["max_work_bytes"] or psutil.disk_usage(str(directory)).free < output_bytes:
            raise OSError("分类预处理文件预算不足")
        tic = perf_counter()
        image = nib.Nifti1Image(normalized, np.asarray(geometry["target_affine"]))
        image.header.set_xyzt_units("mm")
        pending = directory / "normalized.pending.nii.gz"
        nib.save(image, pending)
        output = directory / "normalized.nii.gz"
        pending.replace(output)
        training = directory / "training_xyz.npy"
        pending_training = directory / "training_xyz.pending.npy"
        np.save(pending_training, normalized.astype(np.float16), allow_pickle=False)
        pending_training.replace(training)
        timings["save_seconds"] = perf_counter() - tic
    record = {"identity": key, "case_id": case["case_id"], "method": recipe, "stage": "normalized_xyz",
              "input_statistics": statistics,
              "reused": False, "crop": crop_record, "geometry": geometry, "timing": timings,
              "measurement": measured, "resources": applied_resources, "memory_guard": guarded,
              "resample_memory_estimate": memory_estimate,
              "training_updates": 0, "model_prediction": False,
              "outputs": {p.name: signature(p) for p in (output, training)}}
    write_json(receipt, record)
    event("classification_prepared", case_id=case["case_id"], seconds=measured["seconds"], recipe=recipe["recipe"])
    return record
