from copy import deepcopy
from time import perf_counter

import nibabel as nib
import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import psutil

from ..artifacts.store import file_digest
from ..observability.events import Stage
from ..observability.events import event
from ..training.host import host_capacity


def bounded_locations(seg, classes_or_regions, seed=1234, verbose=False, min_num_samples=10000, min_percent_coverage=.01):
    flat, block, result = seg[0].ravel(), 1_000_000, {}
    rng = np.random.default_rng(seed)
    for value in classes_or_regions:
        labels = tuple(value) if isinstance(value, (tuple, list)) else (int(value),)
        key = labels if isinstance(value, (tuple, list)) else int(value)
        counts = [int(np.isin(flat[start:start + block], labels).sum()) for start in range(0, len(flat), block)]
        total = sum(counts)
        if not total:
            result[key] = []
            continue
        size = min(total, max(min(min_num_samples, total), int(np.ceil(total * min_percent_coverage))))
        offsets = np.sort(rng.choice(total, size, replace=False, shuffle=False))
        indices, seen = np.empty(size, dtype=np.int64), 0
        for index, count in enumerate(counts):
            left, right = np.searchsorted(offsets, [seen, seen + count])
            if right > left:
                start = index * block
                positions = np.flatnonzero(np.isin(flat[start:start + block], labels))
                indices[left:right] = start + positions[offsets[left:right] - seen]
            seen += count
        coordinates = np.column_stack(np.unravel_index(indices, seg.shape[1:]))
        result[key] = np.column_stack((np.zeros(size, dtype=np.int32), coordinates)).astype(np.int32)
    return result


class BoundedPreprocessor(DefaultPreprocessor):
    _sample_foreground_locations = staticmethod(bounded_locations)


class CaseTransform:
    def __init__(self, prepared, reserve_ram_bytes, with_label=True, log_dir=None, logging=None):
        self.prepared, self.reserve, self.with_label = prepared, reserve_ram_bytes, with_label
        self.checked = {}
        self.log_dir, self.logging = log_dir, logging or {}

    def stage(self, name, **fields):
        # 外层的病例/供数阶段负责心跳；细节记录不单独创建计时线程。
        return Stage(self.log_dir, name, heartbeat_seconds=0, console=False,
                     enabled=self.log_dir is not None and self.logging.get("case_details", True), **fields)

    def _source(self, row, kind):
        from pathlib import Path
        path = Path(row[kind])
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        expected = row.get("sha256", {}).get(kind)
        if not expected or kind not in row.get("source_stamp", {}):
            raise ValueError("来源缺少已审计摘要或持久文件标记")
        if list(stamp) != row["source_stamp"][kind]:
            raise ValueError("原始文件与准备阶段的身份不同")
        key = (str(path), kind)
        if key in self.checked and self.checked[key] != stamp:
            raise ValueError("运行期间原始文件发生变化")
        if key not in self.checked:
            with self.stage("source_verify", case_id=row["case_id"], source=kind, bytes=stat.st_size):
                if file_digest(path) != expected:
                    raise ValueError("原始文件摘要与清单不一致")
            after = path.stat()
            if stamp != (after.st_size, after.st_mtime_ns):
                raise ValueError("核验期间原始文件发生变化")
            self.checked[key] = stamp
        return path

    def check_sources(self, row):
        self._source(row, "image")
        if self.with_label:
            self._source(row, "label")

    def read_case(self, row):
        started = perf_counter()
        image = nib.load(self._source(row, "image"))
        if len(image.shape) != 3:
            raise ValueError("原图必须是三维")
        label = nib.load(self._source(row, "label")) if self.with_label else None
        if label is not None:
            if image.shape != label.shape:
                raise ValueError("图像与标签尺寸不一致")
            if row["geometry_policy"] == "matching_affine_author_spacing":
                if not np.allclose(image.affine, label.affine, atol=1e-5, rtol=0):
                    raise ValueError("ImageCHD 图像与标签空间变换不一致")
            else:
                if not np.array_equal(image.affine[:3, :3], label.affine[:3, :3]):
                    raise ValueError("1282 的图像与标签轴矩阵不一致")
                delta = np.linalg.solve(image.affine[:3, :3], label.affine[:3, 3] - image.affine[:3, 3])
                if not (np.allclose(delta, 0) or np.allclose(delta, 1)):
                    raise ValueError("1282 空间头不符合已确认的占位原点模式")
        roi = self.prepared["technical"]["roi_xyz"]
        roi = roi if roi is not None else [[0, int(size)] for size in image.shape]
        if len(roi) != 3 or any(not (0 <= begin < end <= size) for (begin, end), size in zip(roi, image.shape)):
            raise ValueError("技术子区域超出真实图像范围")
        region = tuple(slice(begin, end) for begin, end in roi)
        shape = np.array([end - begin for begin, end in roi])
        native_spacing = np.asarray(row["spacing_xyz"], dtype=np.float32).astype(float)
        estimate = int(np.prod(shape) * 32)
        if host_capacity()["available_bytes"] < self.reserve + estimate:
            raise MemoryError("病例构建所需内存超出安全余量")
        with self.stage("read_decompress", case_id=row["case_id"], shape_xyz=shape.tolist(),
                        spacing_xyz=native_spacing.tolist(), with_label=self.with_label):
            data = np.asarray(image.dataobj[region], dtype=np.float32)
            segmentation = np.asanyarray(label.dataobj[region]) if label is not None else None
        if not np.isfinite(data).all() or data.min() == data.max():
            raise ValueError("图像包含非有限数值或无灰度变化")
        if segmentation is not None:
            if not np.isfinite(segmentation).all() or not np.equal(segmentation, np.rint(segmentation)).all():
                raise ValueError("标签含非整数或非有限数值")
            segmentation = segmentation.astype(np.int16)
            extras = row.get("ignore_label_ids", [])
            if extras:
                segmentation[np.isin(segmentation, extras)] = -1
            classes = len(self.prepared["dataset_json"]["labels"])
            if not ((segmentation >= -1) & (segmentation < classes)).all():
                raise ValueError("标签超出已声明的类别范围")
            if any((segmentation == c).any() for c in row["unknown_classes"]):
                raise ValueError("读取标签与未知类别声明矛盾")
        read_seconds = perf_counter() - started
        return data, segmentation, image, native_spacing, roi, read_seconds

    def __call__(self, row):
        data, segmentation, image, native_spacing, roi, read_seconds = self.read_case(row)
        plan = self.prepared["plan"]
        target_spacing = np.asarray(plan["configurations"]["3d_fullres"]["spacing"])
        transposed_spacing = native_spacing[::-1][plan["transpose_forward"]]
        planned_shape = np.ceil(np.asarray(data.shape)[::-1][plan["transpose_forward"]] *
                                transposed_spacing / target_spacing)
        if host_capacity()["available_bytes"] < self.reserve + int(np.prod(planned_shape) * 24):
            raise MemoryError("重采样所需内存超出安全余量")
        properties = {"spacing": native_spacing[::-1].tolist()}
        manager = PlansManager(deepcopy(plan))
        configuration = manager.get_configuration("3d_fullres")
        transform_started = perf_counter()
        with self.stage("crop_normalize_resample", case_id=row["case_id"], native_shape_xyz=list(data.shape),
                        target_spacing=target_spacing.tolist(), estimated_shape=planned_shape.astype(int).tolist()):
            values, targets, properties = BoundedPreprocessor(verbose=False).run_case_npy(
                data.transpose(2, 1, 0)[None], None if segmentation is None else segmentation.transpose(2, 1, 0)[None],
                properties, manager, configuration, self.prepared["dataset_json"])
        if not np.isfinite(values).all() or values.shape != targets.shape:
            raise ValueError("确定性预处理输出无效")
        if self.log_dir is not None and self.logging.get("case_details", True):
            event(self.log_dir, "case_geometry", console=False, case_id=row["case_id"],
                  raw_shape_xyz=list(image.shape), input_shape_xyz=list(data.shape),
                  cropped_shape=properties["shape_after_cropping_and_before_resampling"],
                  resampled_shape=list(values.shape[1:]), rss_bytes=psutil.Process().memory_info().rss,
                  elapsed_seconds=perf_counter() - transform_started)
        if segmentation is not None:
            present = set(map(int, np.unique(segmentation))) - {-1, 0}
            remaining = set(map(int, np.unique(targets)))
            if not present <= remaining:
                raise ValueError("裁剪或重采样使技术范围内已标类别消失")
        working_affine = image.affine.copy()
        if row["geometry_policy"] == "shared_index_excel":
            working_affine = np.diag([*native_spacing, 1.])
        else:
            working_affine[:3, :3] *= native_spacing / np.linalg.norm(working_affine[:3, :3], axis=0)
        return {"case_id": row["case_id"], "image": values, "label": targets[0] if segmentation is not None else None,
                "unknown_classes": row["unknown_classes"], "properties": properties,
                "source_shape_xyz": list(image.shape), "source_affine": image.affine.tolist(),
                "source_units": image.header.get_xyzt_units()[0], "working_affine": working_affine.tolist(),
                "spacing_xyz": native_spacing.tolist(), "roi_xyz": roi,
                "timing": {"read_decompress_seconds": read_seconds,
                           "deterministic_transform_seconds": perf_counter() - transform_started}}
