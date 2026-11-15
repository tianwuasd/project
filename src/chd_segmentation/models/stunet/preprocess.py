from copy import deepcopy
from time import perf_counter

import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import psutil

from ...observability.events import event
from ...preprocessing.validation import SourceReader
from ...training.host import host_capacity


CONTRACT = "stunet-ct-nnunet-bounded-v1"


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


class StunetTransform(SourceReader):
    def __call__(self, row):
        data, segmentation, image, native_spacing, roi, read_seconds = self.read_case(row)
        plan = self.prepared["plan"]
        target_spacing = np.asarray(plan["configurations"]["3d_fullres"]["spacing"])
        transposed_spacing = native_spacing[::-1][plan["transpose_forward"]]
        planned_shape = np.ceil(np.asarray(data.shape)[::-1][plan["transpose_forward"]] * transposed_spacing / target_spacing)
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
            event(self.log_dir, "case_geometry", console=False, case_id=row["case_id"], raw_shape_xyz=list(image.shape),
                  input_shape_xyz=list(data.shape), cropped_shape=properties["shape_after_cropping_and_before_resampling"],
                  resampled_shape=list(values.shape[1:]), rss_bytes=psutil.Process().memory_info().rss,
                  elapsed_seconds=perf_counter() - transform_started)
        if segmentation is not None:
            present = set(map(int, np.unique(segmentation))) - {-1, 0}
            if not present <= set(map(int, np.unique(targets))):
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
