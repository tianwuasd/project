from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

from ..artifacts.store import file_digest
from ..observability.events import Stage
from ..training.host import host_capacity


class SourceReader:
    def __init__(self, prepared, reserve_ram_bytes, with_label=True, log_dir=None, logging=None):
        self.prepared, self.reserve, self.with_label = prepared, reserve_ram_bytes, with_label
        self.checked = {}
        self.log_dir, self.logging = log_dir, logging or {}

    def stage(self, name, **fields):
        return Stage(self.log_dir, name, heartbeat_seconds=0, console=getattr(self, "console", False),
                     enabled=self.log_dir is not None and self.logging.get("case_details", True), **fields)

    def _source(self, row, kind):
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
        if self.with_label and row.get("label"):
            self._source(row, "label")

    def read_case(self, row):
        started = perf_counter()
        image = nib.load(self._source(row, "image"))
        if len(image.shape) != 3:
            raise ValueError("原图必须是三维")
        label = nib.load(self._source(row, "label")) if self.with_label and row.get("label") else None
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
        if host_capacity()["available_bytes"] < self.reserve + int(np.prod(shape) * 32):
            raise MemoryError("病例构建所需内存超出安全余量")
        with self.stage("read_decompress", case_id=row["case_id"], shape_xyz=shape.tolist(),
                        spacing_xyz=native_spacing.tolist(), with_label=label is not None):
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
        return data, segmentation, image, native_spacing, roi, perf_counter() - started
