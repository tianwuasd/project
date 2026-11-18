from contextlib import contextmanager
import ctypes
import gzip
from itertools import product
import os
import shutil
from pathlib import Path
from time import perf_counter

from batchgenerators.augmentations.utils import _resample_seg_by_argmax
from filelock import FileLock
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_propagation, map_coordinates, spline_filter1d
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape, determine_do_sep_z_and_axis

from ...observability.events import event
from ...observability.progress import console_progress, resource_summary
from ...artifacts.store import read_json, write_json
from ...artifacts.cache import release_private_mapping, discard_file_pages
from ...preprocessing.validation import SourceReader
from .preprocess import bounded_locations
from .block_execution import BlockExecution


IMPLEMENTATION = "stunet-global-coordinate-blocks-v2"


def tile_extent(shape, voxels, full_axis=None):
    extent = list(map(int, shape))
    while int(np.prod(extent)) > voxels:
        choices = [i for i in range(len(shape)) if i != full_axis and extent[i] > 1]
        if not choices:
            raise MemoryError("单条插值轴线超过配置块预算")
        axis = max(choices, key=lambda i: extent[i])
        extent[axis] = (extent[axis] + 1) // 2
    return extent


def tiles(shape, voxels, full_axis=None):
    extent = tile_extent(shape, voxels, full_axis)
    for begin in product(*(range(0, size, step) for size, step in zip(shape, extent))):
        yield tuple(slice(b, min(b + step, size)) for b, step, size in zip(begin, extent, shape))


class BlockWorkspace:
    def __init__(self, path, config, guard, case_id):
        self.path, self.config, self.guard, self.case_id = Path(path), config, guard, case_id
        self.path.mkdir(exist_ok=False)
        self.maps, self.last_log, self.peak_block = {}, 0., 0
        self.streams = []
        self.started = perf_counter()
        self.phase, self.phase_started = None, self.started
        self.plane_context = None
        self.block_bytes = config["preprocessing"]["block_bytes"]
        self.execution = BlockExecution(self)

    def create(self, name, shape, dtype):
        path = self.path / (name + ".npy")
        if name in self.maps:
            self.drop(name)
        value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(map(int, shape)))
        self.maps[name] = value
        return value

    def release_pages(self):
        for value in self.maps.values():
            release_private_mapping(value)
        for stream in self.streams:
            stream.flush()
            os.fsync(stream.fileno())
            discard_file_pages(stream)
        self.release_raw_pages()
        if os.name == "nt" and sum(value.nbytes for value in self.maps.values()) > 64 * 1024 ** 2:
            # 仅回收当前预处理进程的可分页工作集，不触碰其他任务。
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            if not psapi.EmptyWorkingSet(kernel.GetCurrentProcess()):
                raise ctypes.WinError(ctypes.get_last_error())

    def release_raw_pages(self):
        if hasattr(os, "posix_fadvise"):
            for name in ("image.nii", "label.nii"):
                path = self.path / name
                if path.is_file():
                    with path.open("rb") as stream:
                        discard_file_pages(stream)

    def drop(self, name):
        value = self.maps.pop(name)
        value.flush()
        value._mmap.close()
        (self.path / (name + ".npy")).unlink()

    @contextmanager
    def allocation(self, count, label):
        available = 0 if self.execution is None else getattr(self.execution.local, "allowance", 0)
        if available:
            if count > available:
                raise MemoryError("分块内部临时量超出已核算额度")
            yield
            return
        with self.guard.admit(int(count), label=f"{self.case_id}:{label}", release=self.release_pages) as ticket:
            if ticket is None:
                raise MemoryError("持续内存压力已停止当前病例；完整病例及冻结计划保留，可重新运行预处理")
            self.peak_block = max(self.peak_block, int(count))
            yield

    def blocks(self, shape, per_voxel=64, full_axis=None, label="block", block_bytes=None):
        limit = max(1, (self.block_bytes if block_bytes is None else block_bytes) // per_voxel)
        extent = tile_extent(shape, limit, full_axis)
        total = int(np.prod([(size + step - 1) // step for size, step in zip(shape, extent)]))
        for index, section in enumerate(tiles(shape, limit, full_axis)):
            self.guard.observed_boundary()
            with self.allocation(int(np.prod([s.stop - s.start for s in section])) * per_voxel, label):
                yield section
            if perf_counter() - self.last_log >= self.config["logging"]["heartbeat_seconds"] or index + 1 == total:
                self.report(label, index + 1, total)

    def checkpoint(self):
        if self.guard.boundary()["stop_requested"]:
            raise MemoryError("持续内存压力已停止当前病例；完整结果保留")

    def report(self, phase, done, total=None):
        plane_record = {"plane_index": None, "plane_total": None}
        if self.plane_context is not None and total is not None:
            plane, planes, kind = self.plane_context
            plane_record = {"plane_index": plane + 1, "plane_total": planes}
            phase = kind + ":" + phase
        now = perf_counter()
        changed = phase != self.phase
        if changed:
            self.phase, self.phase_started = phase, now
        finished = done == total and (self.plane_context is None or plane_record["plane_index"] == plane_record["plane_total"])
        if now - self.last_log < self.config["logging"]["heartbeat_seconds"] and not finished and not changed:
            return
        path = Path(self.config["run_dir"]) / "progress-preprocess.json"
        memory = self.guard.snapshot()
        owned = [*self.path.iterdir(), *self.path.parent.iterdir()]
        if len(owned) > 512:
            raise ValueError("单病例工作目录超过已声明的有界文件数量")
        disk = {"task_storage_bytes": sum(p.stat().st_size for p in owned if p.is_file()),
                "data_free_bytes": shutil.disk_usage(self.path).free}
        with FileLock(str(path.parent / '.progress-preprocess.lock')):
            overall = read_json(path) if path.exists() else {"total": None, "done": 0, "new": 0, "reused": 0, "failures": 0}
            record = {**overall, "stage": "preprocess", "current_case": self.case_id, "phase": phase,
                      "block_done": done, "block_total": total, "block_remaining": None if total is None else total - done,
                      "resource": resource_summary(memory), "disk": disk, "elapsed_seconds": now - self.started,
                      "case_seconds": now-self.started, "stage_seconds": now-self.phase_started,
                      "peak_block_bytes": self.peak_block, **plane_record}
            if self.execution is not None and self.execution.phase == phase:
                record["execution"] = self.execution.metrics()
            else:
                record['execution'] = {'applied_case_threads': 1, 'applied_block_bytes': self.block_bytes}
            write_json(path, record)
        event(self.config["run_dir"], "preprocess_blocks", console=False, **record)
        console_progress(record)
        self.last_log = now

    def close(self):
        self._close_execution()
        for name in list(self.maps):
            self.drop(name)
        for path in self.path.iterdir():
            if not path.is_file() or path.is_symlink() or path.name not in ("image.nii", "label.nii"):
                raise ValueError("分块工作目录存在未知文件")
            path.unlink()
        self.path.rmdir()

    def release_handles(self):
        self._close_execution()
        for value in self.maps.values():
            value._mmap.close()
        self.maps.clear()

    def _close_execution(self):
        if self.execution is not None:
            self.execution.close()
            self.execution = None


def local_image(image, source, name, workspace):
    if source.suffix.lower() != ".gz" or workspace.config["technical"]["roi_xyz"] is not None:
        return image
    target = workspace.path / (name + ".nii")
    with gzip.open(source, "rb") as incoming, target.open("xb") as outgoing:
        workspace.streams.append(outgoing)
        try:
            count = 0
            while True:
                with workspace.allocation(8 * 1024 ** 2, "decompress"):
                    block = incoming.read(4 * 1024 ** 2)
                    if not block:
                        break
                    outgoing.write(block)
                    count += 1
                    workspace.report("decompress", count)
                    del block
            outgoing.flush()
            os.fsync(outgoing.fileno())
        finally:
            workspace.streams.remove(outgoing)
    return nib.load(target, mmap="r", keep_file_open=False)


def unique_values(array, workspace):
    values = set()
    for section in workspace.blocks(array.shape, 24, label="label_values"):
        values.update(map(int, np.unique(array[section])))
    return np.array(sorted(values), dtype=array.dtype)


def read_sources(reader, row, workspace):
    with workspace.allocation(64 * 1024, "source_metadata"):
        image, label, spacing, roi = reader.read_metadata(row)
    native_image = image
    image = local_image(image, Path(row["image"]), "image", workspace)
    if label is not None:
        label = local_image(label, Path(row["label"]), "label", workspace)
    xyz = tuple(end - begin for begin, end in roi)
    order = reader.prepared["plan"]["transpose_forward"]
    shape = tuple(np.asarray(xyz[::-1])[order])
    data = workspace.create("source", shape, np.float32)
    seg = workspace.create("source-seg", shape, np.int16)
    mask = workspace.create("mask", shape, np.bool_)
    lo, hi, present = np.inf, -np.inf, set()
    for part in workspace.blocks(xyz, 40, label="read_decompress", block_bytes=workspace.config["preprocessing"]["read_block_bytes"]):
        source = tuple(slice(s.start + start, s.stop + start) for s, (start, _) in zip(part, roi))
        target = tuple(part[::-1][i] for i in order)
        values = np.asarray(image.dataobj[source], dtype=np.float32).transpose(2, 1, 0).transpose(order)
        if not np.isfinite(values).all():
            raise ValueError("原图包含非有限数值")
        lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
        data[target] = values
        mask[target] = values != 0
        if label is None:
            seg[target] = 0
        else:
            labels = np.asarray(label.dataobj[source]).transpose(2, 1, 0).transpose(order)
            if not np.isfinite(labels).all() or not np.equal(labels, np.rint(labels)).all():
                raise ValueError("标签含非整数或非有限数值")
            allowed = set(range(len(reader.prepared["dataset_json"]["labels"]))) | {-1} | set(row.get("ignore_label_ids", []))
            if not set(map(int, np.unique(labels))) <= allowed:
                raise ValueError("原始标签在窄整数转换前超出已声明范围")
            labels = labels.astype(np.int64)
            labels[np.isin(labels, row.get("ignore_label_ids", []))] = -1
            labels = labels.astype(np.int16)
            classes = len(reader.prepared["dataset_json"]["labels"])
            if not ((labels >= -1) & (labels < classes)).all():
                raise ValueError("标签超出已声明范围")
            present.update(map(int, np.unique(labels)))
            if set(row["unknown_classes"]) & present:
                raise ValueError("未标注类别与标签实际值冲突")
            seg[target] = labels
        del values
    if lo == hi:
        raise ValueError("原图无灰度变化")
    workspace.release_raw_pages()
    return native_image, spacing, roi, data, seg, mask, present - {-1, 0}, label is not None


def fill_and_crop(mask, workspace):
    outside = workspace.create("outside", mask.shape, np.bool_)
    depth = max(1, workspace.block_bytes // (int(np.prod(mask.shape[1:])) * 16))
    spans = [(z, min(z + depth, mask.shape[0])) for z in range(0, mask.shape[0], depth)]
    changed, passes = True, 0
    while changed:
        changed, passes = False, passes + 1
        for index, (begin, end) in enumerate(spans + list(reversed(spans))):
            left, right = max(0, begin - 1), min(mask.shape[0], end + 1)
            with workspace.allocation((right - left) * int(np.prod(mask.shape[1:])) * 16, "fill_holes"):
                background = ~mask[left:right]
                seeds = np.array(outside[left:right], copy=True)
                for axis in (1, 2):
                    for edge in (0, -1):
                        face = [slice(None)] * 3
                        face[axis] = edge
                        seeds[tuple(face)] |= background[tuple(face)]
                if left == 0:
                    seeds[0] |= background[0]
                if right == mask.shape[0]:
                    seeds[-1] |= background[-1]
                propagated = binary_propagation(seeds, mask=background)
                core = slice(begin - left, end - left)
                changed |= bool(np.any(propagated[core] != outside[begin:end]))
                outside[begin:end] = propagated[core]
                del seeds, propagated, background
            workspace.report(f"fill_holes_pass_{passes}", index + 1, 2 * len(spans))
        event(workspace.config["run_dir"], "preprocess_fill_pass", case_id=workspace.case_id, passes=passes, changed=changed)
    minimum, maximum = np.asarray(mask.shape), np.zeros(3, dtype=int)
    for part in workspace.blocks(mask.shape, 8, label="crop_bounds"):
        filled = ~outside[part]
        mask[part] = filled
        if filled.any():
            for axis in range(3):
                indices = np.flatnonzero(filled.any(axis=tuple(i for i in range(3) if i != axis)))
                minimum[axis] = min(minimum[axis], part[axis].start + indices[0])
                maximum[axis] = max(maximum[axis], part[axis].start + indices[-1] + 1)
    workspace.drop("outside")
    if np.any(minimum >= maximum):
        raise ValueError("原图非零裁剪范围为空")
    return [[int(a), int(b)] for a, b in zip(minimum, maximum)]


def normalize_crop(data, seg, mask, bbox, plan, workspace):
    cfg = plan["configurations"]["3d_fullres"]
    if cfg["normalization_schemes"] != ["CTNormalization"]:
        raise ValueError("分块 STU-Net 需要已声明的 CTNormalization 计划")
    stats = plan["foreground_intensity_properties_per_channel"]["0"]
    shape = tuple(b - a for a, b in bbox)
    normalized = workspace.create("normalized", shape, np.float32)
    labels = workspace.create("labels", shape, np.int16)
    for part in workspace.blocks(shape, 24, label="normalize"):
        source = tuple(slice(s.start + a, s.stop + a) for s, (a, _) in zip(part, bbox))
        block = np.array(data[source], dtype=np.float32, copy=True)
        np.clip(block, stats["percentile_00_5"], stats["percentile_99_5"], out=block)
        block -= stats["mean"]
        block /= max(stats["std"], 1e-8)
        normalized[part] = block
        value = np.array(seg[source], copy=True)
        value[(value == 0) & (~mask[source])] = -1
        labels[part] = value
        del block, value
    workspace.drop("source")
    workspace.drop("source-seg")
    workspace.drop("mask")
    return normalized, labels


def prepare_coefficients(source, order, workspace, name):
    padding = 12 if order > 1 else 0
    shape = tuple(n + 2 * padding for n in source.shape)
    coeff = workspace.create(name, shape, np.float64)
    def copy(part):
        indices = [np.clip(np.arange(s.start, s.stop) - padding, 0, size - 1) for s, size in zip(part, source.shape)]
        coeff[part] = source[np.ix_(*indices)]
    parallel_sections(workspace, shape, 24, copy, "prefilter_copy")
    if order > 1:
        for axis in range(source.ndim):
            def filter_lines(part):
                spline_filter1d(coeff[part], order=order, axis=axis, output=coeff[part], mode="nearest")
            parallel_sections(workspace, shape, 24, filter_lines, f"prefilter_axis_{axis}", full_axis=axis)
    return coeff, padding


def parallel_sections(workspace, shape, cost, operation, label, full_axis=None):
    workspace.execution.sections(shape, cost, operation, label, full_axis)


def linear_points(span, source_shape, output_shape, padding=0):
    indices = np.unravel_index(np.arange(*span, dtype=np.int64), output_shape)
    points = np.asarray([(axis.astype(np.float64) + .5) * (float(old) / new) - .5 + padding
                         for axis, old, new in zip(indices, source_shape, output_shape)])
    return indices, points


def support_window(span, source_shape, output_shape):
    bounds = []
    for axis, size in enumerate(output_shape):
        stride = int(np.prod(output_shape[axis + 1:]))
        first, last = span[0] // stride, (span[1] - 1) // stride
        begin, end = (0, size) if first // size != last // size else (first % size, last % size + 1)
        old = source_shape[axis]
        lo = max(0, min(old - 1, int(np.floor((begin + .5) * (float(old) / size) - .5))))
        hi = min(old, max(lo + 1, int(np.ceil((end - .5) * (float(old) / size) - .5)) + 1))
        bounds.append(slice(lo, hi))
    return tuple(bounds)


def coordinates(part, source_shape, output_shape, padding=0):
    axes = [(np.arange(s.start, s.stop, dtype=np.float64) + .5) * (float(old) / new) - .5 + padding
            for s, old, new in zip(part, source_shape, output_shape)]
    return np.asarray(np.meshgrid(*axes, indexing="ij"))


def resize_image(source, output, order, workspace, name, clip=True):
    own_coeff = order > 1
    coeff, padding = prepare_coefficients(source, order, workspace, name) if own_coeff else (source, 0)
    low, high = np.inf, -np.inf
    if clip:
        for part in workspace.blocks(source.shape, 8, label="intensity_bounds"):
            low, high = min(low, float(source[part].min())), max(high, float(source[part].max()))
    def interpolate(span):
        indices, points = linear_points(span, source.shape, output.shape, padding)
        block = map_coordinates(coeff, points, output=np.float64, order=order, mode="nearest", prefilter=False)
        if clip:
            np.clip(block, low, high, out=block)
        if not np.isfinite(block).all():
            raise ValueError("图像插值产生非有限数值")
        output[indices] = block
        del block, points
    workspace.execution.linear(output.shape, 128, interpolate, "image_resample")
    if own_coeff:
        workspace.drop(name)


def resize_labels(source, output, order, workspace):
    labels = unique_values(source, workspace)
    coefficients = {}
    if order > 1:
        for index, label in enumerate(labels):
            indicator = workspace.create("indicator", source.shape, np.float64)
            for section in workspace.blocks(source.shape, 24, label="label_indicator"):
                indicator[section] = source[section] == label
            name = f"label-coeff-{index}"
            coefficients[int(label)] = (*prepare_coefficients(indicator, order, workspace, name), name)
            workspace.drop("indicator")
    # 最近邻及每类分数遵循当前 batchgenerators 的精确并列决策。
    def interpolate(span):
        indices, points = linear_points(span, source.shape, output.shape)
        if order == 0:
            output[indices] = map_coordinates(source, points, order=0, mode="nearest", prefilter=False)
        else:
            shape = (span[1] - span[0],)
            def scores(value):
                if order > 1:
                    coeff, padding, _ = coefficients[int(value)]
                    result = map_coordinates(coeff, points + padding, order=order, mode="nearest", prefilter=False)
                else:
                    window = support_window(span, source.shape, output.shape)
                    required = int(np.prod([s.stop - s.start for s in window])) * 16
                    with workspace.allocation(required, "label_local_support"):
                        indicator = (source[window] == value).astype(np.float64)
                        local = points - np.asarray([s.start for s in window])[:, None]
                        result = map_coordinates(indicator, local, order=order, mode="nearest", prefilter=False)
                np.clip(result, 0., 1., out=result)
                return result
            output[indices] = _resample_seg_by_argmax(labels, scores,
                lambda: map_coordinates(source, points, order=0, mode="nearest", prefilter=False),
                shape, source.dtype, "nearest")
        del points
    extra = None if order != 1 else lambda span: int(np.prod([
        s.stop - s.start for s in support_window(span, source.shape, output.shape)])) * 16
    workspace.execution.linear(output.shape, 128, interpolate, "label_resample", extra)
    for _, _, name in coefficients.values():
        workspace.drop(name)


def resample(source, output, current, target, kwargs, is_seg, workspace):
    order, order_z = kwargs["order"], kwargs["order_z"]
    separate, axis = determine_do_sep_z_and_axis(kwargs.get("force_separate_z"), current, target,
        kwargs.get("separate_z_anisotropy_threshold", 3))
    if source.shape == output.shape:
        for part in workspace.blocks(source.shape, 16, label="unchanged_grid"):
            output[part] = source[part]
        return
    if not separate:
        if is_seg:
            resize_labels(source, output, order, workspace)
        else:
            resize_image(source, output, order, workspace, "coeff")
        return
    middle_shape = list(output.shape)
    middle_shape[axis] = source.shape[axis]
    middle = workspace.create("separate", middle_shape, np.float64)
    for plane in range(source.shape[axis]):
        workspace.plane_context = (plane, source.shape[axis], "label_inplane" if is_seg else "image_inplane")
        section = [slice(None)] * 3
        section[axis] = plane
        section = tuple(section)
        if is_seg:
            resize_labels(source[section], middle[section], order, workspace)
        else:
            resize_image(source[section], middle[section], order, workspace, "coeff")
    workspace.plane_context = None
    if source.shape[axis] == output.shape[axis]:
        for part in workspace.blocks(output.shape, 16, label="separate_copy"):
            output[part] = middle[part]
    elif not is_seg or order_z == 0:
        resize_image(middle, output, order_z, workspace, "coeff-z", clip=False)
    else:
        labels = unique_values(middle, workspace)
        for part in workspace.blocks(output.shape, 128, label="separate_labels"):
            points = coordinates(part, middle.shape, output.shape)
            block = np.zeros(tuple(s.stop - s.start for s in part), dtype=output.dtype)
            for label in labels:
                indicator = workspace.create("indicator", middle.shape, np.float64)
                for section in workspace.blocks(middle.shape, 24, label="separate_indicator"):
                    indicator[section] = middle[section] == label
                coeff, padding = prepare_coefficients(indicator, order_z, workspace, "indicator-coeff")
                block[np.round(map_coordinates(coeff, points + padding, order=order_z,
                    mode="nearest", prefilter=False)) > .5] = label
                workspace.drop("indicator-coeff")
                workspace.drop("indicator")
            output[part] = block
            del block, points
    workspace.drop("separate")


def stream_case(prepared, row, config, pending, guard):
    workspace = BlockWorkspace(Path(pending) / "work", config, guard, row["case_id"])
    succeeded = False
    try:
        entry = stream_into_workspace(prepared, row, config, guard, workspace)
        succeeded = True
        return entry, workspace
    finally:
        if not succeeded:
            workspace.release_handles()


def stream_into_workspace(prepared, row, config, guard, workspace):
    began = perf_counter()
    reader = SourceReader(prepared, config["resources"]["reserve_ram_bytes"], with_label=row.get("label") is not None,
                          log_dir=config["run_dir"], logging=config["logging"])
    image, spacing, roi, data, seg, mask, present, has_label = read_sources(reader, row, workspace)
    read_seconds = perf_counter() - began
    plan = prepared["plan"]
    bbox = fill_and_crop(mask, workspace)
    properties = {"spacing": spacing[::-1].tolist(), "shape_before_cropping": list(data.shape),
                  "bbox_used_for_cropping": bbox,
                  "shape_after_cropping_and_before_resampling": [b - a for a, b in bbox]}
    normalized, labels = normalize_crop(data, seg, mask, bbox, plan, workspace)
    current = spacing[::-1][plan["transpose_forward"]]
    cfg = plan["configurations"]["3d_fullres"]
    shape = compute_new_shape(normalized.shape, current, cfg["spacing"])
    values = workspace.create("result-image", (1, *shape), np.float32)
    targets = workspace.create("result-label", shape, np.int8 if len(prepared["dataset_json"]["labels"]) <= 128 else np.int16)
    resample(normalized, values[0], current, cfg["spacing"], cfg["resampling_fn_data_kwargs"], False, workspace)
    resample(labels, targets, current, cfg["spacing"], cfg["resampling_fn_seg_kwargs"], True, workspace)
    if has_label:
        if not present <= set(map(int, unique_values(targets, workspace))):
            raise ValueError("裁剪或重采样使技术范围内已标类别消失")
        properties["class_locations"] = bounded_locations(targets[None],
            [int(v) for v in prepared["dataset_json"]["labels"].values() if int(v) != 0], workspace=workspace)
    affine = image.affine.copy()
    if row["geometry_policy"] == "shared_index_excel":
        affine = np.diag([*spacing, 1.])
    else:
        affine[:3, :3] *= spacing / np.linalg.norm(affine[:3, :3], axis=0)
    event(config["run_dir"], "preprocess_execution", console=False, case_id=row["case_id"],
          **workspace.execution.metrics())
    entry = {"case_id": row["case_id"], "image": values, "label": targets if has_label else None,
        "unknown_classes": row["unknown_classes"], "properties": properties, "source_shape_xyz": list(image.shape),
        "source_affine": image.affine.tolist(), "source_units": image.header.get_xyzt_units()[0],
        "working_affine": affine.tolist(), "spacing_xyz": spacing.tolist(), "roi_xyz": roi,
        "timing": {"read_decompress_seconds": read_seconds, "deterministic_transform_seconds": perf_counter() - began - read_seconds}}
    return entry
