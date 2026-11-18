from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
import os
from pathlib import Path
import re
import shutil

import numpy as np
import psutil

from ..artifacts.store import digest, operation_lock, read_json, write_json
from ..models.stunet.preprocess import CONTRACT


FORMAT = "chd-offline-npy-v1"


def case_directory(root, key):
    return Path(root) / key[:16]


def recipe(prepared, config):
    plan = prepared["plan"]
    fit_rows = plan.get("generation", {}).get("request", {}).get("training_cases")
    if fit_rows is None:
        fit_ids = set(plan.get("fit_ids", []))
        fit_rows = [r for r in prepared["manifest"]["cases"] if r["case_id"] in fit_ids or
                    not fit_ids and "train" in r["roles"] and r.get("cohort") != "auxiliary"]
    sources = [{k: row.get(k) for k in ("case_id", "sha256", "spacing_xyz", "geometry_policy", "unknown_classes", "ignore_label_ids")}
               for row in sorted(fit_rows, key=lambda r: r["case_id"])]
    return {"format": FORMAT, "transform": CONTRACT, "dataset": prepared["dataset"],
            "statistics_source": config["preprocessing"]["statistics_source"], "fold": prepared["fold"],
            "statistics_inputs": sources,
            "scope": prepared["scope"], "technical": prepared["technical"],
            "plan": {k: v for k, v in prepared["plan"].items() if k != "generation"},
            "labels": prepared["dataset_json"],
            "versions": {n: version(n) for n in ("nnunetv2", "numpy", "scipy", "monai", "nibabel")}}


def case_key(row, recipe_identity):
    return digest({"recipe": recipe_identity, "case_id": row["case_id"], "sha256": row.get("sha256", {}),
                   "spacing_xyz": row["spacing_xyz"], "geometry_policy": row["geometry_policy"],
                   "unknown_classes": row["unknown_classes"], "ignore_label_ids": row.get("ignore_label_ids", [])})


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def record_identity(record):
    entry = {k: v for k, v in record["entry"].items() if k != "timing"}
    files = {name: {k: v for k, v in info.items() if k != "stamp"} for name, info in record["files"].items()}
    return digest({"format": record["format"], "case_key": record["case_key"], "recipe_identity": record["recipe_identity"],
                   "entry": entry, "files": files, "location_keys": record["location_keys"]})


def owned_remove(path):
    path = Path(path)
    allowed = {"image.npy", "label.npy", "locations.npz", "case.json", "case.json.pending", "owner.json"}
    work_allowed = {"source", "source-seg", "mask", "outside", "normalized", "labels", "result-image",
                    "result-label", "coeff", "coeff-z", "separate", "indicator", "indicator-coeff"}
    def known(p):
        if p.is_symlink():
            return False
        if p.is_file():
            return p.name in allowed
        if p.name != "work" or not p.is_dir():
            return False
        return all(c.is_file() and not c.is_symlink() and
            (c.name in ("image.nii", "label.nii") or c.suffix == ".npy" and
             (c.stem in work_allowed or re.fullmatch(r"label-coeff-\d+", c.stem))) for c in p.iterdir())
    if path.is_symlink() or any(not known(p) for p in path.iterdir()):
        raise ValueError("病例目录包含未知文件或链接，拒绝清理")
    shutil.rmtree(path)


def inspect_case(directory, key, recipe_identity, full=False, *, evict=False, boundary=None):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or not (directory / "case.json").is_file():
        return None
    record = read_json(directory / "case.json")
    for name, expected in record["files"].items():
        if name not in ("image.npy", "label.npy", "locations.npz"):
            raise ValueError("离线数据文件名称无效")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return None
    return record


def begin_case(directory, key, recipe_identity):
    directory = Path(directory)
    pending = directory.with_name(directory.name + ".pending")
    if pending.exists():
        owner = read_json(pending / "owner.json")
        if owner != {"case_key": key, "recipe_identity": recipe_identity}:
            raise ValueError("未完成病例目录归属不同")
        owned_remove(pending)
    pending.mkdir(parents=True, exist_ok=False)
    write_json(pending / "owner.json", {"case_key": key, "recipe_identity": recipe_identity})
    return pending


def save_case(directory, key, recipe_identity, entry, workspace=None):
    directory = Path(directory)
    pending = begin_case(directory, key, recipe_identity) if workspace is None else directory.with_name(directory.name + ".pending")
    if read_json(pending / "owner.json") != {"case_key": key, "recipe_identity": recipe_identity}:
        raise ValueError("病例提交目录归属不符")
    metadata = {k: deepcopy(v) for k, v in entry.items() if k not in ("image", "label", "properties")}
    metadata["properties"] = {k: v for k, v in entry["properties"].items() if k != "class_locations"}
    arrays = {"image.npy": entry["image"]}
    if entry["label"] is not None:
        arrays["label.npy"] = entry["label"]
    for name, array in arrays.items():
        if workspace is None:
            with (pending / name).open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            # 已完成的磁盘数组直接转交，避免再次复制整例。
            source = Path(array.filename)
            if not source.resolve().is_relative_to(workspace.path.resolve()):
                raise ValueError("结果数组不属于当前病例工作目录")
            array.flush()
            array._mmap.close()
            workspace.maps = {k: v for k, v in workspace.maps.items() if v is not array}
            os.replace(source, pending / name)
            with (pending / name).open("r+b") as stream:
                os.fsync(stream.fileno())
            workspace.report("write_array", 1, 1)
    locations = entry["properties"].get("class_locations", {})
    keys = list(locations)
    with (pending / "locations.npz").open("wb") as stream:
        np.savez(stream, **{f"p{i}": np.asarray(locations[k], dtype=np.int32) for i, k in enumerate(keys)})
        stream.flush()
        os.fsync(stream.fileno())
    names = [*arrays, "locations.npz"]
    record = {"format": FORMAT, "case_key": key, "recipe_identity": recipe_identity,
              "entry": json_value(metadata), "location_keys": json_value(keys),
              "files": {n: {"stamp": stamp(pending / n)} for n in names}}
    for name, array in arrays.items():
        record["files"][name].update(shape=list(array.shape), dtype=str(array.dtype), array_bytes=array.nbytes)
    record["identity"] = record_identity(record)
    write_json(pending / "case.json", record)
    if workspace is not None:
        workspace.close()
    os.replace(pending, directory)
    return record


def validate_storage(prepared, full=False, config=None, expected_transform=CONTRACT):
    storage = prepared.get("storage")
    if storage is None or storage.get("format") != FORMAT or not storage.get("complete"):
        raise ValueError("需要先运行 preprocess 完成离线预处理；在线旧任务使用冻结版本继续")
    root = Path(storage["root"]).resolve(strict=True)
    catalog = read_json(root / "catalog.json")
    if expected_transform is not None and catalog["recipe"]["transform"] != expected_transform:
        raise ValueError("离线处理方式与模型输入要求不同")
    if catalog['format']!=FORMAT or catalog['recipe']['dataset']!=prepared['dataset'] or catalog['recipe']['labels']!=prepared['dataset_json']:
        raise ValueError('离线描述的数据集、格式或类别不同')
    missing = {row['case_id'] for row in prepared['manifest']['cases']}-storage['cases'].keys()
    if missing:
        raise ValueError('离线描述缺少当前病例：'+', '.join(sorted(missing)))
    return root


def live_references(root):
    path = Path(root) / "references.json"
    rows = read_json(path) if path.exists() else {}
    active = [r for r in rows.values() if r.get("active") and psutil.pid_exists(r["pid"]) and
              psutil.Process(r["pid"]).create_time() == r["birth"]]
    return rows, active


@contextmanager
def storage_lease(prepared, config, *, expected_transform=CONTRACT, expected_recipe=None):
    root = validate_storage(prepared, config=config if expected_recipe is None else None,
                            expected_transform=expected_transform)
    name = str(Path(config["run_dir"]).resolve())
    me = psutil.Process()
    with operation_lock(root):
        rows, _ = live_references(root)
        if rows.get(name, {}).get("active") and psutil.pid_exists(rows[name]["pid"]) and psutil.Process(rows[name]["pid"]).create_time() == rows[name]["birth"]:
            raise RuntimeError("同一任务仍在引用离线数据")
        rows[name] = {"run_dir": name, "prepared_identity": prepared["identity"], "pid": me.pid,
                      "birth": me.create_time(), "active": True, "dataset": prepared["dataset"], "model": config["model"]}
        write_json(root / "references.json", rows)
    try:
        yield
    finally:
        with operation_lock(root):
            rows, _ = live_references(root)
            rows[name]["active"] = False
            write_json(root / "references.json", rows)


class OfflineTransform:
    def __init__(self, prepared, with_label=True):
        self.storage, self.with_label = prepared["storage"], with_label

    def check_sources(self, row):
        if row["case_id"] not in self.storage["cases"]:
            raise KeyError("预处理数据缺少病例：" + row["case_id"])

    def __call__(self, row):
        self.check_sources(row)
        pointer = self.storage["cases"][row["case_id"]]
        directory = case_directory(self.storage["root"], pointer["key"])
        record = read_json(directory / "case.json")
        entry = deepcopy(record["entry"])
        entry["image"] = np.load(directory / "image.npy", mmap_mode="r", allow_pickle=False)
        entry["label"] = np.load(directory / "label.npy", mmap_mode="r", allow_pickle=False) if self.with_label else None
        entry["properties"]["class_locations"] = {}
        if self.with_label:
            with np.load(directory / "locations.npz", allow_pickle=False) as arrays:
                for i, key in enumerate(record["location_keys"]):
                    values = arrays[f"p{i}"]
                    entry["properties"]["class_locations"][tuple(key) if isinstance(key, list) else key] = [] if values.size == 0 else values
        entry["preprocessing_timing"] = entry["timing"]
        entry["timing"] = {"read_decompress_seconds": 0., "deterministic_transform_seconds": 0.}
        return entry
