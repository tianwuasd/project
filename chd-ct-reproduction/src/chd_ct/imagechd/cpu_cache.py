"""Versioned, read-only mmap inputs shared by loader processes and CV folds."""

import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..recovery import atomic_path, file_hash, output_lock, read_json, write_json
from .common import load_case, safe_child
from .geometry import blood_target, bounding_box, resize

FORMAT = "imagechd-training-inputs-v1"


@contextmanager
def cache_lock(root, timeout=600):
    """Only cache publishers wait; scientific output locks remain fail-fast."""
    deadline = time.monotonic() + timeout
    while True:
        lock = output_lock(root)
        try:
            lock.__enter__()
            break
        except ValueError:
            if time.monotonic() >= deadline:
                raise TimeoutError("等待训练输入缓存写锁超时：" + str(root)) from None
            time.sleep(0.2)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def prepare_input(directory, row, stage, config):
    image, target = load_case(directory, row)
    if not stage.startswith("crop"):
        roi = bounding_box((target > 0) & (target < 8), config["roi_margin"])
        image, target = image[roi], target[roi]
    if stage.startswith("blood"):
        target = blood_target(target)
        # Contiguous slices; morphology remains three-dimensional and unchanged.
        image, target = np.moveaxis(image, 2, 0), np.moveaxis(target, 2, 0)
    else:
        size = config["stages"][stage]["size"]
        image = resize(image, (size,) * 3)
        target = resize(target, (size,) * 3, labels=True)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(target, dtype=np.uint8)


def cached_input(root, directory, row, stage, config, max_gib=32, reserve_gib=2):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    source = safe_child(directory, row["cache"])
    blood = stage.startswith("blood")
    identity = {
        "format": FORMAT,
        "source": str(source),
        "source_sha256": file_hash(source),
        "source_shape": list(row["cache_shape"]),
        "roi_margin": config["roi_margin"],
        "variant": "blood" if blood else stage,
        "size": None if blood else config["stages"][stage]["size"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / (key + ".json")
    paths = {kind: root / (key + "." + kind + ".npy") for kind in ("image", "target")}

    def verify():
        record = read_json(metadata)
        if record["identity"] != identity:
            raise ValueError("训练输入缓存来源不匹配")
        for kind, path in paths.items():
            if file_hash(path) != record["sha256"][kind]:
                raise ValueError("训练输入缓存校验失败：" + str(path))
        return {"paths": paths, "shape": record["shape"], "reused": True}

    if metadata.is_file():
        return verify()
    with cache_lock(root):
        if metadata.is_file():
            return verify()
        image, target = prepare_input(directory, row, stage, config)
        required = image.nbytes + target.nbytes + 1024
        used = sum(p.stat().st_size for p in root.glob("*.npy"))
        replaced = sum(p.stat().st_size for p in paths.values() if p.exists())
        if used - replaced + required > max_gib * 1024**3:
            raise ValueError("训练输入缓存超过 cache_max_gib；请增加预算或使用独立缓存目录")
        if shutil.disk_usage(root).free < required + reserve_gib * 1024**3:
            raise ValueError("训练输入缓存磁盘余量不足，保留空间要求未满足")
        for kind, array in (("image", image), ("target", target)):
            with atomic_path(paths[kind]) as temporary:
                np.save(temporary, array, allow_pickle=False)
        write_json(
            metadata,
            {
                "identity": identity,
                "shape": list(image.shape),
                "sha256": {kind: file_hash(path) for kind, path in paths.items()},
            },
        )
        return {"paths": paths, "shape": list(image.shape), "reused": False}
