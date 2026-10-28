"""Versioned, read-only mmap inputs shared by loader processes and CV folds."""

import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..recovery import atomic_path, file_hash, output_lock, read_json, write_json
from .common import load_case, safe_child
from .geometry import blood_target, resize
from .methods import LABEL_RESIZE
from .roi_inputs import row_roi

FORMAT = "imagechd-training-inputs-v3"
GIB = 1024**3


def cache_used_bytes(root):
    return sum(p.stat().st_size for p in root.glob("*.npy"))


def disk_budget(root, used, reserve_gib):
    """Recheck writable space under the shared writer lock, not total disk size.

    used + free stays constant as this cache grows, so 80% does not creep
    towards 100%. Other disk users can reduce it. A known allocation may
    further bound the pool; filesystem free is not proof of a personal quota.
    """
    free = shutil.disk_usage(root).free
    capacity = used + free
    allocation = None
    value = os.environ.get("CHD_CT_CACHE_CAPACITY_GIB")
    if value is not None:
        try:
            amount = float(value)
            if not math.isfinite(amount) or amount <= 0 or not math.isfinite(amount * GIB):
                raise ValueError
            allocation = int(amount * GIB)
        except ValueError:
            raise ValueError("CHD_CT_CACHE_CAPACITY_GIB 需要有限正数（缓存可分配总容量，GiB）") from None
        capacity = min(capacity, allocation)
    reserve = max(2 * GIB, int(reserve_gib * GIB))
    return {
        "policy": "disk-available-80-percent-v1",
        "used_bytes": used,
        "free_bytes": free,
        "allocation_bytes": allocation,
        "limit_bytes": min(capacity * 4 // 5, max(0, capacity - reserve)),
        "reserve_bytes": reserve,
        "quota_scope": "explicit-cache-allocation" if allocation is not None else "filesystem-visible-only",
    }


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
        roi = row_roi(row, image.shape)
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
    # max_gib is retained for old callers/frozen configs, not enforced. Capacity
    # policy is operational and must not change tensor identities or contracts.
    root, directory = Path(root).resolve(), Path(directory).resolve()
    source = safe_child(directory, row["cache"])
    blood = stage.startswith("blood")
    identity = {
        "format": FORMAT,
        "label_resize": LABEL_RESIZE,
        "source": str(source),
        "source_sha256": file_hash(source),
        "source_shape": list(row["cache_shape"]),
        "roi_margin": config["roi_margin"],
        "predicted_roi": row.get("predicted_roi") if not stage.startswith("crop") else None,
        "roi_sha256": row.get("roi_sha256") if not stage.startswith("crop") else None,
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
        used = cache_used_bytes(root)
        replaced = sum(p.stat().st_size for p in paths.values() if p.exists())
        budget = disk_budget(root, used, reserve_gib)
        budget.update(legacy_cache_max_gib=max_gib, required_bytes=required, checked_at=time.time())
        write_json(root / "cache-budget.json", budget)
        if budget["free_bytes"] < required + budget["reserve_bytes"]:
            raise ValueError("训练输入缓存磁盘余量不足，至少保留2GiB；已有缓存与检查点保留")
        if used - replaced + required > budget["limit_bytes"]:
            raise ValueError(
                f"训练输入缓存超过当前可分配容量的80%：已用{used / GIB:.2f}GiB，"
                f"上限{budget['limit_bytes'] / GIB:.2f}GiB；详见 {root / 'cache-budget.json'}"
            )
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
        return {"paths": paths, "shape": list(image.shape), "reused": False, "budget": budget}
