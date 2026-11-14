import os
from contextlib import contextmanager
from pathlib import Path

import torch

from ..artifacts.store import digest, file_digest, operation_lock, read_json, write_json


def tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tensors_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tensors_equal(a, b) for a, b in zip(left, right))
    return left == right


def load_checkpoint(directory, identity, pointer=None):
    directory = Path(directory)
    pointer = read_json(directory / "latest.json") if pointer is None else pointer
    filename = pointer["file"]
    if Path(filename).name != filename or not filename.startswith("state-") or not filename.endswith(".pt"):
        raise ValueError("检查点索引包含非法路径")
    path = directory / filename
    if path.is_symlink() or file_digest(path) != pointer["sha256"]:
        raise ValueError("最新检查点内容摘要不符")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"identity", "model", "optimizer", "torch_rng", "sampler_rng", "attempts", "updates", "spec", "config",
                "skipped", "pretrained_receipt"}
    if not required <= state.keys() or state["identity"] != identity:
        raise ValueError("检查点不完整或训练身份不一致")
    if state["attempts"] != pointer["attempts"]:
        raise ValueError("检查点与进度索引不一致")
    return state


@contextmanager
def prediction_snapshot(directory):
    with operation_lock(directory):
        yield read_json(Path(directory) / "latest.json")


def save_checkpoint(directory, state):
    directory = Path(directory)
    previous = read_json(directory / "latest.json") if (directory / "latest.json").exists() else None
    progress = state.get("progress", {"attempts": state["attempts"]})
    if previous and previous.get("progress", {"attempts": previous["attempts"]}) == progress:
        return previous
    filename = f"state-{state['attempts']:09d}-{digest(progress)[:8]}.pt"
    path = directory / filename
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(pending, map_location="cpu", weights_only=True)
    if not tensors_equal(state, loaded):
        raise ValueError("新检查点独立读回与保存状态不一致")
    del loaded
    os.replace(pending, path)
    pointer = {"file": filename, "sha256": file_digest(path), "attempts": state["attempts"],
               "updates": state["updates"], "progress": progress}
    write_json(directory / "latest.json", pointer)
    if previous and previous["file"] != filename:
        old = directory / previous["file"]
        if old.parent != directory or old.is_symlink() or file_digest(old) != previous["sha256"]:
            raise ValueError("旧检查点归属或内容发生变化，停止清理")
        old.unlink()
    return pointer
