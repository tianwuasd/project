import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from ..artifacts.store import digest, file_digest, operation_lock, read_json, write_json
from ..observability.events import event, logged_directory


def tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tensors_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tensors_equal(a, b) for a, b in zip(left, right))
    return left == right


def require_resume_device(saved,device):
    if torch.device(saved).type!=torch.device(device).type:
        raise ValueError('完整训练恢复需要原CPU或CUDA类型；权重可用于跨设备预测')


def verify_optimizer_device(optimizer):
    count = 0
    for parameter,values in optimizer.state.items():
        for name,value in values.items():
            if isinstance(value,torch.Tensor) and name!='step':
                if value.device!=parameter.device or value.dtype!=parameter.dtype:
                    raise ValueError('优化器状态的设备或精度与参数不同')
                count += 1
    return count


@logged_directory("checkpoint_load")
def load_checkpoint(directory, identity, pointer=None):
    directory = Path(directory)
    pointer = read_json(directory / "latest.json") if pointer is None else pointer
    filename = pointer["file"]
    if Path(filename).name != filename or not filename.startswith("state-") or not filename.endswith(".pt"):
        raise ValueError("检查点索引包含非法路径")
    path = directory / filename
    if path.is_symlink():
        raise ValueError('检查点不能是符号链接')
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"identity", "model", "optimizer", "torch_rng", "sampler_rng", "attempts", "updates", "spec", "config",
                "skipped", "pretrained_receipt"}
    if not required <= state.keys() or state["identity"] != identity:
        raise ValueError("检查点不完整或训练身份不一致")
    if state["attempts"] != pointer["attempts"]:
        raise ValueError("检查点与进度索引不一致")
    event(directory, "checkpoint_loaded", file=filename, bytes=path.stat().st_size,
          attempts=state["attempts"], updates=state["updates"])
    return state


@contextmanager
def prediction_snapshot(directory, selection='latest'):
    with operation_lock(directory):
        if selection not in ('latest','best'):
            raise ValueError('检查点选择必须为best或latest')
        yield read_json(Path(directory) / (selection+'.json'))


def remove_owned_state(directory, pointer, identity, protected):
    filename = pointer['file']
    directory = Path(directory).resolve()
    path = directory/filename
    if filename in protected:
        return
    if (not re.fullmatch(r'state-[0-9]{9}-[0-9a-f]{8}\.pt', filename) or path.parent != directory or
            path.is_symlink() or not path.is_file() or pointer.get('writer_identity') != identity or
            'written_stamp' not in pointer):
        event(directory, 'checkpoint_retained', file=filename, reason='缺少本运行写出登记或路径归属不明确')
        return
    stat = path.stat()
    if [stat.st_size, stat.st_mtime_ns] != pointer['written_stamp']:
        event(directory, 'checkpoint_retained', file=filename, reason='文件信息与本运行写出登记不同')
        return
    path.unlink()


@logged_directory("checkpoint_save")
def save_checkpoint(directory, state):
    directory = Path(directory)
    previous = read_json(directory / "latest.json") if (directory / "latest.json").exists() else None
    progress = state.get("progress", {"attempts": state["attempts"]})
    if previous and previous.get("progress", {"attempts": previous["attempts"]}) == progress:
        event(directory, "checkpoint_reused", file=previous["file"], attempts=state["attempts"])
        return previous
    filename = f"state-{state['attempts']:09d}-{digest(progress)[:8]}.pt"
    path = directory / filename
    pending = path.with_name(path.name + ".pending")
    if path.is_symlink() or pending.is_symlink():
        raise ValueError('检查点写入路径不能是符号链接')
    event(directory, "checkpoint_writing", file=filename, attempts=state["attempts"], updates=state["updates"])
    with pending.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    stat = path.stat()
    pointer = {"file": filename, "writer_identity": state['identity'],
               "written_stamp": [stat.st_size, stat.st_mtime_ns], "attempts": state["attempts"],
               "updates": state["updates"], "progress": progress}
    write_json(directory / "latest.json", pointer)
    best_path = directory/'best.json'
    best = read_json(best_path) if best_path.exists() else None
    old_best = best
    selection = progress.get('selection')
    same_scope = best is not None and best.get('progress',{}).get('selection_scope')==progress.get('selection_scope')
    if selection and (best is None or not same_scope or selection['best'] < best['metric']):
        best = dict(pointer, metric=selection['best'], monitor=selection['monitor'], position=selection['best_position'])
        write_json(best_path, best)
    protected = {filename}
    if best:
        protected.add(best['file'])
    if previous and previous["file"] != filename:
        remove_owned_state(directory, previous, state['identity'], protected)
    if old_best and (not previous or old_best['file'] != previous['file']):
        remove_owned_state(directory, old_best, state['identity'], protected)
    event(directory, "checkpoint_committed", **pointer, bytes=path.stat().st_size)
    return pointer
