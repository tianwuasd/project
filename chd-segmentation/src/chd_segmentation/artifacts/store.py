from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

from filelock import FileLock


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    value = sha256()
    for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
        value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


@contextmanager
def operation_lock(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".run.lock"), timeout=0):
        yield


def require_output(path, work_root, protected=()):
    result, root = Path(path).resolve(), Path(work_root).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("输出必须位于工作目录的专用子目录中")
    for source in protected:
        source = Path(source).resolve()
        if result == source or result.is_relative_to(source) or source.is_relative_to(result):
            raise ValueError("输出与受保护的输入或代码位置重叠")
    return result
