"""Crash-safe writes and exclusive writers for resumable scientific outputs."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def atomic_path(path):
    path = Path(path)
    # Retain suffixes for writers such as nibabel and numpy.
    temporary = path.with_name("." + uuid4().hex + "." + path.name)
    try:
        yield temporary
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )


@contextmanager
def output_lock(directory):
    """Kernel-owned lock: SIGKILL releases it; a stale file is harmless."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("输出目录有任务正在写入：" + str(directory)) from error
            try:
                yield stream.fileno()
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("输出目录有任务正在写入：" + str(directory)) from error
            yield stream.fileno()


def archive_child(path):
    """Keep an incomplete short task as a sibling; never delete experimental output."""
    path = Path(path).resolve()
    if path.exists():
        target = path.with_name(path.name + ".interrupted-" + uuid4().hex[:8])
        path.rename(target)
        return target
