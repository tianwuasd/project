"""Process-wide payload read slots, independent of compute worker count."""

import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

_local = threading.local()


def create(env, limit):
    root = Path(env.get("TMPDIR", tempfile.gettempdir())) / ("ct-v2-reads-" + uuid4().hex)
    root.mkdir(parents=True)
    for number in range(max(1, int(limit))):
        (root / f"{number}.lock").write_bytes(b"0")
    return str(root)


@contextmanager
def slot():
    root = os.environ.get("CHD_V2_READ_GATE")
    if not root or getattr(_local, "depth", 0):
        yield
        return
    stream = None
    while stream is None:
        for file in sorted(Path(root).glob("*.lock")):
            candidate = file.open("r+b")
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt

                    msvcrt.locking(candidate.fileno(), msvcrt.LK_NBLCK, 1)
                stream = candidate
                break
            except OSError as error:
                candidate.close()
                if error.errno not in (11, 13, 36):
                    raise
        if stream is None:
            if not any(Path(root).glob("*.lock")):
                raise ValueError("读取槽缺失，不能无界读取")
            time.sleep(0.02)
    _local.depth = 1
    try:
        yield
    finally:
        _local.depth = 0
        stream.close()


def payload(function, *args, **kwargs):
    with slot():
        return function(*args, **kwargs)
