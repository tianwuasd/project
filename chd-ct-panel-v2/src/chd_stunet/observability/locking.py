"""OS-backed file locks; stale filenames do not imply a live run."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def try_lock(stream):
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return True
    except (OSError,BlockingIOError):
        return False


def unlock(stream):
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        if not try_lock(f):
            raise RuntimeError('Another process owns this output directory')
        try:
            yield
        finally:
            unlock(f)


@contextmanager
def read_slot(gate):
    """Throttle actual source open/hash/decompression, not CPU transforms."""
    gate = Path(gate)
    held = None
    while held is None:
        slots = json.loads(gate.read_text(encoding='utf-8'))['slots']
        for i in range(slots):
            path = gate.with_name(f'.read-slot-{i}')
            f = path.open('a+b')
            if f.tell() == 0:
                f.write(b'0'); f.flush()
            if try_lock(f):
                held = f
                break
            f.close()
        if held is None:
            time.sleep(.05)
    try:
        yield
    finally:
        unlock(held)
        held.close()
