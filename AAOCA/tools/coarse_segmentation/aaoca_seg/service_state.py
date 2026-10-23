"""Standard-library process ownership and private, durable server state."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

CODE_ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)
    return path


def write_private(path, text):
    path = Path(path)
    private_dir(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    write_private(path, json.dumps(value, ensure_ascii=False, indent=2))


@contextmanager
def file_lock(path):
    """OS lock, never unlink: a stale filename is not a held lock."""
    path = Path(path)
    private_dir(path.parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    stream = os.fdopen(fd, "r+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except OSError as error:
        if not locked:
            raise ValueError("已有操作正在进行，请稍后重试") from error
        raise
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def process_identity(pid=None):
    pid = pid or os.getpid()
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            values = [wintypes.FILETIME() for _ in range(4)]
            try:
                if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
                    return None
                if values[1].dwHighDateTime or values[1].dwLowDateTime:
                    return None
                started = (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
                return {"pid": int(pid), "started": str(started), "platform": "windows"}
            finally:
                kernel.CloseHandle(handle)
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": int(pid), "started": fields[19],
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def alive(identity):
    return bool(identity and process_identity(identity.get("pid", -1)) == identity)


def detached(command, log_path, env=None):
    private_dir(Path(log_path).parent)
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    with os.fdopen(fd, "ab") as log:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, cwd=str(CODE_ROOT), env=env,
                                close_fds=True, **kwargs)


def redact(text):
    return re.sub(r"aca_[A-Za-z0-9_-]+", "[许可已隐藏]", str(text))


def tail(path, size=24000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return redact(stream.read().decode("utf-8", errors="replace"))
    except OSError:
        return ""


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
