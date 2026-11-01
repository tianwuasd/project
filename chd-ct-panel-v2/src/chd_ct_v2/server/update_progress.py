"""Drain Git output concurrently, show real progress, and retain bounded safe logs."""

import codecs
import os
import queue
import re
import signal
import subprocess
import threading
import time

HEARTBEAT_SECONDS = 5
TAIL_CHARS = 65536
MAX_LOG_BYTES = 1024 * 1024
PROGRESS = re.compile(
    r"^(?:remote: )?(?:Enumerating|Counting|Compressing|Receiving|Resolving|Unpacking|Updating|Checking out) (?:objects|deltas|files):\s*\d"
)


def safe_text(value):
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    value = "".join(c for c in value if c.isprintable() or c in "\n\t")
    value = re.sub(r"(?i)(authorization|proxy-authorization)\s*[:=].*", r"\1: [redacted]", value)
    value = re.sub(
        r"(?i)\b(password|token|secret|access_token|api[_-]?key)\s*[=:]\s*\S+", r"\1=[redacted]", value
    )
    value = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[redacted]@", value)
    value = re.sub(r"(?i)(https?://[^\s?#]+)[?#][^\s]*", r"\1?[redacted]", value)
    value = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b", "[redacted]", value)
    return value


def failure_reason(stderr):
    text = stderr.lower()
    if "would be overwritten" in text:
        return "本地文件将被更新覆盖，已拒绝应用；请先保留并处理这些文件"
    if "no space left" in text:
        return "磁盘空间不足，请先检查代码和 runtime 所在磁盘"
    if any(
        word in text for word in ("rpc failed", "early eof", "index-pack failed", "unexpected disconnect")
    ):
        return "对象传输或解包中断；先检查当前下载线路和磁盘，不据此重建密钥"
    if "permission denied" in text or "authentication failed" in text:
        return "仓库认证或访问权限失败，请检查当前配置的密钥/权限"
    if "host key verification failed" in text:
        return "SSH 主机指纹校验失败，请核对已知主机记录"
    if "could not resolve" in text or "name or service not known" in text:
        return "目标域名解析失败"
    if any(
        word in text
        for word in (
            "ssl",
            "tls",
            "banner exchange",
            "connection reset",
            "connection refused",
            "connection timed out",
            "connect to host",
        )
    ):
        return "代理或远端连接/握手失败，请查看本次目标的代理日志"
    if "not found" in text or "does not appear to be a git repository" in text:
        return "仓库不可访问、地址不存在或权限不足"
    return "Git 进程异常退出；具体输出见日志"


def terminate(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


class StageTimeout(subprocess.TimeoutExpired):
    def __init__(self, stage, timeout, log):
        super().__init__(stage, timeout)
        self.log = log

    def __str__(self):
        return f"{self.cmd}总超时（上限 {self.timeout:g} 秒）；日志：{self.log}。请查更新记录确认实际状态。"


class StageFailure(ValueError):
    def __init__(self, stage, stderr, log):
        super().__init__(f"{stage}失败：{failure_reason(stderr)}；日志：{log}")
        self.output_tail = stderr[-4000:]


class Journal:
    def __init__(self, path):
        self.stream = path.open("wb")
        self.size = 0
        self.full = False

    def write(self, message):
        if self.full:
            return
        data = (safe_text(message) + "\n").encode("utf-8")
        if self.size + len(data) > MAX_LOG_BYTES - 160:
            data = "日志达到大小上限；仍继续下载并在控制台显示进度。\n".encode("utf-8")
            self.full = True
        self.stream.write(data)
        self.stream.flush()
        self.size += len(data)


def communicate(process, timeout, log, stage):
    """Keep a fixed deadline even when Git keeps writing carriage-return updates."""
    started = time.monotonic()
    deadline, heartbeat = started + timeout, started + HEARTBEAT_SECONDS
    events = queue.Queue(maxsize=128)
    stopped = threading.Event()
    journal = Journal(log)
    tails = {"stdout": "", "stderr": ""}
    handlers = {}

    def emit(item):
        while not stopped.is_set():
            try:
                events.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def reader(name, stream):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""

        def consume(text):
            nonlocal pending
            for char in text:
                if char in "\r\n":
                    if pending:
                        emit((name, pending))
                        pending = ""
                elif len(pending) < 16384:
                    pending += char

        try:
            while not stopped.is_set():
                data = os.read(stream.fileno(), 4096)
                if not data:
                    break
                consume(decoder.decode(data))
            consume(decoder.decode(b"", final=True))
            if pending:
                emit((name, pending))
        except OSError:
            if not stopped.is_set():
                emit(("error", "无法读取 Git 子进程输出"))
        finally:
            emit(("done", name))

    threads = [
        threading.Thread(
            target=reader, args=(name, getattr(process, name)), name="chd-git-" + name, daemon=True
        )
        for name in tails
    ]
    try:
        journal.write(f"阶段：{stage}；总上限 {timeout:g} 秒")
        for thread in threads:
            thread.start()
        done = 0
        while done < 2:
            now = time.monotonic()
            if now >= deadline:
                raise StageTimeout(stage, timeout, log)
            if now >= heartbeat:
                print(
                    f"{stage}仍在进行/等待输出，已用 {now - started:.0f} 秒（上限 {timeout:g} 秒）",
                    flush=True,
                )
                heartbeat = now + HEARTBEAT_SECONDS
            try:
                name, text = events.get(timeout=min(0.1, deadline - now))
            except queue.Empty:
                continue
            if name == "done":
                done += 1
            elif name == "error":
                raise OSError(text)
            else:
                clean = safe_text(text)
                tails[name] = (tails[name] + clean + "\n")[-TAIL_CHARS:]
                journal.write(f"[{name}] {clean}")
                if name == "stderr" and PROGRESS.match(clean):
                    print(clean, flush=True)
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise StageTimeout(stage, timeout, log) from None
        journal.write(f"阶段结束：exit={process.returncode}；耗时 {time.monotonic() - started:.2f} 秒")
        return tails["stdout"], tails["stderr"]
    except BaseException as error:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, signal.SIG_IGN)
        error.output_tail = tails["stderr"][-4000:]
        stopped.set()
        terminate(process)
        process.wait()
        journal.write("阶段停止：" + (str(error) or type(error).__name__))
        raise
    finally:
        stopped.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=2)
        journal.stream.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
