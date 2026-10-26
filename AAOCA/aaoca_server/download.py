"""Bounded Git streaming: drain both pipes, render CR progress, reap descendants."""
import codecs
from collections import deque
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time


class GitFailure(ValueError):
    """A classified diagnostic that never includes raw credential-helper output."""


def diagnostic(stderr, *, local_only=False):
    if local_only:
        return '本地归档失败，隐式联网已禁止；请检查缓存对象完整性和本地 Git 状态'
    text = stderr.lower()
    categories = [
        (('host key verification failed', 'remote host identification has changed'), 'SSH 主机校验失败；请核验已保存主机指纹'),
        (('permission denied', 'authentication failed', 'repository not found', 'could not read username'),
         '仓库认证/权限检查失败；请核对原密钥或 HTTPS 凭据'),
        (('could not resolve', 'name or service not known'), '主机名解析失败；请检查网络与代理'),
        (('proxy error', 'proxy connect', 'connect to proxy', 'proxy authentication'), '代理连接失败；请检查本机代理与出口'),
        (('banner exchange', 'connection timed out', 'connection refused', 'operation timed out'),
         '连接/握手失败；请检查代理出口，尚不能据此判断密钥失效'),
        (('early eof', 'index-pack', 'connection reset', 'unexpected disconnect', 'recv failure'),
         '下载连接中断；保留缓存，检查线路后重试'),
    ]
    for signatures, message in categories:
        if any(value in text for value in signatures):
            return message
    return 'Git 操作失败；请核对网络、仓库状态和认证配置（原始认证输出不回显）'


def safe_progress(line):
    # Only Git's transfer counters are publishable. Other helper/remote output
    # is classified, never copied into the terminal or persistent log.
    line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip()
    match = re.match(r'^((?:remote:\s*)?(?:Enumerating|Counting|Compressing|Receiving|Resolving|Unpacking|Updating|Checking out) (?:objects|deltas|files)):\s*(.*)', line)
    if not match:
        return None
    label, body = match.groups()
    counters = re.match(r'(\d{1,3}%\s*\(\d{1,12}/\d{1,12}\)|\d{1,12})(?=$|[,\s])', body)
    if not counters:
        return None
    result = label + ': ' + counters[0]
    tail = body[counters.end():]
    size = r'\d{1,12}(?:\.\d{1,6})?\s+(?:bytes|[KMGT]i?B)'
    transfer = re.match(r',\s*(' + size + r')(?:\s*\|\s*(' + size + r'/s))?', tail)
    if transfer:
        result += ', ' + transfer[1]
        if transfer[2]:
            result += ' | ' + transfer[2]
        tail = tail[transfer.end():]
    if re.match(r',\s*done\.', tail):
        result += ', done.'
    return result


def reap(child):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    else:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
    finally:
        if os.name != 'nt':
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=3)


def run(args, *, env=None, cwd=None, timeout=900, emit=print, log=None, heartbeat=5):
    started = time.monotonic()
    deadline = started + timeout
    child = None
    handlers, cancelled = {}, []
    threads = []
    messages = queue.Queue()
    stdout, stderr = [], deque(maxlen=32)
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    pending = ''
    stream = None

    def report(text):
        if stream:
            stream.write(text + '\n')
            stream.flush()
        emit(text)

    def interrupt(sig, frame):
        cancelled.append(True)
        if child is not None:
            raise KeyboardInterrupt

    def reader(pipe, name):
        try:
            while True:
                data = os.read(pipe.fileno(), 8192)
                if not data:
                    break
                messages.put((name, data))
        finally:
            messages.put((name, None))

    try:
        if log is not None:
            stream = Path(log).open('a', encoding='utf-8')
            if os.name == 'posix':
                os.chmod(log, 0o600)
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupt)
        child = subprocess.Popen(args, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=os.name != 'nt')
        if cancelled:
            raise KeyboardInterrupt
        for name, pipe in [('stdout', child.stdout), ('stderr', child.stderr)]:
            thread = threading.Thread(target=reader, args=(pipe, name), daemon=True)
            thread.start()
            threads.append(thread)
        ended = 0
        last_display = started
        while ended < 2 or child.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            try:
                name, chunk = messages.get(timeout=min(.1, remaining))
            except queue.Empty:
                name, chunk = None, None
            if name and chunk is None:
                ended += 1
            elif name == 'stdout':
                stdout.append(chunk)  # May be a binary archive: never render it.
            elif name == 'stderr':
                stderr.append(chunk)
                pending += decoder.decode(chunk)
                parts = re.split(r'[\r\n]', pending)
                pending = parts.pop()[-8192:]
                for line in parts:
                    safe = safe_progress(line)
                    if safe:
                        report(safe)
                        last_display = time.monotonic()
            if time.monotonic() - last_display >= heartbeat:
                report(f'等待 Git 输出；本阶段已用 {time.monotonic() - started:.1f} 秒，上限 {timeout:.0f} 秒')
                last_display = time.monotonic()
        safe = safe_progress(pending + decoder.decode(b'', final=True))
        if safe:
            report(safe)
        if child.wait() != 0:
            reason = diagnostic(b''.join(stderr).decode('utf-8', 'replace'),
                                local_only=bool(env is not None and env.get('GIT_ALLOW_PROTOCOL') == ''))
            report(reason)
            raise GitFailure(reason)
        return b''.join(stdout)
    except BaseException:
        # Repeated interrupts cannot release the caller's lock while helpers live.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            reap(child)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=2)
        if child is not None:
            for pipe in (child.stdout, child.stderr):
                if pipe:
                    pipe.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if stream:
            stream.close()
