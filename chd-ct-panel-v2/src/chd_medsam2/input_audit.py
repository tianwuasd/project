import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
from threading import Thread
import time

from chd_stunet.acceleration.preprocess_cpu import interpreter_command
from chd_stunet.config import ROOT

from .io_recovery import InputUnavailable, input_path


DEFAULT_TIMEOUT = 120


class AuditCancelled(RuntimeError):
    pass


def read_events(stream, events):
    try:
        for line in stream:
            events.put(line)
    finally:
        events.put(None)


def audit_prepared(root, fold, *, smoke=False, train_cases=(), val_cases=(),
                   timeout=DEFAULT_TIMEOUT, progress=None, cancelled=None):
    root = Path(os.path.abspath(root))
    request = dict(root=str(root), fold=fold, smoke=smoke, train_cases=list(train_cases), val_cases=list(val_cases))
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES='')
    command = [interpreter_command(env), '-u', '-m', 'chd_medsam2.input_audit']
    process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True, encoding='utf-8')
    events = SimpleQueue()
    Thread(target=read_events, args=(process.stdout, events), daemon=True).start()
    latest = dict(path=str(root/'manifest.json'), bytes_read=0, audit_pid=process.pid)
    last_progress = last_report = time.monotonic()
    reported_bytes = False
    try:
        with process.stdin as stream:
            stream.write(json.dumps(request))
        if progress is not None:
            progress(latest)
        while True:
            if cancelled is not None and cancelled():
                raise AuditCancelled('MedSAM2 输入校验已停止')
            try:
                line = events.get(timeout=min(1, max(.01, timeout-(time.monotonic()-last_progress))))
            except Empty:
                if time.monotonic()-last_progress >= timeout:
                    raise InputUnavailable([latest['path']], dict(
                        latest, reason='input_read_timeout', timeout_seconds=timeout))
                continue
            if line is None:
                raise RuntimeError(f'MedSAM2 输入校验进程退出且未提交结果，PID {process.pid}')
            message = json.loads(line)
            kind = message.pop('kind')
            if kind == 'progress':
                latest = dict(message, audit_pid=process.pid)
                last_progress = time.monotonic()
                if progress is not None and (last_progress-last_report >= 10 or (
                        not reported_bytes and latest['bytes_read'] > 0)):
                    progress(latest)
                    last_report = last_progress
                    reported_bytes |= latest['bytes_read'] > 0
                continue
            if process.wait(timeout=5):
                raise RuntimeError(f'MedSAM2 输入校验进程返回错误，PID {process.pid}')
            if kind == 'ready':
                return message
            if kind == 'unavailable':
                raise InputUnavailable([message['path']], message)
            if kind == 'invalid':
                raise ValueError(message['error'])
            raise ValueError('未知的 MedSAM2 输入校验结果')
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


def send(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields)), flush=True)


def run():
    from .preprocessing import open_prepared

    request = json.load(sys.stdin)
    root, fold = request.pop('root'), request.pop('fold')
    try:
        root, manifest, cases, metadata, train, val, window = open_prepared(
            root, fold, **request,
            progress=lambda path, size: send('progress', path=str(path), bytes_read=size))
    except OSError as error:
        unavailable = input_path(error, root)
        if unavailable is None:
            raise
        send('unavailable', path=unavailable, error=str(error), reason='input_unavailable')
        return
    except ValueError as error:
        send('invalid', error=str(error))
        return
    send('ready', root=str(root), manifest=manifest, cases=cases, metadata=metadata,
         train=train, val=val, window=window)


if __name__ == '__main__':
    run()
