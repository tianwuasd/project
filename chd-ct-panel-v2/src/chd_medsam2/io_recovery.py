import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from chd_stunet.acceleration.preprocess_cpu import interpreter_command
from chd_stunet.observability.runlog import write_json

from .contracts import read_json


TEMPORARY = frozenset({errno.ENOENT, errno.ECONNRESET, errno.ETIMEDOUT, errno.ENETDOWN,
                       errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ESTALE})


class InputUnavailable(OSError):
    def __init__(self, paths, details):
        self.paths = sorted(set(paths))
        self.details = details
        super().__init__('MedSAM2 已审核输入暂不可读：'+', '.join(self.paths))


class InputReadWatch:
    def __init__(self, recovery, generation, rank, lane):
        self.marker = Path(recovery)/'control'/f'input-read-{generation}-{rank}-{lane}.json'
        self.record = None

    def __enter__(self):
        return self.note

    def __exit__(self, *_):
        if self.record is not None:
            write_json(self.marker, None)
            self.record = None

    def note(self, path, frame=None):
        now = time.monotonic()
        name = str(path)
        if (self.record is None or self.record['path'] != name
                or (self.record['frame'] is None and frame is not None)
                or now-self.record['progress_at'] >= .5):
            self.record = dict(path=name, frame=None if frame is None else int(frame),
                               progress_at=now, pid=os.getpid())
            write_json(self.marker, self.record)


def expired_read(recovery, generation, workers, timeout):
    # 世代和实际 Popen 身份限定了可处理的计算进程。
    for rank, (process, _) in enumerate(workers):
        if process.poll() is not None:
            continue
        for lane in ('train', 'prefetch', 'validation'):
            marker = InputReadWatch(recovery, generation, rank, lane).marker
            if not marker.is_file():
                continue
            record = read_json(marker)
            if (record and record['pid'] == process.pid
                    and time.monotonic()-record['progress_at'] >= timeout
                    and read_json(marker) == record):
                return dict(record, rank=rank, lane=lane, reason='input_read_timeout',
                            seconds_without_progress=time.monotonic()-record['progress_at'])
    return None


def input_path(error, prepared):
    if error.errno not in TEMPORARY or not error.filename:
        return None
    root = Path(prepared)
    path = Path(os.path.abspath(error.filename))
    if not path.is_relative_to(root):
        return None
    parts = path.relative_to(root).parts
    if not parts:
        return str(root/'manifest.json')
    if parts in (('manifest.json',), ('summary.json',)):
        return str(path)
    if len(parts)==3 and parts[0]=='cases' and parts[-1] in ('ct.npy', 'labels.npy', 'complete.json'):
        return str(path)
    return None


def check_readable(code, paths):
    # 独立进程只试读首字节，正式 worker 重新校验完整文件摘要。
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(code/'src'), CUDA_VISIBLE_DEVICES='')
    command = [interpreter_command(env), '-m', 'chd_medsam2.io_recovery']
    try:
        result = subprocess.run(command, input=json.dumps(paths), text=True, capture_output=True,
                                encoding='utf-8', env=env, timeout=10, check=True)
    except subprocess.TimeoutExpired:
        return {'readable': False, 'reason': 'probe_timeout'}
    return json.loads(result.stdout)


def probe(paths):
    for name in paths:
        try:
            with Path(name).open('rb') as stream:
                if not stream.read(1):
                    return {'readable': False, 'fatal': True, 'reason': 'empty_file', 'path': name}
        except OSError as error:
            return {'readable': False, 'fatal': error.errno not in TEMPORARY,
                    'reason': str(error), 'path': name}
    return {'readable': True}


if __name__ == '__main__':
    print(json.dumps(probe(json.load(sys.stdin))))
