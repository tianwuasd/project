import errno
import json
import os
from pathlib import Path
import subprocess
import sys

from chd_stunet.acceleration.preprocess_cpu import interpreter_command


TEMPORARY = frozenset({errno.ENOENT, errno.ECONNRESET, errno.ETIMEDOUT, errno.ENETDOWN,
                       errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ESTALE})


class InputUnavailable(OSError):
    def __init__(self, paths, details):
        self.paths = sorted(set(paths))
        self.details = details
        super().__init__('MedSAM2 已审核输入暂不可读：'+', '.join(self.paths))


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
