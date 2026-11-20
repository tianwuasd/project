import json
from pathlib import Path
import subprocess
import sys

from packaging.version import Version


QUERY = (
    'import json, sys, numpy, torch; '
    'print(json.dumps(dict(executable=sys.executable, prefix=sys.prefix, paths=sys.path, '
    'numpy_version=numpy.__version__, numpy_file=numpy.__file__, torch_file=torch.__file__)))'
)


def inspect_environment(python):
    result = subprocess.check_output([python, '-I', '-B', '-c', QUERY], text=True, encoding='utf-8')
    record = json.loads(result)
    if Path(record['executable']).resolve() != Path(python).resolve():
        raise ValueError('MedSAM3解释器与显式配置不同')
    if Version(record['numpy_version']) >= Version('2'):
        raise ValueError('固定SAM3源码需要独立的numpy<2环境')
    prefix = Path(record['prefix']).resolve()
    for key in ('numpy_file', 'torch_file'):
        if not Path(record[key]).resolve().is_relative_to(prefix):
            raise ValueError('MedSAM3依赖来自解释器环境之外：' + key)
    return record


def verify_worker_environment(expected):
    import numpy
    import torch
    actual = dict(executable=sys.executable, numpy_version=numpy.__version__,
        numpy_file=numpy.__file__, torch_file=torch.__file__)
    if any(actual[key] != expected[key] for key in actual):
        raise ValueError('spawn子进程实际依赖路径与独立解释器预检查不同')
    return actual
