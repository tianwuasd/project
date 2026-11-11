import os
import posixpath
from pathlib import Path, PurePosixPath

SERVER_WORKSPACE = PurePosixPath('/data5/zhougaowei/zhangruichen_workspace')
SERVER_DATA_ROOT = '/data_nas/zhangruichen/dataset1282_CHD_CT_back'


def _nas_path(path):
    # 在 Windows 验证 Linux 配置时也检查原始绝对路径。
    value = str(path).replace('\\', '/')
    candidates = (PurePosixPath(value).parts, PurePosixPath(posixpath.normpath(value)).parts)
    return any(parts and parts[0] == '/' and len(parts) > 1 and parts[1] == 'data_nas' for parts in candidates)


def assert_output_location(path):
    if _nas_path(path):
        raise ValueError('Output paths cannot be stored under /data_nas: ' + str(path))
    resolved = Path(path).expanduser().resolve()
    if _nas_path(resolved):
        raise ValueError('Resolved output path cannot be stored under /data_nas: ' + str(resolved))
    if os.name != 'nt' and str(resolved).startswith('/data5/'):
        workspace = Path(str(SERVER_WORKSPACE)).resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError('Server outputs must be inside ' + str(workspace))
    return resolved


def default_server_work_root():
    return str(SERVER_WORKSPACE / 'dataset1282_work')


def validate_input_output_paths(data_root, outputs):
    data = Path(data_root).expanduser().resolve() if data_root else None
    for name, value in outputs.items():
        out = assert_output_location(value)
        if data and (out == data or data in out.parents or out in data.parents):
            raise ValueError('Input/output paths overlap: ' + name)
