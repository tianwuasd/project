from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import stat

import psutil

from ..artifacts.store import file_digest,read_json,write_json


def _inventory(root):
    result = {}
    for path in root.rglob('*'):
        attributes = getattr(path.stat(),'st_file_attributes',0)
        if path.is_symlink() or attributes & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0):
            raise ValueError('初始化目录包含链接，停止重试')
        if path.is_file() and path!=root/'startup.json':
            result[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size,sha256=file_digest(path))
    return result


def _has_progress(root):
    return (root/'summary.json').exists() or any(root.rglob('*.pt*'))


def check_start(root,model,prepared_identity,stage):
    root = Path(root)
    if not root.exists():
        return
    if root.is_symlink() or not (root/'startup.json').is_file():
        raise FileExistsError('已有阶段目录缺少可核验的初始化失败记录')
    record = read_json(root/'startup.json')
    if record['schema']!=1 or (record['model'],record['prepared_identity'],record['stage'])!=(model,prepared_identity,stage):
        raise ValueError('初始化重试的模型、阶段或准备描述不同')
    if record['attempts'][-1]['state']!='failed' or _has_progress(root):
        raise FileExistsError('阶段已有检查点、摘要或活动初始化记录，请使用resume')
    if record['attempts'][-1]['files']!=_inventory(root):
        raise ValueError('初始化失败目录中的文件已修改或新增，停止重试')


@contextmanager
def initialization_attempt(root,config,prepared_identity,stage,resume):
    root = Path(root)
    if not resume:
        check_start(root,config['model'],prepared_identity,stage)
    root.mkdir(parents=True,exist_ok=True)
    path = root/'startup.json'
    record = read_json(path) if path.exists() else dict(schema=1,model=config['model'],
        stage=stage,prepared_identity=prepared_identity,attempts=[])
    process = psutil.Process()
    attempt = dict(state='running',resume=resume,config=deepcopy(config),
        owner=dict(pid=process.pid,birth=process.create_time()))
    record['attempts'].append(attempt)
    write_json(path,record)
    finished = False
    try:
        yield
        finished = True
    finally:
        # 调用方先结束全部已登记子进程；只登记本阶段的独占文件。
        attempt['state'] = 'finished' if finished else 'failed'
        attempt['committed_progress'] = _has_progress(root)
        attempt['files'] = _inventory(root) if not finished and not attempt['committed_progress'] else None
        write_json(path,record)
