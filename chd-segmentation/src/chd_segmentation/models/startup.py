from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import stat

import psutil

from ..artifacts.store import read_json,write_json


def _has_progress(root):
    return (root/'summary.json').exists() or any(root.rglob('*.pt*'))


def check_start(root,model,prepared_identity,stage):
    root = Path(root)
    if not root.exists():
        return
    attributes = getattr(root.stat(),'st_file_attributes',0)
    if root.is_symlink() or attributes & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0):
        raise ValueError('训练阶段目录不能是链接')
    if _has_progress(root):
        raise FileExistsError('阶段已有模型、检查点或训练摘要，请使用resume或新的目录')
    if not (root/'startup.json').is_file():
        return
    record = read_json(root/'startup.json')
    if record['schema']!=1 or (record['model'],record['stage'])!=(model,stage):
        raise ValueError('初始化记录属于其他模型或阶段')
    last = record['attempts'][-1]
    owner = last['owner']
    if last['state']=='running' and psutil.pid_exists(owner['pid']) and psutil.Process(owner['pid']).create_time()==owner['birth']:
        raise RuntimeError('同一阶段的初始化进程仍在运行')


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
        # 阶段操作锁由调用方持有；退出时登记实际保存进度。
        attempt['state'] = 'finished' if finished else 'failed'
        attempt['committed_progress'] = _has_progress(root)
        write_json(path,record)
