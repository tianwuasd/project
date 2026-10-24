"""Small local registry. Reading state never creates files or loads imaging code."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def dispatch_lock(root):
    """One dispatch at a time across terminals using the same registry."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.dispatch.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('另一个终端正在提交任务，请稍后重试') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def job_status(output):
    output = Path(output)
    runtime = output.with_name(output.name + '.runtime')
    identity = read(runtime / 'process.json', {})
    alive = False
    if identity:
        try:
            import psutil
            process = psutil.Process(identity['pid'])
            alive = (abs(process.create_time() - identity['created']) < .01
                     and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
        except ImportError:
            alive = None
        except (KeyError, TypeError, OSError):
            alive = False
        except psutil.Error:
            alive = False
    worker = read(runtime / 'worker.json', {})
    progress = read(output / 'progress.json', {})
    state = worker.get('state', 'not_submitted')
    if alive is False and state in {'submitted', 'checking', 'running'}:
        state = 'interrupted'
    return {'state': state, 'alive': alive, 'worker': worker, 'progress': progress,
            'runtime': str(runtime), 'output': str(output),
            'resources': read(runtime / 'resources.json', {}),
            'stop_requested': (runtime / 'stop.request').exists()}


class Registry:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def entries(self):
        rows = read(self.root / 'history.json', [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not row.get('output') for row in rows):
            raise ValueError('历史索引损坏，请保留 history.json 供核查')
        return [dict(row, name=row.get('name') or Path(row['output']).name) for row in rows]

    def ensure_idle(self):
        for entry in self.entries():
            status = job_status(entry['output'])
            if status['alive'] is not False:
                raise RuntimeError('已有任务运行或状态无法确认：' + entry['name'])

    def register(self, entry):
        entries = self.entries()
        write(self.root / 'history.json', [entry, *entries])


def separate(a, b):
    a, b = Path(a).expanduser().resolve(), Path(b).expanduser().resolve()
    return a != b and a not in b.parents and b not in a.parents


def validate_paths(settings, registry):
    source = settings['input']
    results = settings['results']
    if not source or not results:
        raise ValueError('请先设置 HU NIfTI 输入目录和结果根目录')
    if not Path(source).expanduser().is_dir():
        raise ValueError('输入目录不存在')
    for a, b in [(source, results), (source, registry), (results, registry)]:
        if not separate(a, b):
            raise ValueError('输入、结果根目录和面板运行目录必须彼此独立，不能互相包含')
    if settings.get('scratch'):
        for other in (source, results, registry):
            if not separate(settings['scratch'], other):
                raise ValueError('自定义临时根目录必须与输入、结果、面板目录彼此独立')
