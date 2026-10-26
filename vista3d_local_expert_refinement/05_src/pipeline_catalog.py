"""Fresh, explicit stage-job index and recoverable, identity-checked trash.

Only this module's index is discovered. A trash intent is committed before a
rename, so interruption never leaves an unrecorded destination or a selectable
job that has moved. Readers do not create a runtime, lock, or index.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from uuid import uuid4

from task_runtime import active_processes, load_plan


STAGES = frozenset({'prepare', 'train', 'predict', 'assessment'})
SCHEMA = 2


def _path(value):
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError('任务目录必须为绝对路径')
    path = Path(value).expanduser()
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('任务目录必须为无上级跳转的绝对路径')
    # resolve() alone would silently accept a symlink or Windows junction.
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('任务目录不能包含符号链接或目录联接')
    resolved = path.resolve()
    if os.name == 'nt':
        # Windows may retain the long-path prefix when another handle holds a
        # byte lock; it names the same file and is not a redirected path.
        raw = str(resolved)
        if raw.startswith('\\\\?\\UNC\\'):
            resolved = Path('\\\\' + raw[8:])
        elif raw.startswith('\\\\?\\'):
            resolved = Path(raw[4:])
    if resolved != path:
        raise ValueError(f'任务目录解析后身份不符：{path} -> {resolved}')
    return resolved


def _runtime(value):
    return _path(Path(value).expanduser().absolute())


def _index_path(runtime):
    return runtime / 'pipeline_v2' / 'index.json'


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _empty():
    return dict(schema=SCHEMA, jobs={}, selected=None)


def _read_index(runtime, *, writing=False):
    path = _index_path(runtime)
    try:
        _path(path)
        value = _read_json(path)
        if not isinstance(value, dict) or value.get('schema') != SCHEMA or not isinstance(value.get('jobs'), dict):
            raise ValueError('新版任务索引格式损坏')
        return value
    except FileNotFoundError:
        return _empty()
    except (OSError, ValueError, TypeError):
        if writing:
            raise ValueError('新版任务索引损坏，未覆盖原索引') from None
        return _empty()


def _write_index(runtime, value):
    path = _index_path(runtime)
    _path(path)
    temp = path.with_name('.index-' + uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _file_lock(path, *, wait=True):
    _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    locked = False
    deadline = time.monotonic() + (5 if wait else 0)
    try:
        if os.name != 'posix':
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b'0')
                handle.flush()
        while True:
            try:
                if os.name == 'posix':
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise ValueError('新版任务目录正在使用，请稍后重试') from error
                time.sleep(.025)
        yield
    finally:
        if locked:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
            else:
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _catalog_lock(runtime):
    return _file_lock(runtime / 'pipeline_v2' / '.catalog.lock')


def _identity(path):
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('任务目录不存在或不是目录')
    return [info.st_dev, info.st_ino]


def _upstreams(plan):
    values = plan.get('upstream_jobs', [])
    if not isinstance(values, list):
        raise ValueError('上游任务必须为目录列表')
    values = list(values)
    if plan.get('input_job'):
        values.append(plan['input_job'])
    return list(dict.fromkeys(str(_path(value)) for value in values))


def _guard_process_metadata(job):
    """Check only this known job's shallow metadata before the shared probe."""
    state = _path(job / 'state.json')
    if state.lstat().st_nlink != 1:
        raise ValueError('任务状态记录不能与其他文件共享硬链接')
    processes = _path(job / 'processes')
    if not processes.exists():
        return
    if not processes.is_dir():
        raise ValueError('任务进程记录目录损坏')
    for candidate in processes.glob('*.json'):
        candidate = _path(candidate)
        info = candidate.lstat()
        if candidate.parent != processes or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('任务进程记录不能重定向到其他目录或文件')


def _plan_record(runtime, job):
    job = _path(job)
    # These are the only job files read; no discovery or legacy fallback.
    _path(job / 'plan.json')
    _path(job / 'state.json')
    _guard_process_metadata(job)
    try:
        plan = load_plan(job)
    except (AttributeError, TypeError, KeyError) as error:
        raise ValueError('任务冻结计划或状态格式损坏') from error
    if not isinstance(plan, dict) or plan.get('catalog_schema') != SCHEMA:
        raise ValueError('仅能登记新版独立阶段任务')
    tasks = plan.get('tasks')
    if (not isinstance(tasks, list) or len(tasks) != 1
            or not isinstance(tasks[0], str) or tasks[0] not in STAGES):
        raise ValueError('新版任务必须恰好包含一个独立阶段')
    stage = tasks[0]
    if _path(plan['runtime']) != runtime or _path(plan['directory']) != job:
        raise ValueError('任务与当前运行目录不符')
    root = _path(plan.get('results_root', plan.get('results')))
    if 'results_root' in plan and 'results' in plan and _path(plan['results']) != root:
        raise ValueError('结果根目录记录不一致')
    if root.name != stage or root.parent.name != 'pipeline_v2' or job.parent != root:
        raise ValueError('新版任务必须为 pipeline_v2/阶段 下的直接子目录')
    if job == runtime or job.is_relative_to(runtime) or runtime.is_relative_to(job):
        raise ValueError('不能登记运行环境目录作为任务')
    if plan.get('data'):
        data = _path(plan['data'])
        if data == root or data.is_relative_to(root) or root.is_relative_to(data):
            raise ValueError('结果目录必须独立于原始数据')
    workspace = _path(plan['workspace'])
    if not workspace.is_relative_to(job) or workspace == job:
        raise ValueError('工作区必须位于任务目录内')
    if any(not isinstance(plan.get(field, ''), str) for field in ('name', 'created')):
        raise ValueError('任务名称或时间记录格式损坏')
    return dict(directory=str(job), results_root=str(root), stage=stage,
                runtime=str(runtime), workspace=str(workspace),
                name=plan.get('name') or job.name, created=plan.get('created', ''),
                upstream_jobs=_upstreams(plan), plan_sha256=hashlib.sha256((job / 'plan.json').read_bytes()).hexdigest(),
                root_identity=_identity(root), job_identity=_identity(job), state='live')


def _record_path(runtime, key, record):
    if not isinstance(record, dict) or record.get('runtime') != str(runtime):
        raise ValueError('任务登记身份无效')
    if (any(not isinstance(record.get(field), str) for field in
            ('directory', 'results_root', 'stage', 'workspace', 'name', 'created', 'plan_sha256'))
            or not isinstance(record.get('upstream_jobs'), list)
            or any(not isinstance(value, str) for value in record['upstream_jobs'])
            or record.get('state') not in {'live', 'trashing', 'trashed'}):
        raise ValueError('任务登记字段损坏')
    job, root = _path(record['directory']), _path(record['results_root'])
    stage = record['stage']
    if (key != str(job) or stage not in STAGES or root.name != stage
            or root.parent.name != 'pipeline_v2' or job.parent != root
            or _identity(root) != record['root_identity']):
        raise ValueError('任务登记路径或结果根目录身份不符')
    return job


def _valid_record(runtime, key, record):
    job = _record_path(runtime, key, record)
    if record.get('state') != 'live' or _identity(job) != record['job_identity']:
        raise ValueError('任务已删除或目录身份改变')
    current = _plan_record(runtime, job)
    if current != record:
        raise ValueError('冻结计划与任务登记身份不符')
    return job


def _stored_path(value):
    """Compare pinned ownership without resolving or opening its filesystem."""
    if not isinstance(value, str) or not value:
        raise ValueError('登记目录所有权记录损坏')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('登记目录所有权记录损坏')
    return path


def _check_ownership(index, key, job):
    for other, record in index['jobs'].items():
        if other == key:
            continue
        if not isinstance(record, dict) or record.get('directory') != other:
            raise ValueError('其他任务登记目录所有权损坏，未移动任务')
        owned = [_stored_path(record['directory'])]
        if 'trash_directory' in record:
            owned.append(_stored_path(record['trash_directory']))
        if any(job == path or job.is_relative_to(path) or path.is_relative_to(job) for path in owned):
            raise ValueError('任务目录与其他登记任务或可恢复目录重叠')


def register(runtime, job):
    """Register one verified frozen plan; repeat registration cannot rebind it."""
    runtime = _runtime(runtime)
    with _catalog_lock(runtime):
        index = _read_index(runtime, writing=True)
        record = _plan_record(runtime, job)
        key = record['directory']
        if key in index['jobs'] and index['jobs'][key] != record:
            raise ValueError('任务已经登记，不能重新绑定或恢复已删除任务')
        _check_ownership(index, key, Path(key))
        for upstream in record['upstream_jobs']:
            if upstream == key or upstream not in index['jobs']:
                raise ValueError('上游任务必须来自新版索引，且不能引用自身')
            _valid_record(runtime, upstream, index['jobs'][upstream])
        index['jobs'][key] = record
        index['selected'] = key
        _write_index(runtime, index)


def select(runtime, job):
    runtime = _runtime(runtime)
    with _catalog_lock(runtime):
        index = _read_index(runtime, writing=True)
        key = str(_path(job))
        if key not in index['jobs']:
            raise ValueError('任务不在新版索引中')
        _valid_record(runtime, key, index['jobs'][key])
        index['selected'] = key
        _write_index(runtime, index)


def _row(record, status):
    return {key: record[key] for key in ('directory', 'name', 'workspace', 'created', 'stage', 'upstream_jobs')} | {'status': status}


def _rows(runtime, index):
    rows = []
    for key, record in index['jobs'].items():
        try:
            job = _record_path(runtime, key, record)
            if record.get('state') != 'live':
                continue
            try:
                _valid_record(runtime, key, record)
                state = _read_json(job / 'state.json')
                if not isinstance(state, dict):
                    raise ValueError('任务状态损坏')
                status = state.get('status', 'unknown')
                # Only a validated, registered new job can reach the process probe.
                if active_processes(job):
                    status = 'running'
                elif status in {'running', 'queued', 'stopping', 'waiting_gpu'}:
                    status = 'stale'
            except (OSError, ValueError, KeyError, TypeError):
                status = '记录缺失或损坏'
            rows.append(_row(record, status))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: (row['created'], row['directory']), reverse=True)


def history(runtime):
    runtime = _runtime(runtime)
    return _rows(runtime, _read_index(runtime))


def latest(runtime):
    runtime = _runtime(runtime)
    index = _read_index(runtime)
    keys = [index.get('selected')] + [row['directory'] for row in _rows(runtime, index)]
    for key in dict.fromkeys(key for key in keys if isinstance(key, str)):
        try:
            return _valid_record(runtime, key, index['jobs'][key])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _dependents(runtime, index, key):
    rows = []
    for other, record in index['jobs'].items():
        if other == key:
            continue
        if not isinstance(record, dict):
            continue
        upstreams = record.get('upstream_jobs', [])
        if not isinstance(upstreams, list):
            raise ValueError('下游任务引用记录损坏，不能确认依赖')
        if key not in upstreams:
            continue
        if record.get('state') == 'trashed':
            continue
        if record.get('state') == 'trashing' and _completed_move(runtime, other, record):
            continue
        # Registered provenance remains a blocker when its downstream files,
        # root identity, or display metadata have become unreadable/damaged.
        rows.append(dict(directory=other, name=record.get('name', Path(other).name),
                         workspace=record.get('workspace', ''), created=record.get('created', ''),
                         stage=record.get('stage', ''), upstream_jobs=record.get('upstream_jobs', []),
                         status=record.get('state', '记录缺失或损坏')))
    return rows


def _completed_move(runtime, key, record):
    try:
        source = _record_path(runtime, key, record)
        destination = _path(record['trash_directory'])
        root = Path(record['results_root'])
        return (destination.parent == root / '.trash' and not os.path.lexists(source)
                and _identity(destination) == record['job_identity'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _move_not_started(runtime, key, record, destination):
    try:
        _valid_record(runtime, key, record)
        _path(destination)
        return not os.path.lexists(destination)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def dependents(runtime, job):
    runtime = _runtime(runtime)
    index = _read_index(runtime)
    key = str(_path(job))
    if key not in index['jobs']:
        raise ValueError('任务不在新版索引中')
    _record_path(runtime, key, index['jobs'][key])
    return _dependents(runtime, index, key)


def trash(runtime, job):
    """Move a registered idle leaf job into its result root's .trash directory."""
    runtime = _runtime(runtime)
    # Same lock/order as submit/resume: prevent a launch racing the final probe.
    with _file_lock(runtime / 'pipeline_v2' / '.baseline_job.lock', wait=False), _catalog_lock(runtime):
        index = _read_index(runtime, writing=True)
        key = str(_path(job))
        if key not in index['jobs']:
            raise ValueError('任务不在新版索引中，未移动任何文件')
        record = index['jobs'][key]
        source = _valid_record(runtime, key, record)
        if active_processes(source):
            raise ValueError('任务仍有活动进程，不能删除')
        if _dependents(runtime, index, key):
            raise ValueError('仍有未删除的下游任务，不能删除')
        _check_ownership(index, key, source)
        recycle = _path(Path(record['results_root']) / '.trash')
        recycle.mkdir(exist_ok=True)
        destination = _path(recycle / (source.name + '_' + uuid4().hex))
        pending = dict(record, state='trashing', trash_directory=str(destination),
                       trashed_at=datetime.now(timezone.utc).isoformat())
        previous_selected = index.get('selected')
        index['jobs'][key] = pending
        if index.get('selected') == key:
            index['selected'] = None
        _write_index(runtime, index)
        try:
            # Recheck path identities immediately before the only filesystem move.
            _valid_record(runtime, key, record)
            _check_ownership(index, key, source)
            _path(destination)
            os.rename(source, destination)
        except BaseException:
            # An interruption may arrive after rename has actually succeeded.
            # Restore selection only if identity checks prove no move occurred;
            # otherwise the persisted intent keeps the destination recoverable.
            if _move_not_started(runtime, key, record, destination):
                index['jobs'][key] = record
                index['selected'] = previous_selected
                _write_index(runtime, index)
            raise
        index['jobs'][key] = dict(pending, state='trashed')
        try:
            _write_index(runtime, index)
        except OSError as error:
            raise OSError(f'任务已移入可恢复目录 {destination}；索引保留删除意图') from error
        return destination
