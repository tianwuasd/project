"""Named immutable jobs, detached submission and PID identity checks (stdlib only).

Lock lifetime follows inherited descriptors, as in server-engineering's CHD reference.
Closing a parent descriptor must not explicitly unlock a surviving worker's descriptor.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import math
from uuid import uuid4

from model_catalog import parse_models, parse_folds
from server_data import digest, read_json, write_json

CODE = Path(__file__).resolve().parents[1]
TASKS = {'environment': '环境准备', 'prepare': '数据预处理', 'smoke': '所选模型短测',
         'train': '模型训练', 'predict': '模型预测与逐例评估', 'summarize': '已有评价汇总'}
GPU_TASKS = {'smoke', 'train', 'predict'}


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    return ''.join(c for c in str(value) if c >= ' ' and c != '\x7f')


def parse_tasks(value):
    values = value if isinstance(value, (tuple, list)) else str(value).replace('，', ',').replace(',', ' ').split()
    aliases = {str(i): name for i, name in enumerate(TASKS)}
    values = [aliases.get(str(v), v) for v in values]
    if not values or any(v not in TASKS for v in values):
        raise ValueError('请选择有效步骤：' + ','.join(TASKS))
    return [v for v in TASKS if v in values]


def identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in {'Z', 'X'}:
            return None
        return {'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'start': fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    return bool(record and record.get('identity') and identity(record.get('pid')) == record['identity'])


def active_processes(job):
    records = []
    for path in [Path(job) / 'state.json', *sorted((Path(job) / 'processes').glob('*.json'))]:
        try:
            item = read_json(path)
            if alive(item):
                records.append(item)
        except (OSError, ValueError, TypeError):
            continue
    return records


@contextmanager
def runtime_lock(runtime):
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    handle = (runtime / '.baseline_job.lock').open('a+b')
    try:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        yield handle
    except BlockingIOError as error:
        raise ValueError('此运行目录仍有活动任务或子进程；请查询，不重复提交') from error
    finally:
        handle.close()


def load_plan(job):
    job = Path(job).expanduser().resolve()
    value = read_json(job / 'plan.json')
    if Path(value['directory']).resolve() != job:
        raise ValueError('任务已移动或计划目录不符；可只读查看，不能直接续跑')
    state = read_json(job / 'state.json')
    if state.get('plan_sha256') != digest(job / 'plan.json'):
        raise ValueError('任务计划被改动，停止执行；历史结果仍可查看')
    return value


def history(runtime):
    """Read damaged neighbours independently; never rewrite last-job while browsing."""
    runtime = Path(runtime)
    paths = set()
    try:
        paths.update(read_json(runtime / 'history.json').get('jobs', []))
    except (OSError, ValueError, TypeError):
        pass
    result = []
    for raw in sorted(paths, reverse=True):
        job = Path(raw)
        try:
            plan, state = read_json(job / 'plan.json'), read_json(job / 'state.json')
            if Path(plan['runtime']).resolve() != runtime.resolve():
                continue
            status = state.get('status', 'unknown')
            active = active_processes(job)
            if active:
                status = 'running'
            elif status in {'running', 'queued', 'stopping', 'waiting_gpu'}:
                status = 'stale'
            result.append(dict(directory=str(job), name=plan.get('name') or job.name,
                               status=status, workspace=plan['workspace'], created=plan.get('created', '')))
        except (OSError, ValueError, KeyError, TypeError):
            result.append(dict(directory=str(job), name=job.name, status='记录缺失或损坏', created=''))
    seen = {r.get('workspace') for r in result}
    for old in legacy_configs(runtime):
        if old['work'] in seen:
            continue
        seen.add(old['work'])
        result.append(dict(directory=old['work'], workspace=old['work'], name=Path(old['work']).name,
                           status='旧工作区（只读查看/新任务输入）', created=old.get('created', '')))
    return sorted(result, key=lambda x: x['created'], reverse=True)


def legacy_configs(runtime):
    paths = [CODE / '.baseline_settings.json', *sorted((Path(runtime) / 'logs').glob('*.json'), reverse=True)]
    result = []
    for path in paths:
        try:
            item = read_json(path)
            if Path(item['base']).resolve() == Path(runtime).resolve() and Path(item['work']).is_dir():
                result.append(dict(item, created=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def latest(runtime):
    try:
        return Path(read_json(Path(runtime) / 'last-job.json')['directory'])
    except (OSError, ValueError, KeyError, TypeError):
        old = legacy_configs(runtime)
        return Path(old[0]['work']) if old else None


def snapshot(destination):
    for name in ['05_src', '06_configs', '04_data/manifests']:
        shutil.copytree(CODE / name, destination / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(CODE / 'bootstrap_server.sh', destination / 'bootstrap_server.sh')
    files = {str(p.relative_to(destination)): digest(p) for p in destination.rglob('*') if p.is_file()}
    write_json(destination.parent / 'code_manifest.json', files)


def validate_request(config):
    c = dict(config)
    c['methods'], c['folds'], c['tasks'] = parse_models(c['methods']), parse_folds(c['folds']), parse_tasks(c['tasks'])
    name = str(c.get('name', '')).strip()
    if not name or len(name) > 100 or name != clean(name):
        raise ValueError('任务名称需为1–100个可显示字符，不能包含控制字符')
    c['name'] = name
    memory = c.get('gpu_memory', 5)
    if isinstance(memory, bool) or not isinstance(memory, (int,float)) or not math.isfinite(memory) or memory < 2:
        raise ValueError('规划显存预算需为至少2 GiB的有限数值')
    if c['tasks'] != ['environment'] and not str(c.get('data', '')).strip():
        raise ValueError('必须显式填写原始数据目录')
    for key in ['runtime', 'results', 'data']:
        c[key] = str(Path(c[key]).expanduser().resolve())
    raw, runtime, results = (Path(c[k]) for k in ['data', 'runtime', 'results'])
    if c['tasks'] != ['environment'] and not raw.is_dir():
        raise ValueError('原始数据目录不存在')
    for left, right in [(raw, runtime), (raw, results), (runtime, results)]:
        if left == right or left in right.parents or right in left.parents:
            raise ValueError('原始数据、运行环境、结果根目录必须相互独立')
    for path in [runtime, results]:
        if path == Path(path.anchor) or path == Path.home() or path == CODE or path in CODE.parents or path.is_relative_to(CODE):
            raise ValueError('请选择本项目专用子目录')
    if not c.get('input_job') and 'prepare' not in c['tasks'] and any(t in c['tasks'] for t in ['smoke', 'train', 'predict']):
        raise ValueError('首次模型运行需要选择预处理，或选择已有输入任务')
    if 'predict' in c['tasks'] and 'train' not in c['tasks'] and not c.get('input_job'):
        raise ValueError('单独预测需要提供已训练的输入任务')
    if 'summarize' in c['tasks'] and 'predict' not in c['tasks'] and not c.get('input_job'):
        raise ValueError('单独汇总需要提供已预测的输入任务')
    if GPU_TASKS.intersection(c['tasks']):
        if not c.get('gpus') or not 1 <= int(c.get('gpu_count', 0)) <= len(c['gpus']) or len(c['gpus']) != len(set(c['gpus'])):
            raise ValueError('GPU任务需明确候选GPU和有效并发上限')
    return c


def launch(job, lock, foreground=False):
    plan = load_plan(job)
    executable = Path(plan['code']) / '05_src/task_worker.py'
    command = [sys.executable, '-u', '-B', str(executable), str(job), '--lock-fd', str(lock.fileno())]
    if os.name != 'posix':
        raise RuntimeError('实际后台任务在Linux运行；Windows只支持面板查询与软件测试')
    with (Path(job) / 'launcher.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock.fileno(),))
    write_json(Path(job) / 'processes/worker.json', dict(pid=process.pid, identity=identity(process.pid), role='worker'))
    print(f"已提交：{clean(plan['name'])}\n任务：{job}\n日志：{Path(job) / 'launcher.log'}\n提交成功不代表计算完成。", flush=True)
    if foreground:
        try:
            return process.wait()
        except KeyboardInterrupt:
            stop(job)
            return 130
    return 0


def submit(config, foreground=False):
    config = validate_request(config)
    runtime, results = Path(config['runtime']), Path(config['results'])
    with runtime_lock(runtime) as lock:
        legacy_state = runtime / 'job_state.json'
        if legacy_state.exists():
            legacy = read_json(legacy_state)
            if legacy.get('pid') and identity(legacy['pid']):
                raise ValueError('检测到旧入口记录的进程仍存在，先核查旧任务；未启动新任务')
        from task_inputs import inspect_input, CacheMismatch, source_lock
        source = None
        if config.get('input_job'):
            if config.get('preprocess_cache') == 'new' and 'prepare' in config['tasks']:
                if 'predict' in config['tasks'] and 'train' not in config['tasks']:
                    raise ValueError('重新预处理后预测需同时选择训练，不能静默搭配旧模型')
                config['cache_decision'] = '用户选择独立新建缓存，来源仅作记录'
            else:
                try:
                    with source_lock(config['input_job']):
                        source = inspect_input(config)
                except CacheMismatch as error:
                    if 'prepare' not in config['tasks']:
                        raise
                    if 'predict' in config['tasks'] and 'train' not in config['tasks']:
                        raise ValueError('缓存不兼容；必须新建包含训练的任务') from error
                    config['cache_decision'] = '缓存不能沿用，独立新建：' + str(error)
                    print(config['cache_decision'], flush=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:10]
        job = results / stamp
        job.mkdir(parents=True, exist_ok=False)
        snapshot(job / 'code')
        plan = dict(config, version=1, created=now(), directory=str(job), workspace=str(job / 'workspace'),
                    code=str(job / 'code'), source=source)
        write_json(job / 'plan.json', plan)
        write_json(job / 'state.json', dict(name=plan['name'], status='queued', plan_sha256=digest(job / 'plan.json'),
                   steps={t: {'status': 'pending'} for t in plan['tasks']}))
        try:
            old = read_json(runtime / 'history.json').get('jobs', [])
        except (OSError, ValueError):
            old = []
        write_json(runtime / 'history.json', {'jobs': list(dict.fromkeys([*old, str(job)]))})
        write_json(runtime / 'last-job.json', {'directory': str(job)})
        if config['tasks'] != ['environment']:
            write_json(runtime / 'settings.json', config)
        return launch(job, lock, foreground)


def resume(job, foreground=False, check='simple'):
    plan = load_plan(job)
    with runtime_lock(plan['runtime']) as lock:
        if active_processes(job):
            raise ValueError('任务进程仍存活，不能接续')
        from task_inputs import verify_resume
        verify_resume(job, check)
        state = read_json(Path(job) / 'state.json')
        if all(s['status'] == 'completed' for s in state['steps'].values()):
            print('所选步骤已经完成；如需重新计算，请使用独立重跑。')
            return 0
        state.update(status='queued', resume_check=check)
        write_json(Path(job) / 'state.json', state)
        write_json(Path(plan['runtime']) / 'last-job.json', {'directory': str(job)})
        return launch(job, lock, foreground)


def stop(job):
    records = active_processes(job)
    if not records:
        raise ValueError('没有身份匹配的活动进程；未发送任何信号')
    # Signal coordinators; they stop only their registered children and retain commits.
    unique = {r['pid']: r for r in records}
    for record in unique.values():
        if alive(record):
            os.kill(record['pid'], signal.SIGTERM)
    print('已请求中断，正在退出；保留最后完整检查点。请查询状态确认退出。')


def append_tasks(job, tasks, foreground=False, check='simple'):
    """Append missing downstream operations, retaining the original model selection."""
    plan = load_plan(job)
    added = [t for t in parse_tasks(tasks) if t not in plan['tasks']]
    if not added:
        raise ValueError('没有缺失步骤；已经完成的步骤请使用独立重跑')
    if GPU_TASKS.intersection(added) and (not plan.get('gpus') or not 1 <= plan.get('gpu_count', 0) <= len(plan['gpus'])):
        raise ValueError('原任务没有获准GPU范围；请新建任务、选择此历史输入并明确GPU，不能直接追加计算')
    order = list(TASKS)
    if min(map(order.index, added)) <= max(map(order.index, plan['tasks'])):
        raise ValueError('只能追加尚未执行的后续步骤；重算上游请新建重跑任务')
    with runtime_lock(plan['runtime']) as lock:
        if active_processes(job):
            raise ValueError('任务仍在运行')
        from task_inputs import verify_resume
        verify_resume(job, check)
        job = Path(job)
        state = read_json(job / 'state.json')
        if state['status'] != 'completed':
            raise ValueError('先续跑完成原步骤，再追加后续步骤')
        revision = job / 'revisions' / str(time.time_ns())
        revision.mkdir(parents=True)
        for name in ['plan.json', 'state.json']:
            shutil.copy2(job / name, revision / name)
        plan['tasks'] = parse_tasks([*plan['tasks'], *added])
        write_json(job / 'plan.json', plan)
        state.update(status='queued', plan_sha256=digest(job / 'plan.json'), resume_check=check)
        state['steps'].update({t: {'status': 'pending'} for t in added})
        write_json(job / 'state.json', state)
        write_json(Path(plan['runtime']) / 'last-job.json', {'directory': str(job)})
        return launch(job, lock, foreground)
