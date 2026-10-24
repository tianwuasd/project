"""Named immutable jobs for the independent terminal workbench (stdlib only)."""
import hashlib
import math
import re
import shutil
import sys
import uuid
from pathlib import Path

from .managed import ManagedController
from .operations import python_path, runtime_env
from .schema import read_manifest
from .service_state import (
    CODE_ROOT,
    alive,
    detached,
    file_lock,
    managed_release,
    private_dir,
    process_identity,
    read_json,
    tail,
    terminate_owned,
    timestamp,
    write_json,
)

DEFAULTS = {'manifest': '', 'results': '', 'mode': 'cpu', 'devices': [], 'gpu_count': 1,
            'min_gpu_gib': 8, 'cpu_limit': None, 'memory_gib': None, 'workers': None,
            'threads': None, 'io_workers': None, 'preprocess_output': '/data_nas/zhangruichen/aaoca_preprocessed'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(job):
    job = Path(job)
    state = read_json(job / 'state.json', {})
    if not isinstance(state, dict):
        return {'status': 'invalid', 'active': False}
    state.setdefault('status', 'unknown')
    owners = [job / 'owner.json', job / 'child.json', *job.glob('processes/*.json')]
    request = read_json(job / 'request.json', {})
    if isinstance(request, dict) and request.get('output'):
        owners.extend(Path(request['output']).glob('cases/*/attempt-*/model-process.json'))
    active = any(alive(read_json(p)) for p in owners)
    if not active and state.get('status') in {'queued', 'running'}:
        state['status'] = 'interrupted'
    return dict(state, active=active)


def validate_settings(settings):
    if set(settings) != set(DEFAULTS):
        raise ValueError('配置字段不完整或包含未知字段')
    if settings['mode'] not in {'cpu', 'gpu'}:
        raise ValueError('设备模式须为 cpu 或 gpu')
    for key, cap in [('cpu_limit', 1024), ('workers', 32), ('threads', 64), ('io_workers', 32), ('gpu_count', 32)]:
        value = settings[key]
        if value is not None and (type(value) is not int or not 1 <= value <= cap):
            raise ValueError(f'{key} 须为 1–{cap} 整数或 auto')
    for key in ('memory_gib', 'min_gpu_gib'):
        value = settings[key]
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{key} 须为正数')
    if settings['gpu_count'] is None or settings['min_gpu_gib'] is None:
        raise ValueError('GPU 并发和空闲显存阈值不能为 auto')
    if settings['mode'] == 'gpu' and not settings['devices']:
        raise ValueError('请在设备页明确选择分配给你的 GPU')
    if settings['mode'] == 'cpu' and settings['devices']:
        raise ValueError('CPU 模式不能携带 GPU 分配')


class Workbench:
    def __init__(self, workspace, overrides=None):
        self.root = Path(workspace).expanduser().resolve()
        if self.root.is_relative_to(CODE_ROOT) or (CODE_ROOT.is_relative_to(self.root) and not managed_release(self.root)):
            raise ValueError('运行目录必须与源码分开')
        self.children = []
        self.overrides = overrides or {}

    @property
    def settings(self):
        settings = {**DEFAULTS, 'results': str(self.root / 'results'),
                    **read_json(self.root / 'seg-settings.json', {}), **self.overrides}
        if not settings['preprocess_output']:
            settings['preprocess_output'] = DEFAULTS['preprocess_output']
        return settings

    def save(self, settings):
        validate_settings(settings)
        with file_lock(self.root / 'control.lock'):
            self.idle()
            write_json(self.root / 'seg-settings.json', settings)
            self.overrides = {}  # Later edits in the same session take effect.

    def entries(self):
        entries = []
        for job in (self.root / 'jobs').glob('*'):
            request = read_json(job / 'request.json')
            if isinstance(request, dict) and request.get('kind') == 'segmentation':
                entries.append({'job': str(job), **request, **status(job)})
        return sorted(entries, key=lambda row: row.get('created_at', ''), reverse=True)

    def resolve(self, job_id):
        if not re.fullmatch('[0-9a-f]{32}', str(job_id)):
            raise ValueError('任务编号无效；请选择本面板历史任务')
        job = self.root / 'jobs' / job_id
        request = read_json(job / 'request.json', {})
        if not isinstance(request, dict) or request.get('kind') != 'segmentation':
            raise ValueError('不是本面板的分割任务')
        return job

    def idle(self):
        self.children[:] = [child for child in self.children if child.poll() is None]
        for job in (self.root / 'jobs').glob('*'):
            if status(job).get('active'):
                raise ValueError('此运行目录还有后台任务或模型子进程，请先查询或停止')
        with file_lock(self.root / 'operation.lock'):
            pass

    def _paths(self, settings, cases):
        root = Path(settings['results']).expanduser().resolve()
        if root.is_relative_to(CODE_ROOT) or CODE_ROOT.is_relative_to(root):
            raise ValueError('结果目录必须与源码分开')
        if self.root.is_relative_to(root) or any(root.is_relative_to(self.root / p) for p in ('jobs', 'venv', 'weights', 'private')):
            raise ValueError('结果不能覆盖运行记录、环境或模型缓存')
        for row in cases:
            source = Path(row['image']).resolve()
            if root.is_relative_to(source.parent) or source.is_relative_to(root):
                raise ValueError('输入与结果目录必须分开')
            if row.get('preprocessing'):
                producer = Path(row['preprocessing']['root'])
                self._producer_runtime(producer)
                if root.is_relative_to(producer) or producer.is_relative_to(root):
                    raise ValueError('分割结果和运行目录不能写入或包含来源预处理任务')
        return root

    def _producer_runtime(self, producer):
        producer = Path(producer).expanduser().resolve()
        if self.root.is_relative_to(producer) or producer.is_relative_to(self.root):
            raise ValueError('分割运行目录必须与来源预处理任务分开')

    def create(self, name, steps, settings=None, input_job=None, case_ids=None, rerun_of=None, preprocessed=None):
        settings = {**DEFAULTS, **(settings or self.settings)}
        validate_settings(settings)
        if input_job and preprocessed:
            raise ValueError('历史分割输入和预处理输入不能同时选择')
        # Check before control/operation locks create or chmod the runtime.
        if preprocessed:
            self._producer_runtime(preprocessed)
        if not 1 <= len(name) <= 80 or any(ord(c) < 32 for c in name):
            raise ValueError('名称须为 1–80 字且不含控制字符')
        if not steps or set(steps) - {'plan', 'infer', 'merge'} or len(set(steps)) != len(steps):
            raise ValueError('步骤只能从 plan、infer、merge 中选择')
        steps = [s for s in ('plan', 'infer', 'merge') if s in steps]
        with file_lock(self.root / 'control.lock'):
            self.idle()
            parent = self.resolve(input_job) if input_job else None
            source = read_json(parent / 'request.json') if parent else None
            if preprocessed:
                from .preprocessed import load
                cases = load(preprocessed)
                settings['preprocess_output'] = str(Path(preprocessed).expanduser().resolve())
            else:
                cases = source['cases'] if parent else read_manifest(settings['manifest'])
            if case_ids:
                wanted = set(case_ids)
                if wanted - {r['case_id'] for r in cases}:
                    raise ValueError('所选病例不在输入任务中')
                cases = [r for r in cases if r['case_id'] in wanted]
            if not cases:
                raise ValueError('病例列表为空')
            cases = [dict(row, source_stat={'size': Path(row['image']).stat().st_size,
                                            'mtime_ns': Path(row['image']).stat().st_mtime_ns}) for row in cases]
            root = self._paths(settings, cases)
            inputs = {}
            if 'merge' in steps and 'infer' not in steps:
                if not source:
                    raise ValueError('仅合并需要选择历史输入任务')
                for case in cases:
                    key = case['case_id']
                    pointer = read_json(Path(source['output']) / 'cases' / key / 'latest-inference.json', {})
                    path = pointer.get('checkpoint') or source.get('inputs', {}).get(key)
                    if not path:
                        raise ValueError(f'{key} 缺少推理掩膜；请选择包含推理的步骤')
                    inputs[key] = path
            job_id = uuid.uuid4().hex
            job = private_dir(self.root / 'jobs' / job_id)
            code = private_dir(job / 'code')
            shutil.copytree(CODE_ROOT / 'aaoca_seg', code / 'aaoca_seg', ignore=shutil.ignore_patterns('__pycache__'))
            for file in ('launch_job.py', 'queue_job.py', 'run.py', 'requirements.txt', 'requirements-inference.txt'):
                shutil.copy2(CODE_ROOT / file, code / file)
            request = {'kind': 'segmentation', 'id': job_id, 'name': name, 'created_at': timestamp(),
                       'steps': steps, 'settings': settings, 'cases': cases, 'inputs': inputs,
                       'output': str(root / ('seg-' + job_id[:12])), 'input_job': input_job, 'rerun_of': rerun_of,
                       'preprocessing_source': (str(Path(preprocessed).expanduser().resolve()) if preprocessed
                                                else (source or {}).get('preprocessing_source')),
                       'source_request_sha256': digest(parent / 'request.json') if parent else None,
                       'code_hashes': {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()}}
            write_json(job / 'request.json', request)
            write_json(job / 'receipt.json', {'request_sha256': digest(job / 'request.json')})
            write_json(job / 'state.json', {'id': job_id, 'name': name, 'action': 'run', 'status': 'prepared'})
            write_json(self.root / 'seg-settings.json', settings)
            return job

    def start(self, job):
        job = self.resolve(Path(job).name)
        with file_lock(self.root / 'control.lock'):
            self.idle()
            request = read_json(job / 'request.json')
            if read_json(job / 'receipt.json', {}).get('request_sha256') != digest(job / 'request.json'):
                raise ValueError('冻结任务配置改变，不能续跑')
            for relative, expected in request['code_hashes'].items():
                source = (job / 'code' / relative).resolve()
                if not source.is_relative_to((job / 'code').resolve()) or digest(source) != expected:
                    raise ValueError('代码快照改变，不能续跑')
            if not python_path(self.root).is_file():
                raise ValueError('请先通过 e 安装独立推理环境')
            # Keep every owner attempt; the scientific request and output stay frozen.
            old = read_json(job / 'state.json', {})
            write_json(job / 'runs' / (uuid.uuid4().hex + '.json'), old)
            (job / 'stop').unlink(missing_ok=True)
            write_json(job / 'state.json', {**old, 'status': 'queued', 'started_at': timestamp()})
            write_json(self.root / 'current.json', {'id': job.name})
            try:
                child = detached([sys.executable, '-u', str(job / 'code' / 'launch_job.py'), str(self.root), job.name],
                                 job / 'worker.log', runtime_env(self.root))
                self.children.append(child)
                write_json(job / 'owner.json', process_identity(child.pid))
            except Exception:
                write_json(job / 'state.json', {**old, 'status': 'failed', 'error': '后台进程未启动'})
                raise
        return job

    def rerun(self, job_id, steps=None, name=None):
        parent = self.resolve(job_id)
        request = read_json(parent / 'request.json')
        if status(parent).get('status') != 'completed':
            raise ValueError('k 用于已完成任务；未完成任务请用 r 续跑')
        return self.create(name or (request['name'][:70] + ' / 重跑'), steps or request['steps'],
                           request['settings'], job_id, rerun_of=job_id)

    def stop(self, job_id):
        with file_lock(self.root / 'control.lock'):
            job = self.resolve(job_id)
            if not status(job)['active']:
                raise ValueError('选中任务没有存活进程')
            (job / 'stop').touch()
            request = read_json(job / 'request.json')
            for record in Path(request['output']).glob('cases/*/attempt-*/model-process.json'):
                if not alive(read_json(record.parent / 'executor-process.json')):
                    terminate_owned(read_json(record))
        return '已请求停止；完整推理掩膜与结果保留，停止状态以进程退出为准。'

    def logs(self, job):
        request = read_json(Path(job) / 'request.json', {})
        output = Path(request['output'])
        text = tail(Path(job) / 'worker.log') + tail(Path(job) / 'queue.log')
        progress = read_json(output / 'status.json', {})
        for key, item in progress.get('active', {}).items():
            attempt = Path(item['attempt'])
            text += f'\n[{key}] {item["phase"]}\n' + tail(attempt / (item['phase'] + '-worker.log'), 6000)
            phase = read_json(attempt / 'phase.json', {}).get('phase')
            if phase in {'total', 'heartchambers_highres', 'coronary_arteries'}:
                text += tail(attempt / (phase + '.log'), 6000)
        for row in progress.get('cases', [])[-20:]:
            if row.get('status') != 'failed':
                continue
            text += f"\n[{row['case_id']}] 失败：{row.get('error', row.get('phase', '未知'))}\n"
            if row.get('log'):
                failed_log = Path(row['log']).resolve()
                if failed_log.is_relative_to(output.resolve()):
                    text += tail(failed_log, 8000)
                    phase = read_json(failed_log.parent / 'phase.json', {}).get('phase')
                    if phase in {'total', 'heartchambers_highres', 'coronary_arteries'}:
                        text += tail(failed_log.parent / (phase + '.log'), 8000)
        return text

    def setup(self, action, payload=None):
        # The legacy controller uses this same current pointer and control lock.
        self.idle()
        return ManagedController(self.root).submit(action, payload)
