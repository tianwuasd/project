"""Verify a known preprocessing cache without changing its scientific records."""
from contextlib import ExitStack
import json
from pathlib import Path
import time

from aaoca_preprocessing.io import RunLock, discover, file_hash
from aaoca_preprocessing.config import Config
from .state import job_status, read


class CacheMismatch(ValueError):
    """Only a cache/source-contract mismatch permits a new output fallback."""


def document(path, kind=dict):
    try:
        value = read(path)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CacheMismatch(f'缓存记录损坏：{path}') from exc
    if not isinstance(value, kind):
        raise CacheMismatch(f'缓存记录缺失或格式不符：{path}')
    return value


def inspect_cache(output, settings, config, emit=print):
    """Return reuse/resume; propagate raw-input, permission and lock errors."""
    source = Path(settings['input']).expanduser().resolve()
    cases = discover(source, settings.get('manifest') or None)
    signature = config.signature()
    output = Path(output).expanduser().resolve()
    runtime = output.with_name(output.name + '.runtime')
    try:
        active = job_status(output)['alive']
    except (ValueError, TypeError) as exc:
        raise RuntimeError('缓存进程记录损坏，无法确认是否仍在写入') from exc
    if active is not False:
        raise RuntimeError('缓存任务仍在运行或无法确认存活状态；未创建新任务')
    # Operational failures take precedence over every cache mismatch. In
    # particular a changed recipe must not mask a writer or invalid raw input.
    with ExitStack() as locks:
        for directory in (runtime, output):
            if directory.exists():
                if not directory.is_dir() or (directory / '.preprocess.lock').is_symlink():
                    raise RuntimeError('缓存或运行锁路径不符；未创建新任务')
                locks.enter_context(RunLock(directory))
        start = time.monotonic()
        verified_sources = []
        for index, case in enumerate(cases, 1):
            emit(f'原始输入核验 [{index}/{len(cases)}] {case["case_id"]}；已用 {time.monotonic()-start:.1f} 秒')
            stat = case['source'].stat()
            if case['expected_stat'] is not None and [stat.st_size, stat.st_mtime_ns] != case['expected_stat']:
                raise ValueError('原始输入文件状态不符合本次冻结清单，停止；未尝试新目录')
            digest = file_hash(case['source'])
            if case['expected_sha256'] and digest != case['expected_sha256']:
                raise ValueError('原始输入不符合本次清单的 SHA256，停止；未尝试新目录')
            _source_unchanged(case, stat)
            verified_sources.append((stat, digest))
        return _inspect_contract(output, runtime, source, cases, verified_sources, signature, emit, start)


def _source_unchanged(case, stat):
    final_stat = case['source'].stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError('核验期间原始输入发生变化，停止；未创建新任务')


def _inspect_contract(output, runtime, source, cases, verified_sources, signature, emit, start):
    """Caller owns both cache locks and has validated the current input contract."""
    spec = document(runtime / 'job.json')
    if spec.get('input') != str(source) or spec.get('output') != str(output):
        raise CacheMismatch('缓存来源目录或输出目录不一致')
    if spec.get('config_sha256') != signature:
        raise CacheMismatch('缓存参数、实现或依赖与本次计划不同')
    try:
        frozen_config = Config(**spec['config'])
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheMismatch('冻结参数记录损坏') from exc
    if frozen_config.signature() != signature:
        raise CacheMismatch('冻结参数与记录摘要不一致')
    inputs = spec.get('inputs')
    if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
        raise CacheMismatch('冻结输入清单格式损坏')
    if document(runtime / 'inputs.json', list) != inputs:
        raise CacheMismatch('冻结输入清单与任务记录不一致')
    if [c['input'] for c in cases] != [c.get('input') for c in inputs]:
        raise CacheMismatch('输入清单范围发生变化')
    if document(output / 'dataset.json').get('config_sha256') != signature:
        raise CacheMismatch('缓存数据集参数或实现不匹配')
    completed = 0
    if not (output / 'cases').is_dir() or (output / 'cases').is_symlink():
        raise CacheMismatch('缓存病例目录缺失或路径不符')
    for index, (case, frozen, (stat, digest)) in enumerate(zip(cases, inputs, verified_sources), 1):
        emit(f'缓存完整核验 [{index}/{len(cases)}] {case["case_id"]}；已用 {time.monotonic()-start:.1f} 秒')
        _source_unchanged(case, stat)
        if (stat.st_size, stat.st_mtime_ns) != (frozen.get('size'), frozen.get('mtime_ns')):
            raise CacheMismatch('缓存来源文件状态变化')
        if frozen.get('sha256') and digest != frozen['sha256']:
            raise CacheMismatch('缓存来源文件 SHA256 变化')
        directory = output / 'cases' / case['case_id']
        if directory.exists():
            if directory.is_symlink():
                raise CacheMismatch('缓存病例目录为符号链接')
            record = document(directory / 'record.json')
            if (record.get('status') != 'complete' or record.get('input') != case['input']
                    or record.get('input_sha256') != digest or record.get('config_sha256') != signature):
                raise CacheMismatch('缓存病例来源或方法不一致')
            for name, key, label in [('image.nii.gz', 'output_sha256', '归一化影像'),
                                      ('image_hu.nii.gz', 'hu_output_sha256', 'HU 配套影像')]:
                path = directory / name
                if not path.is_file() or path.is_symlink():
                    raise CacheMismatch(f'缓存{label}缺失或路径不符')
                if file_hash(path) != record.get(key):
                    raise CacheMismatch(f'缓存{label} SHA256 核验失败')
            completed += 1
        _source_unchanged(case, stat)
    expected = {case['case_id'] for case in cases}
    if any(path.name not in expected for path in (output / 'cases').iterdir()):
        raise CacheMismatch('缓存包含本次清单之外的病例')
    progress = document(output / 'progress.json')
    action = 'reuse' if completed == len(cases) and progress.get('state') == 'completed' else 'resume'
    return {'action': action, 'output': str(output), 'verified': completed, 'total': len(cases),
            'verification': 'full_source_and_paired_output_sha256', 'config_sha256': signature}
