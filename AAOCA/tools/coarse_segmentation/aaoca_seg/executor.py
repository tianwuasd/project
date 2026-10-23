"""Isolated case processes; GPU checkpoints precede CPU-only merge/QC."""
import gzip
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .images import combine, inspect_image
from .pipeline import commands, sha256, stop_child
from .preprocessed import validate_case
from .schema import LABELS
from .service_state import process_identity, read_json, timestamp, write_json


def lock_fds():
    return tuple(int(v) for v in os.environ.get('AAOCA_LOCK_FDS', '').split(',') if v) if os.name != 'nt' else ()


def checkpoint(path, case):
    validate_case(case, full=True)
    record = read_json(path)
    if not isinstance(record, dict) or record.get('status') != 'inferred':
        raise ValueError('缺少完整推理检查点')
    if record.get('case_id') != case['case_id'] or record.get('source') != str(Path(case['image']).resolve()):
        raise ValueError('推理检查点输入来源不符')
    if record.get('source_sha256') != sha256(case['image']):
        raise ValueError('原始影像已改变，不能复用旧掩膜')
    if record.get('preprocessing') != case.get('preprocessing'):
        raise ValueError('推理检查点的预处理来源不符')
    root = Path(record['mask_root']).resolve()
    expected = {str(Path(task) / f'{name}.nii.gz') for _, name, _, task in LABELS}
    if set(record.get('artifacts', {})) != expected:
        raise ValueError('推理检查点缺少必要掩膜')
    for relative, digest in record['artifacts'].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or sha256(target) != digest:
            raise ValueError('推理检查点掩膜校验失败')
    return record


def execute_model(command, log, stop, registry):
    with Path(log).open('ab') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=os.name != 'nt',
                                 **({'pass_fds': lock_fds()} if os.name != 'nt' else {}))
        write_json(registry, process_identity(child.pid))
        try:
            while child.poll() is None:
                if Path(stop).exists():
                    raise KeyboardInterrupt
                time.sleep(.2)
            if child.returncode:
                raise ValueError(f'模型退出 {child.returncode}，详见 {log}')
        finally:
            stop_child(child)


def infer(case, attempt, threads, device, stop, identity):
    validate_case(case, full=True)
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(case['image'])
    if case.get('source_sha256') and case['source_sha256'] != source_hash:
        raise ValueError('排队期间原始输入改变，不能推理')
    frozen = attempt.parent / 'input-identity.json'
    input_identity = {'source': str(Path(case['image']).resolve()), 'sha256': source_hash}
    if frozen.exists() and read_json(frozen) != input_identity:
        raise ValueError('冻结输入已改变，请新建任务')
    write_json(frozen, input_identity)
    temporary = attempt / 'input.nii'
    image = Path(case['image'])
    try:
        if str(image).lower().endswith('.gz'):
            # One lossless decompression for all three models, CRC checked by gzip.
            with gzip.open(image, 'rb') as source, temporary.open('wb') as target:
                while chunk := source.read(8 * 1024 * 1024):
                    if Path(stop).exists():
                        raise KeyboardInterrupt
                    target.write(chunk)
            image = temporary
        geometry = inspect_image(image, full=True)
        started = time.monotonic()
        for task, command in commands(image, attempt, device, threads):
            if Path(stop).exists():
                raise KeyboardInterrupt
            write_json(attempt / 'phase.json', {'phase': task, 'time': timestamp()})
            write_json(attempt / f'{task}.command.json', command)
            execute_model(command, attempt / f'{task}.log', stop, attempt / 'model-process.json')
        if sha256(case['image']) != source_hash:
            raise ValueError('推理期间输入改变')
        validate_case(case, full=True)
        artifacts = {str(Path(task) / f'{name}.nii.gz'): sha256(attempt / task / f'{name}.nii.gz')
                     for _, name, _, task in LABELS}
        record = {'status': 'inferred', 'case_id': case['case_id'], 'source': str(Path(case['image']).resolve()),
                  'source_sha256': source_hash, 'geometry': geometry, 'mask_root': str(attempt.resolve()),
                  'artifacts': artifacts, 'identity': identity, 'seconds': time.monotonic() - started,
                  'ended_at': timestamp()}
        if case.get('preprocessing'):
            record['preprocessing'] = case['preprocessing']
        write_json(attempt / 'inference.json', record)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def merge(case, source, destination):
    record = checkpoint(source, case)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files, report = combine(case['image'], destination, record['mask_root'])
    if sha256(case['image']) != record['source_sha256']:
        raise ValueError('质检期间输入改变')
    validate_case(case, full=True)
    result = {'status': 'completed', 'case_id': case['case_id'], 'inference': str(Path(source).resolve()),
              'source_sha256': record['source_sha256'], 'review_required': True,
              'empty_masks': report['empty_masks'], 'ended_at': timestamp(),
              'artifacts': {str(p.resolve()): sha256(p) for p in files}}
    if case.get('preprocessing'):
        result['preprocessing'] = case['preprocessing']
        result['label_reference_image'] = case['preprocessing']['normalized_image']
    write_json(destination / 'result.json', result)
    return result


def completed(path, case, source):
    checkpoint(source, case)
    result = read_json(path, {})
    if result.get('status') != 'completed' or result.get('case_id') != case['case_id']:
        return False
    if result.get('inference') != str(Path(source).resolve()):
        return False
    artifacts = result.get('artifacts', {})
    required = {str((Path(path).parent / n).resolve()) for n in ('aaoca_11_labels.nii.gz', 'qc.json', 'labels.ctbl')}
    if not required.issubset(artifacts):
        return False
    return all(Path(p).is_file() and sha256(p) == digest for p, digest in artifacts.items())


def main():
    payload = read_json(sys.argv[1])
    write_json(Path(payload['attempt']) / 'executor-process.json', process_identity())
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if payload['phase'] == 'infer':
            infer(payload['case'], payload['attempt'], payload['threads'], payload['device'],
                  payload['stop'], payload['identity'])
        elif payload['phase'] == 'verify':
            record = checkpoint(payload['source'], payload['case'])
            # Own frozen inference must match its original environment/code contract.
            if payload.get('expected_identity') and record.get('identity') != payload['expected_identity']:
                raise ValueError('推理环境或方法记录改变，请新建任务')
            done = completed(Path(payload['attempt']) / 'result.json', payload['case'], payload['source'])
            write_json(payload['verification'], {'valid': True, 'completed': done})
        else:
            merge(payload['case'], payload['source'], payload['attempt'])
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(str(error), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
