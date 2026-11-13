import gzip
from pathlib import Path
import time

from .images import inspect_image
from .pipeline import sha256
from .service_state import read_json, timestamp, write_json


def source_stat(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def validate_prepared(case, receipt):
    receipt = Path(receipt).resolve()
    record = read_json(receipt)
    pointer = read_json(receipt.parent.parent / 'latest-prepared.json', {})
    if pointer.get('checkpoint') != str(receipt) or pointer.get('sha256') != sha256(receipt):
        raise ValueError('CPU 准备记录完整性校验失败')
    source = Path(case['image']).resolve()
    if not isinstance(record, dict) or record.get('status') != 'prepared':
        raise ValueError('缺少完整的 CPU 准备记录')
    if record['case_id'] != case['case_id'] or record['source'] != str(source):
        raise ValueError('CPU 准备记录来源不符')
    if source_stat(source) != record['source_stat'] or sha256(source) != record['source_sha256']:
        raise ValueError('准备后的原始输入发生变化')
    if case.get('source_sha256') and case['source_sha256'] != record['source_sha256']:
        raise ValueError('冻结的原始输入哈希改变')
    image = Path(record['image']).resolve()
    expected = receipt.parent / 'input.nii' if str(source).lower().endswith('.gz') else source
    if image != expected or image.is_symlink() or sha256(image) != record['image_sha256']:
        raise ValueError('CPU 准备文件路径或哈希不符')
    return record


def prepare(case, attempt, stop):
    if case.get('preprocessing'):
        raise ValueError('CPU 提前准备仅接收原始 HU/mm CT')
    attempt = Path(attempt).resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    receipt = attempt / 'prepared.json'
    if receipt.exists():
        return validate_prepared(case, receipt)
    started = time.monotonic()
    source = Path(case['image']).resolve()
    before = source_stat(source)
    if case.get('source_stat') != before:
        raise ValueError('提交后的原始输入发生变化')
    source_hash = sha256(source)
    if case.get('source_sha256') and case['source_sha256'] != source_hash:
        raise ValueError('冻结的原始输入哈希改变')
    frozen = attempt.parent / 'input-identity.json'
    identity = {'source': str(source), 'sha256': source_hash}
    if frozen.exists() and read_json(frozen) != identity:
        raise ValueError('冻结输入已改变，请新建任务')
    write_json(frozen, identity)
    partial = attempt / 'input.partial.nii'
    if partial.is_symlink():
        raise ValueError('准备临时文件不能是符号链接')
    partial.unlink(missing_ok=True)
    image = source
    committed = False
    created_image = False
    try:
        if str(source).lower().endswith('.gz'):
            with gzip.open(source, 'rb') as compressed, partial.open('xb') as output:
                while chunk := compressed.read(8 * 1024 * 1024):
                    if Path(stop).exists():
                        raise KeyboardInterrupt
                    output.write(chunk)
            image = partial
        geometry = inspect_image(image, full=True)
        if Path(stop).exists():
            raise KeyboardInterrupt
        if source_stat(source) != before or sha256(source) != source_hash:
            raise ValueError('CPU 准备期间原始输入发生变化')
        image_hash = sha256(image)
        if image == partial:
            image = attempt / 'input.nii'
            if image.exists():
                raise ValueError('准备目标已存在，保留供核查')
            partial.replace(image)
            created_image = True
        record = {'status': 'prepared', 'case_id': case['case_id'], 'source': str(source),
                  'source_stat': before, 'source_sha256': source_hash, 'image': str(image),
                  'image_sha256': image_hash, 'geometry': geometry, 'seconds': time.monotonic() - started,
                  'ended_at': timestamp()}
        write_json(receipt, record)
        write_json(attempt.parent / 'latest-prepared.json', {'checkpoint': str(receipt), 'sha256': sha256(receipt)})
        committed = True
        return record
    finally:
        partial.unlink(missing_ok=True)
        if not committed:
            if created_image:
                image.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
