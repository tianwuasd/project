"""Bounded CPU batches with case-level atomic publication and verified reuse."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

import psutil

from .config import Config
from .image import inspect_image, preprocess_image
from .io import RunLock, discover, file_hash, now, read_json, write_json


def plan(input_root, config=Config(), manifest=None):
    rows = []
    for case in discover(input_root, manifest):
        row = {k: v for k, v in case.items() if k != 'source'}
        try:
            row.update(inspect_image(case['source'], config))
            row['status'] = 'ready'
        except Exception as exc:
            row.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return {'state': 'planned', 'total': len(rows), 'invalid': sum(r['status'] != 'ready' for r in rows),
            'output_bytes_uncompressed': sum(r.get('output_bytes_uncompressed', 0) for r in rows),
            'max_estimated_working_bytes': max(r.get('estimated_working_bytes', 0) for r in rows),
            'note': 'headers only; HU calibration, pixel integrity and hashes are verified at run time',
            'config': config.recipe(), 'cases': rows}


def available_memory():
    """Respect a visible Linux cgroup limit when one is reported."""
    available = int(psutil.virtual_memory().available)
    for limit_path, used_path in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                                   '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            limit, used = int(Path(limit_path).read_text()), int(Path(used_path).read_text())
            available = min(available, max(0, limit - used))
        except (OSError, ValueError):
            pass
    return available


def disjoint(a, b):
    return not (a == b or a.is_relative_to(b) or b.is_relative_to(a))


def process_case(case, output, scratch_root, config, signature, info):
    started = time.monotonic()
    source = case['source']
    initial_stat = source.stat()
    digest = file_hash(source)
    if case['expected_sha256'] and digest != case['expected_sha256']:
        raise ValueError('source SHA256 differs from the supplied original-CT manifest')
    destination = output / 'cases' / case['case_id']
    image_path = destination / 'image.nii.gz'
    if destination.exists():
        record = read_json(destination / 'record.json')
        if (record.get('status') != 'complete' or record.get('input') != case['input']
                or record.get('input_sha256') != digest or record.get('config_sha256') != signature):
            raise ValueError('existing case has different source/configuration; choose a new output directory')
        if file_hash(image_path) != record['output_sha256']:
            raise ValueError('existing output failed SHA256 validation; preserved for inspection')
        return {'case_id': case['case_id'], 'input': case['input'], 'state': 'reused',
                'output': record['output'], 'seconds': time.monotonic() - started}
    if info['estimated_working_bytes'] > config.max_case_gib * 2**30:
        raise ValueError('estimated case memory exceeds --max-case-gib; review the plan and resource allocation')
    # Private temporary directories; only these generated temporary paths are cleaned up.
    with tempfile.TemporaryDirectory(prefix='case-', dir=output / '.staging') as stage_name:
        stage = Path(stage_name)
        with tempfile.TemporaryDirectory(prefix='aaoca-', dir=scratch_root) as scratch_name:
            checks = preprocess_image(source, stage / 'image.nii.gz', Path(scratch_name), config, info)
        final_stat = source.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns) or file_hash(source) != digest:
            raise ValueError('source changed while processing; no output was published')
        relative_output = (Path('cases') / case['case_id'] / 'image.nii.gz').as_posix()
        record = {'schema': 1, 'status': 'complete', 'case_id': case['case_id'],
                  'input': case['input'], 'output': relative_output,
                  'input_sha256': digest, 'output_sha256': file_hash(stage / 'image.nii.gz'),
                  'config_sha256': signature, 'config': config.recipe(), **info, **checks,
                  'completed_at': now(), 'seconds': time.monotonic() - started}
        write_json(stage / 'record.json', record)
        # Destination never exists here; the directory rename publishes image and record together.
        stage.rename(destination)
    return {'case_id': case['case_id'], 'input': case['input'], 'state': 'completed',
            'output': relative_output, 'seconds': time.monotonic() - started}


def run(input_root, output_root, config=Config(), manifest=None, scratch_root=None, progress=None):
    input_root, output = Path(input_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve() if scratch_root else output / '.scratch'
    if not disjoint(input_root, output) or not disjoint(input_root, scratch):
        raise ValueError('input tree must be separate from output and scratch trees')
    cases = discover(input_root, manifest)
    signature = config.signature()
    output.mkdir(parents=True, exist_ok=True)
    with RunLock(output):
        owner = output / 'dataset.json'
        if owner.exists():
            if read_json(owner).get('config_sha256') != signature:
                raise ValueError('output configuration/implementation differs; choose a new output directory')
        else:
            if any(p.name != '.preprocess.lock' for p in output.iterdir()):
                raise ValueError('output contains unrelated files; choose an empty output directory')
            write_json(owner, {'schema': 1, 'created_at': now(), 'config_sha256': signature, 'config': config.recipe()})
        for folder in [output / 'cases', output / '.staging', scratch]:
            folder.mkdir(parents=True, exist_ok=True)
        state = {'state': 'checking', 'started_at': now(), 'updated_at': now(),
                 'total': len(cases), 'completed': 0, 'reused': 0, 'failed': 0, 'pending': len(cases),
                 'resources': asdict(config), 'python': platform.python_version(),
                 'platform': platform.system(), 'results': [], 'errors': []}

        def save():
            state['updated_at'] = now()
            write_json(output / 'progress.json', state)

        def accept(result):
            if result['state'] == 'failed':
                state['failed'] += 1
                state['errors'].append(result)
            else:
                state[result['state']] += 1
                state['results'].append(result)
            state['pending'] -= 1
            save()
            if progress:
                progress(result, state)

        save()
        try:
            ready = []
            for case in cases:
                try:
                    info = inspect_image(case['source'], config)
                    ready.append((case, info))
                except Exception as exc:
                    accept({'case_id': case['case_id'], 'input': case['input'], 'state': 'failed',
                            'error': f'{type(exc).__name__}: {exc}'})
            # Existing cases can be verified even if there is no room for new image generation.
            fresh = [info for case, info in ready if not (output / 'cases' / case['case_id']).exists()
                     and info['estimated_working_bytes'] <= config.max_case_gib * 2**30]
            maximum = max((info['estimated_working_bytes'] for info in fresh), default=0)
            memory = available_memory()
            affinity = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else (os.cpu_count() or 1)
            if config.workers > affinity:
                raise ValueError('workers exceed the current process CPU affinity allocation')
            if maximum * config.workers > memory * .75:
                raise ValueError('estimated concurrent working memory exceeds 75% of currently available memory; reduce workers')
            payload = sum(info['output_bytes_uncompressed'] for info in fresh)
            scratch_bytes = sum(sorted((info['output_bytes_uncompressed'] for info in fresh), reverse=True)[:config.workers])
            # Conservatively reserve scratch on both volumes; quotas/NFS limits still need local checks.
            required = int(payload * 1.05) + scratch_bytes + (64 * 2**20 if fresh else 0)
            if shutil.disk_usage(output).free < required or shutil.disk_usage(scratch).free < scratch_bytes:
                raise ValueError('insufficient free disk for uncompressed scratch and conservatively estimated outputs')
            state['state'] = 'running'
            state['resources'].update(available_memory_bytes=memory, cpu_affinity_count=affinity,
                                      estimated_output_reserve_bytes=required)
            save()

            def execute(item):
                case, info = item
                try:
                    return process_case(case, output, scratch, config, signature, info)
                except Exception as exc:
                    return {'case_id': case['case_id'], 'input': case['input'], 'state': 'failed',
                            'error': f'{type(exc).__name__}: {exc}'}

            pool = ThreadPoolExecutor(max_workers=config.workers)
            futures = []
            try:
                futures = [pool.submit(execute, item) for item in ready]
                for future in as_completed(futures):
                    accept(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            state['state'] = 'completed' if not state['failed'] else 'completed_with_errors'
            save()
            write_json(output / 'manifest.json', state['results'])
            write_json(output / 'failures.json', state['errors'])
            return state
        except BaseException as exc:
            state.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         error=f'{type(exc).__name__}: {exc}')
            save()
            raise
