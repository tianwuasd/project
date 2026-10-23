"""Bounded CPU batches with case-level atomic publication and verified reuse."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
import platform
import shutil
import tempfile
import time
import threading

import numpy as np

from .config import Config
from .image import inspect_image, preprocess_image
from .io import RunLock, discover, file_hash, now, read_json, write_json
from .control import Control, StopRequested
from .resources import allocation, RuntimeSampler


def suggest_spacing(input_root, manifest=None):
    """nnU-Net v2 median/anisotropy heuristic on native full-FOV RAS headers.

    This is deliberately not nnU-Net experiment planning: no crop, labels,
    foreground intensity statistics, patch size or network are inferred.
    """
    cases = discover(input_root, manifest)
    info = [inspect_image(case['source'], Config()) for case in cases]
    spacings = np.array([row['canonical_spacing_mm'] for row in info])
    shapes = np.array([row['canonical_shape'] for row in info])
    target = np.percentile(spacings, 50, axis=0)
    median_shape = np.percentile(shapes, 50, axis=0)
    worst = int(np.argmax(target))
    others = [axis for axis in range(3) if axis != worst]
    adjusted = bool(target[worst] > 3 * max(target[others]) and
                    median_shape[worst] * 3 < min(median_shape[others]))
    if adjusted:
        candidate = float(np.percentile(spacings[:, worst], 10))
        target[worst] = max(candidate, max(target[others]) + 1e-5) if candidate < max(target[others]) else candidate
    return {'method': 'nnUNet-v2-inspired-median-anisotropy', 'source_count': len(cases),
            'spacing_mm': target.tolist(), 'axis_order': 'RAS',
            'spacing_percentiles_mm': {str(p): np.percentile(spacings, p, axis=0).tolist() for p in [10, 50, 90]},
            'shape_basis': 'full_FOV_not_cropped', 'anisotropy_adjusted': adjusted,
            'anisotropy_threshold': 3, 'median_shape': median_shape.tolist(),
            'note': 'header-based geometry suggestion, not full nnU-Net planning or validated optimal resolution; fit on training data only'}


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
    return allocation(Config())['available_memory_bytes']


def disjoint(a, b):
    return not (a == b or a.is_relative_to(b) or b.is_relative_to(a))


def process_case(case, output, scratch_root, config, signature, info, control=None):
    control = control or Control()
    started = time.monotonic()
    source = case['source']
    initial_stat = source.stat()
    if case.get('expected_stat') is not None and [initial_stat.st_size, initial_stat.st_mtime_ns] != case['expected_stat']:
        raise ValueError('source changed since frozen job creation')
    control.stage(case['case_id'], 'input_hash')
    with control.io():
        digest = file_hash(source, control.check)
    hash_seconds = time.monotonic() - started
    if case['expected_sha256'] and digest != case['expected_sha256']:
        raise ValueError('source SHA256 differs from the supplied original-CT manifest')
    destination = output / 'cases' / case['case_id']
    image_path = destination / 'image.nii.gz'
    if destination.exists():
        record = read_json(destination / 'record.json')
        if (record.get('status') != 'complete' or record.get('input') != case['input']
                or record.get('input_sha256') != digest or record.get('config_sha256') != signature):
            raise ValueError('existing case has different source/configuration; choose a new output directory')
        with control.io():
            if file_hash(image_path, control.check) != record['output_sha256']:
                raise ValueError('existing output failed SHA256 validation; preserved for inspection')
        return {'case_id': case['case_id'], 'input': case['input'], 'state': 'reused',
                'output': record['output'], 'seconds': time.monotonic() - started}
    if info['estimated_working_bytes'] > config.max_case_gib * 2**30:
        raise ValueError('estimated case memory exceeds --max-case-gib; review the plan and resource allocation')
    # Private temporary directories; only these generated temporary paths are cleaned up.
    with tempfile.TemporaryDirectory(prefix='case-', dir=output / '.staging') as stage_name:
        stage = Path(stage_name)
        with tempfile.TemporaryDirectory(prefix='aaoca-', dir=scratch_root) as scratch_name:
            checks = preprocess_image(source, stage / 'image.nii.gz', Path(scratch_name), config, info, control, case['case_id'])
        control.stage(case['case_id'], 'source_recheck')
        check_started = time.monotonic()
        final_stat = source.stat()
        with control.io():
            if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns) or file_hash(source, control.check) != digest:
                raise ValueError('source changed while processing; no output was published')
            output_hash = file_hash(stage / 'image.nii.gz', control.check)
        checks['timings_seconds'].update(input_hash=hash_seconds, source_and_output_hash=time.monotonic()-check_started)
        relative_output = (Path('cases') / case['case_id'] / 'image.nii.gz').as_posix()
        record = {'schema': 1, 'status': 'complete', 'case_id': case['case_id'],
                  'input': case['input'], 'output': relative_output,
                  'input_sha256': digest, 'output_sha256': output_hash,
                  'config_sha256': signature, 'config': config.recipe(), **info, **checks,
                  'completed_at': now(), 'seconds': time.monotonic() - started}
        write_json(stage / 'record.json', record)
        # Destination never exists here; the directory rename publishes image and record together.
        stage.rename(destination)
    return {'case_id': case['case_id'], 'input': case['input'], 'state': 'completed',
            'output': relative_output, 'seconds': time.monotonic() - started}


def run(input_root, output_root, config=Config(), manifest=None, scratch_root=None, progress=None, stop=None, event=None, spacing_report=None):
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
            write_json(owner, {'schema': 1, 'created_at': now(), 'config_sha256': signature,
                               'config': config.recipe(), 'spacing_selection': spacing_report})
        for folder in [output / 'cases', output / '.staging', scratch]:
            folder.mkdir(parents=True, exist_ok=True)
        resources = allocation(config)
        if config.compute_threads > max(1, int(resources['cpu_budget'])):
            raise ValueError('compute threads exceed the visible CPU allocation')
        cancelled = threading.Event()
        should_stop = lambda: cancelled.is_set() or bool(stop and stop())
        state = {'state': 'checking', 'started_at': now(), 'updated_at': now(),
                 'total': len(cases), 'completed': 0, 'reused': 0, 'failed': 0, 'pending': len(cases),
                 'resources': {**asdict(config), **resources}, 'python': platform.python_version(),
                 'platform': platform.system(), 'results': [], 'errors': [], 'active': {}}
        mutex = threading.RLock()
        last_save = [0.0]

        def save():
            with mutex:
                state['updated_at'] = now()
                write_json(output / 'progress.json', state)
                last_save[0] = time.monotonic()

        def stage(case_id, name, **fields):
            with mutex:
                previous = state['active'].get(case_id, {}).get('stage')
                state['active'][case_id] = {'stage': name, 'observed_at': now(), **fields}
                if previous != name or time.monotonic() - last_save[0] >= 2:
                    save()
            if event:
                event(case_id, name, **fields)
        control = Control(resources['read_slots'], should_stop, stage)

        def accept(result):
            with mutex:
                state['active'].pop(result['case_id'], None)
            if result['state'] == 'stopped':
                save()
                return
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
                     and info['estimated_working_bytes'] <= min(config.max_case_gib * 2**30, resources['memory_budget_bytes'])]
            payload = sum(info['output_bytes_uncompressed'] for info in fresh)
            scratch_bytes = sum(sorted((info['scratch_bytes'] for info in fresh), reverse=True)[:resources['case_slots']])
            # Conservatively reserve scratch on both volumes; quotas/NFS limits still need local checks.
            required = int(payload * 1.05) + scratch_bytes + (64 * 2**20 if fresh else 0)
            if shutil.disk_usage(output).free < required or shutil.disk_usage(scratch).free < scratch_bytes:
                raise ValueError('insufficient free disk for uncompressed scratch and conservatively estimated outputs')
            state['state'] = 'running'
            state['resources'].update(estimated_output_reserve_bytes=required)
            save()

            def execute(item):
                case, info = item
                try:
                    return process_case(case, output, scratch, config, signature, info, control)
                except StopRequested:
                    return {'case_id': case['case_id'], 'input': case['input'], 'state': 'stopped'}
                except Exception as exc:
                    return {'case_id': case['case_id'], 'input': case['input'], 'state': 'failed',
                            'error': f'{type(exc).__name__}: {exc}'}

            queue, futures, reserved = list(ready), {}, 0
            with RuntimeSampler() as measured, ThreadPoolExecutor(max_workers=resources['case_slots']) as pool:
                try:
                    while queue or futures:
                        if not should_stop():
                            index = 0
                            while index < len(queue) and len(futures) < resources['case_slots']:
                                case, info = queue[index]
                                reuse = (output / 'cases' / case['case_id']).exists()
                                needed = 0 if reuse else info['estimated_working_bytes']
                                if needed > min(config.max_case_gib * 2**30, resources['memory_budget_bytes']):
                                    queue.pop(index)
                                    accept({'case_id': case['case_id'], 'input': case['input'], 'state': 'failed',
                                            'error': 'estimated case memory exceeds --max-case-gib or allocated memory budget'})
                                    continue
                                if reserved + needed > resources['memory_budget_bytes']:
                                    index += 1
                                    continue
                                queue.pop(index)
                                future = pool.submit(execute, (case, info))
                                futures[future] = needed
                                reserved += needed
                                state['resources']['max_active_cases'] = max(state['resources']['max_active_cases'], len(futures))
                        if not futures:
                            break
                        done, _ = wait(futures, timeout=.2, return_when=FIRST_COMPLETED)
                        for future in done:
                            reserved -= futures.pop(future)
                            with mutex:
                                accept(future.result())
                except BaseException:
                    cancelled.set()
                    raise
            state['performance'] = measured.result
            state['state'] = 'stopped' if should_stop() else ('completed' if not state['failed'] else 'completed_with_errors')
            save()
            write_json(output / 'manifest.json', state['results'])
            write_json(output / 'failures.json', state['errors'])
            return state
        except BaseException as exc:
            state.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         error=f'{type(exc).__name__}: {exc}')
            save()
            raise
