"""Frozen, detached CPU jobs. Stop is cooperative; never signal a guessed PID."""
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

import psutil

from .config import Config
from .io import RunLock, discover, now, read_json, write_json
from .pipeline import disjoint, run


def runtime_path(output):
    output = Path(output).expanduser().resolve()
    return output.with_name(output.name + '.runtime')


def alive(identity):
    try:
        process = psutil.Process(identity['pid'])
        return abs(process.create_time() - identity['created']) < .01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (KeyError, psutil.Error, TypeError):
        return False


def status(output):
    runtime = runtime_path(output)
    spec = read_json(runtime / 'job.json')
    identity = read_json(runtime / 'process.json') if (runtime / 'process.json').exists() else {}
    result = {'runtime': str(runtime), 'output': spec['output'], 'process_alive': alive(identity),
              'process': identity, 'stop_requested': (runtime / 'stop.request').exists()}
    for name, path in [('worker', runtime / 'worker.json'), ('progress', Path(spec['output']) / 'progress.json')]:
        if path.exists():
            result[name] = read_json(path)
    return result


def _launch(runtime, spec):
    # Called under dispatch lock. The child has its own lifetime lock.
    previous = read_json(runtime / 'process.json') if (runtime / 'process.json').exists() else {}
    if alive(previous):
        raise RuntimeError('this job already has a live worker')
    if Config(**spec['config']).signature() != spec['config_sha256']:
        raise ValueError('implementation/dependencies changed; resume requires the original environment and code')
    # Probe the child lifetime lock too: PID records are not the source of locking truth.
    with RunLock(runtime):
        pass
    (runtime / 'stop.request').unlink(missing_ok=True)
    attempt = uuid.uuid4().hex
    write_json(runtime / 'worker.json', {'state': 'submitted', 'attempt': attempt, 'submitted_at': now()})
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get('PYTHONPATH', '')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env.setdefault(name, '1')
    env['PYTHONUTF8'] = '1'
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    try:
        with (runtime / 'worker.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-m', 'aaoca_preprocessing.jobs', str(runtime), attempt],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, close_fds=True, **kwargs)
    except OSError as exc:
        write_json(runtime / 'worker.json', {'state': 'failed', 'attempt': attempt, 'error': str(exc), 'updated_at': now()})
        raise
    try:
        created = psutil.Process(process.pid).create_time()
    except psutil.Error:
        created = None
    write_json(runtime / 'process.json', {'pid': process.pid, 'created': created, 'attempt': attempt, 'submitted_at': now()})
    # Reap children when embedded in a long-running Python application. Daemon
    # threads do not keep a command-line launcher alive after submission.
    threading.Thread(target=process.wait, daemon=True).start()
    return {'state': 'submitted', 'pid': process.pid, 'runtime': str(runtime), 'output': spec['output'],
            'note': 'submission is not completion; inspect status and worker.log'}


def start(input_root, output_root, config, manifest=None, scratch=None, spacing_report=None):
    source, output = Path(input_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    runtime = runtime_path(output)
    if config.target_spacing is None:
        raise ValueError('background preprocessing requires an explicit target spacing')
    if not disjoint(source, output) or not disjoint(source, runtime):
        raise ValueError('input, output and runtime must be separate')
    scratch = Path(scratch).expanduser().resolve() if scratch else output / '.scratch'
    if not disjoint(source, scratch) or not disjoint(runtime, scratch):
        raise ValueError('scratch must be separate from source and runtime')
    runtime.mkdir(parents=True, exist_ok=True)
    dispatch = runtime / '.dispatch'; dispatch.mkdir(exist_ok=True)
    with RunLock(dispatch):
        if (runtime / 'job.json').exists():
            raise ValueError('job already exists; use resume with its frozen settings')
        if any(p.name != '.dispatch' for p in runtime.iterdir()):
            raise ValueError('runtime contains unrelated files')
        cases = discover(source, manifest)
        inputs = []
        for case in cases:
            stat = case['source'].stat()
            inputs.append({'input': case['input'], 'sha256': case['expected_sha256'],
                           'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        spec = {'schema': 1, 'created_at': now(), 'input': str(source), 'output': str(output),
                'scratch': str(scratch), 'config': asdict(config), 'config_sha256': config.signature(),
                'spacing_selection': spacing_report, 'inputs': inputs}
        write_json(runtime / 'inputs.json', inputs)
        write_json(runtime / 'job.json', spec)
        return _launch(runtime, spec)


def resume(output):
    runtime = runtime_path(output)
    with RunLock(runtime / '.dispatch'):
        return _launch(runtime, read_json(runtime / 'job.json'))


def stop(output):
    runtime = runtime_path(output)
    with RunLock(runtime / '.dispatch'):
        read_json(runtime / 'job.json')
        write_json(runtime / 'stop.request', {'requested_at': now()})
    return {'state': 'stop_requested', 'note': 'cooperative; an ongoing gzip save finishes before the next checkpoint', 'runtime': str(runtime)}


def worker(runtime, attempt):
    runtime = Path(runtime)
    with RunLock(runtime):
        spec = read_json(runtime / 'job.json')
        config = Config(**spec['config'])
        def report(state, **extra):
            write_json(runtime / 'worker.json', {'attempt': attempt, 'state': state, 'updated_at': now(), **extra})
        try:
            report('checking')
            if config.signature() != spec['config_sha256']:
                raise ValueError('frozen implementation or dependencies changed')
            if read_json(runtime / 'inputs.json') != spec['inputs']:
                raise ValueError('frozen input manifest changed')
            cases = discover(spec['input'], runtime / 'inputs.json')
            for case, frozen in zip(cases, spec['inputs']):
                stat = case['source'].stat()
                if (stat.st_size, stat.st_mtime_ns) != (frozen['size'], frozen['mtime_ns']):
                    raise ValueError('input changed since job creation; use a new output/job')
            report('running')
            def progress(case, state):
                print(f"{now()} {case['case_id']} {case['state']} pending={state['pending']}", flush=True)
            result = run(spec['input'], spec['output'], config, runtime / 'inputs.json', spec['scratch'], progress,
                         stop=lambda: (runtime / 'stop.request').exists(), spacing_report=spec['spacing_selection'])
            report(result['state'], completed=result['completed'], reused=result['reused'], failed=result['failed'])
            return 1 if result['failed'] else 0
        except BaseException as exc:
            report('failed', error=f'{type(exc).__name__}: {exc}')
            print(f'{now()} {type(exc).__name__}: {exc}', flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1], sys.argv[2]))
