"""One bounded queue for inference, verification and CPU merge workers."""
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .executor import lock_fds
from .images import inspect_image
from .pipeline import environment, require_backend, sha256
from .preprocessed import validate_case
from .resources import GIB, available_devices, budget, child_env, estimate
from .service_state import file_lock, process_identity, read_json, timestamp, write_json


def gpu_info():
    # A short-lived probe avoids creating CUDA contexts in the queue owner.
    from .operations import PROBE
    result = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True,
                            encoding='utf-8', timeout=60, check=True)
    import json
    return json.loads(result.stdout.strip().splitlines()[-1])


def run(job, probe=None, gpu_query=None, allocation=None, popen=None, backend=None):
    job = Path(job)
    request = read_json(job / 'request.json')
    settings, cases = request['settings'], request['cases']
    staged = bool(settings.get('prepare_workers')) and not request.get('preprocessing_source')
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    stop = job / 'stop'
    selected = settings['devices'] if 'infer' in request['steps'] else []
    alloc = allocation or budget(settings)
    write_json(job / 'resources.json', alloc)
    identity = {'versions': (backend or environment())['versions'],
                'source': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                'method': ('aaoca-totalseg-2.18.0-native-hu-slicer-seg-v1' if request.get('output_format') == 'slicer-seg-v1' else
                           'aaoca-totalseg-2.18.0-resampled-hu-11-v1' if request.get('preprocessing_source')
                           else 'aaoca-totalseg-2.18.0-native-hu-11-v1')}
    previous = read_json(job / 'identity.json')
    if previous and previous != identity:
        raise ValueError('依赖版本或代码快照改变，请恢复原环境或新建任务')
    write_json(job / 'identity.json', identity)
    if 'infer' in request['steps'] and backend is None:
        require_backend()
    state = {'status': 'running', 'total': len(cases), 'finished': 0, 'failed': 0,
             'reused': 0, 'active': {}, 'cases': [], 'started_at': timestamp(), 'max_parallel': 0}
    pending, active = [], {}
    frozen_inputs = read_json(job / 'inputs.json', {})
    # All header checks happen on CPU before selecting or waiting for a GPU.
    for case in cases:
        try:
            if stop.exists():
                break
            if case.get('preprocessing'):
                print(f"[{case['case_id']}] CPU 预处理配套影像与空间信息核验", flush=True)
                validate_case(case, full=True)
            before = Path(case['image']).stat()
            signature = {'size': before.st_size, 'mtime_ns': before.st_mtime_ns}
            if case.get('source_stat') != signature:
                raise ValueError('提交后原始输入改变，请新建任务')
            geom = inspect_image(case['image'])
            # Freeze every selected source before any GPU wait. Resumes compare
            # against the original digest, including cases never dispatched.
            if any(step in request['steps'] for step in ('infer', 'merge')) and not staged:
                print(f"[{case['case_id']}] CPU 输入哈希核验", flush=True)
                source_hash = sha256(case['image'])
                after = Path(case['image']).stat()
                if signature != {'size': after.st_size, 'mtime_ns': after.st_mtime_ns}:
                    raise ValueError('读取过程中原始输入改变')
                key = case['case_id']
                if key in frozen_inputs and frozen_inputs[key] != source_hash:
                    raise ValueError('冻结的原始输入哈希改变')
                frozen_inputs[key] = source_hash
                write_json(job / 'inputs.json', frozen_inputs)
                case = dict(case, source_sha256=source_hash)
            elif case['case_id'] in frozen_inputs:
                case = dict(case, source_sha256=frozen_inputs[case['case_id']])
            memory = estimate(geom)
            if not staged and memory > alloc['memory_bytes']:
                raise ValueError(f'病例内存估算 {memory/GIB:.1f} GiB 超出预算 {alloc["memory_bytes"]/GIB:.1f} GiB')
            if not staged and shutil.disk_usage(output).free < math_disk(geom):
                raise ValueError('结果磁盘不足以保存本病例临时解压和掩膜')
            own = read_json(output / 'cases' / case['case_id'] / 'latest-inference.json', {})
            source = own.get('checkpoint') or request.get('inputs', {}).get(case['case_id'])
            phase = 'verify' if source else (('prepare' if staged else 'infer') if 'infer' in request['steps'] else 'plan')
            if staged and phase in {'prepare', 'infer'} and memory > alloc['memory_bytes']:
                raise ValueError(f'病例推理内存估算 {memory/GIB:.1f} GiB 超出预算')
            if phase == 'plan' and 'merge' in request['steps']:
                raise ValueError('合并步骤需要本次推理或历史输入任务的完整掩膜')
            prior_result = read_json(output / 'cases' / case['case_id'] / 'latest-result.json', {})
            prepared = read_json(output / 'cases' / case['case_id'] / 'latest-prepared.json', {}).get('checkpoint') if staged else None
            if prepared and not Path(prepared).resolve().is_relative_to((output / 'cases' / case['case_id']).resolve()):
                raise ValueError('CPU 准备记录路径越界')
            attempt = prior_result.get('directory') or (str(Path(prepared).parent) if prepared else None) or str(output / 'cases' / case['case_id'] / ('attempt-' + uuid.uuid4().hex[:12]))
            pending.append({'case': case, 'memory': memory, 'phase': phase, 'source': source,
                            'prepare_memory': GIB + geom['shape'][0] * geom['shape'][1] * min(32, geom['shape'][2]) * 16,
                            'prepare_disk': (math_disk(geom) - GIB) // 4 + GIB,
                            'export_disk': (math_disk(geom) - GIB) * 3 // 8 + GIB,
                            'own': bool(own), 'attempt': attempt, 'disk_bytes': math_disk(geom), 'admitted': False})
        except Exception as error:
            state['cases'].append({'case_id': case['case_id'], 'status': 'failed', 'error': str(error)})
            state['failed'] += 1
            state['finished'] += 1
    write_json(job / 'plan.json', {'cases': pending, 'invalid': state['cases'], 'resources': alloc})
    started = time.monotonic()
    warm = False
    current = None  # CUDA queried only if an actual inference needs dispatch.
    next_probe = 0
    free = []
    popen = popen or subprocess.Popen
    gpu_query = gpu_query or available_devices
    with file_lock(output / '.queue.lock') as output_fd:
        inherited = (*lock_fds(), output_fd) if os.name != 'nt' else ()
        try:
            while pending or active:
                if stop.exists():
                    pending.clear()
                    state['status'] = 'stopping'
                for key, running in list(active.items()):
                    child, item = running['child'], running['item']
                    code = child.poll()
                    if code is None:
                        continue
                    running['log'].close()
                    del active[key]
                    state['active'].pop(key, None)
                    phase = item['phase']
                    append_event(job, item, 'finished', running.get('device'), exit_code=code)
                    if code:
                        state['cases'].append({'case_id': key, 'status': 'stopped' if stop.exists() else 'failed',
                                               'phase': phase, 'log': running['log_path']})
                        state['failed'] += int(not stop.exists())
                        state['finished'] += 1
                    elif phase == 'prepare':
                        receipt = Path(item['attempt']) / 'prepared.json'
                        prepared = read_json(receipt)
                        if not prepared or prepared.get('status') != 'prepared':
                            raise ValueError('CPU 准备进程缺少完成记录')
                        digest = prepared['source_sha256']
                        if key in frozen_inputs and frozen_inputs[key] != digest:
                            raise ValueError('冻结的原始输入哈希改变')
                        frozen_inputs[key] = digest
                        write_json(job / 'inputs.json', frozen_inputs)
                        item['case'] = dict(item['case'], source_sha256=digest)
                        item.update(phase='infer', prepared=str(receipt), ready_at=time.monotonic())
                        pending.insert(0, item)
                    elif phase == 'infer':
                        warm = True
                        item['source'] = str(Path(item['attempt']) / 'inference.json')
                        write_json(output / 'cases' / key / 'latest-inference.json', {'checkpoint': item['source']})
                        if 'merge' in request['steps'] and not stop.exists():
                            item['phase'] = 'merge'
                            pending.insert(0, item)
                        else:
                            finish(state, item, 'inferred')
                    elif phase == 'verify':
                        report = read_json(Path(item['attempt']) / 'verification.json', {})
                        if not report.get('valid'):
                            raise ValueError('检查进程缺少完成记录')
                        if 'merge' in request['steps'] and not report.get('completed') and not stop.exists():
                            item['phase'] = 'merge'
                            # Preserve a damaged previous result; publish a new attempt.
                            item['attempt'] = str(output / 'cases' / key / ('attempt-' + uuid.uuid4().hex[:12]))
                            pending.insert(0, item)
                        else:
                            state['reused'] += 1
                            finish(state, item, 'reused')
                    else:
                        write_json(output / 'cases' / key / 'latest-result.json', {'directory': item['attempt']})
                        finish(state, item, 'completed')
                    free = []  # Recheck occupancy after releasing an inference slot.
                    next_probe = 0
                waiting_gpu = False
                waiting_disk = False
                # Keep available GPUs busy before consuming the same bounded
                # CPU/RAM slots with optional merge tails.
                pending.sort(key=lambda row: {'verify': 0, 'infer': 1, 'plan': 2, 'merge': 3, 'prepare': 4}[row['phase']])
                for item in list(pending):
                    if stop.exists():
                        break
                    if item['phase'] == 'plan':
                        pending.remove(item)
                        finish(state, item, 'planned')
                        continue
                    if len(active) >= alloc['case_slots']:
                        break
                    if staged:
                        if not stage_capacity(item, pending, [r['item'] for r in active.values()], settings):
                            continue
                    used = sum(phase_memory(r['item']) for r in active.values())
                    if phase_memory(item) > alloc['memory_bytes']:
                        raise ValueError('下一阶段的内存估算超过总预算')
                    if used + phase_memory(item) > alloc['memory_bytes']:
                        continue
                    if staged:
                        disk_held = sum(phase_disk(r['item']) for r in active.values())
                        if shutil.disk_usage(output).free < disk_held + phase_disk(item):
                            if not active:
                                raise ValueError('运行期间磁盘不足以启动下一阶段，请释放空间后续跑')
                            waiting_disk = True
                            continue
                    elif not item['admitted']:
                        admitted = [r['item'] for r in active.values()] + [row for row in pending if row is not item and row['admitted']]
                        disk_held = sum(row['disk_bytes'] for row in admitted)
                        if shutil.disk_usage(output).free < disk_held + item['disk_bytes']:
                            if not admitted:
                                raise ValueError('运行期间磁盘可用空间不足，已停止派发病例')
                            waiting_disk = True
                            continue
                    device = None
                    if item['phase'] == 'infer' and selected:
                        gpu_active = [r for r in active.values() if r.get('device')]
                        if len(gpu_active) >= settings['gpu_count'] or (not warm and gpu_active):
                            continue
                        if time.monotonic() >= next_probe:
                            current = current or (probe or gpu_info)()
                            if settings.get('allow_shared_gpu'):
                                free = gpu_query(selected, current, settings['min_gpu_gib'], allow_shared=True,
                                                 reserve_gib=settings.get('gpu_reserve_gib', 0))
                            else:
                                free = gpu_query(selected, current, settings['min_gpu_gib'])
                            next_probe = time.monotonic() + 2
                        busy = {r['device']['uuid'] for r in gpu_active}
                        device = next((d for d in free if d['uuid'] not in busy), None)
                        if not device:
                            waiting_gpu = True
                            continue
                    elif item['phase'] == 'infer' and not warm and any(r['item']['phase'] == 'infer' for r in active.values()):
                        continue
                    pending.remove(item)
                    item['admitted'] = True
                    target = Path(item['attempt'])
                    target.mkdir(parents=True, exist_ok=True)
                    payload = dict(item, threads=alloc['threads'], device='gpu:0' if device else 'cpu',
                                   output_format=request.get('output_format', 'legacy'),
                                   stop=str(stop), identity=identity,
                                   expected_identity=identity if item['own'] else None,
                                   verification=str(target / 'verification.json'))
                    if device and settings.get('allow_shared_gpu'):
                        payload['gpu_guard'] = {'uuid': device['uuid'], 'minimum_gib': settings['min_gpu_gib'],
                                                'reserve_gib': settings.get('gpu_reserve_gib', 0)}
                    path = target / (item['phase'] + '-request.json')
                    write_json(path, payload)
                    env = child_env(alloc['threads'], device)
                    env['AAOCA_LOCK_FDS'] = ','.join(map(str, inherited))
                    log_path = target / (item['phase'] + '-worker.log')
                    log = log_path.open('ab')
                    try:
                        child = popen([sys.executable, '-u', '-m', 'aaoca_seg.executor', str(path)],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                      **({'pass_fds': inherited} if os.name != 'nt' else {}))
                    except BaseException:
                        log.close()
                        raise
                    key = item['case']['case_id']
                    write_json(job / 'processes' / f'{key}.json', process_identity(child.pid))
                    active[key] = {'child': child, 'item': item, 'device': device, 'log': log, 'log_path': str(log_path)}
                    append_event(job, item, 'started', device)
                    state['active'][key] = {'phase': item['phase'], 'device': device, 'attempt': str(target),
                                            'pid': child.pid, 'memory_gib': phase_memory(item) / GIB}
                    print(f"[{key}] {item['phase']} / {device['uuid'] if device else 'CPU'}", flush=True)
                state['max_parallel'] = max(state['max_parallel'], len(active))
                state['waiting_gpu'] = waiting_gpu
                state['waiting_disk'] = waiting_disk
                state['pending'] = len(pending)
                state['ready'] = sum(item['phase'] == 'infer' and bool(item.get('prepared')) for item in pending)
                state['preparing'] = sum(r['item']['phase'] == 'prepare' for r in active.values())
                elapsed = time.monotonic() - started
                state['elapsed_seconds'] = round(elapsed, 1)
                state['eta_seconds'] = round(elapsed / state['finished'] * (len(cases) - state['finished'])) if state['finished'] and not waiting_gpu and not state['failed'] else None
                write_json(output / 'status.json', state)
                if pending or active:
                    time.sleep(.2)
        except BaseException:
            stop.touch()
            # Registered case owners terminate their own model process groups.
            for running in active.values():
                running['child'].wait()
                running['log'].close()
            raise
    state['status'] = 'stopped' if stop.exists() else ('completed_with_errors' if state['failed'] else 'completed')
    state['ended_at'] = timestamp()
    write_json(output / 'status.json', state)
    return state


def math_disk(geometry):
    import math
    return int(math.prod(geometry['shape']) * 32 + GIB)


def phase_memory(item):
    if item['phase'] == 'verify':
        return GIB
    return item['prepare_memory'] if item['phase'] == 'prepare' else item['memory']


def stage_capacity(item, pending, active, settings):
    phases = [row['phase'] for row in active]
    if item['phase'] == 'prepare':
        ready = sum(row['phase'] == 'infer' and bool(row.get('prepared')) for row in pending)
        return (phases.count('prepare') < settings['prepare_workers'] and
                ready + phases.count('prepare') < settings['prefetch'])
    if item['phase'] in {'merge', 'verify'}:
        return sum(phase in {'merge', 'verify'} for phase in phases) < settings['export_workers']
    return True


def phase_disk(item):
    if item['phase'] == 'prepare':
        return item['prepare_disk']
    if item['phase'] == 'merge':
        return item['export_disk']
    if item['phase'] == 'verify':
        return 1024 * 1024
    return item['disk_bytes']


def append_event(job, item, event, device, exit_code=None):
    import json
    record = {'case_id': item['case']['case_id'], 'phase': item['phase'], 'event': event,
              'time': timestamp(), 'monotonic': time.monotonic(), 'device': device, 'exit_code': exit_code}
    if event == 'started' and item.get('ready_at') and item['phase'] == 'infer':
        record['ready_wait_seconds'] = record['monotonic'] - item['ready_at']
    with (Path(job) / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def finish(state, item, status):
    state['finished'] += 1
    state['cases'].append({'case_id': item['case']['case_id'], 'status': status,
                           'attempt': item['attempt'], 'inference': item.get('source')})


def main(job):
    job = Path(job)
    def stop(*_):
        (job / 'stop').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        result = run(job)
        return 0 if result['status'] == 'completed' else (130 if result['status'] == 'stopped' else 2)
    except Exception as error:
        print(str(error), flush=True)
        return 2
