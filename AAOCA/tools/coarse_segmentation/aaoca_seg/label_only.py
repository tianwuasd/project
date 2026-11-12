from pathlib import Path
import sys
import time
import uuid

from .executor import checkpoint, completed, infer, merge, prune_masks
from .pipeline import require_backend, run_lock, sha256, stamp, validate_device, validate_output, write_json
from .service_state import read_json


def run_batch(cases, output, device='gpu:0', threads=2, stop_file=None):
    output = validate_output(output, cases)
    validate_device(device)
    if not 1 <= threads <= 64:
        raise ValueError('线程数须为 1–64')
    if any(case.get('preprocessing') for case in cases):
        raise ValueError('本入口直接读取原始 HU/mm CT')
    backend = require_backend()
    identity = {'backend': backend, 'format': 'slicer-seg-v1', 'device': device, 'threads': threads,
                'code': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    output.mkdir(parents=True, exist_ok=True)
    stop = Path(stop_file) if stop_file else output / 'stop'
    state = {'status': 'running', 'total': len(cases), 'completed': 0, 'reused': 0,
             'failed': 0, 'finished': 0, 'cases': [], 'started_at': stamp()}
    started = time.monotonic()
    with run_lock(output):
        try:
            for case in cases:
                if stop.exists():
                    raise KeyboardInterrupt
                key = case['case_id']
                folder = output / 'cases' / key
                old = read_json(folder / 'latest-result.json', {})
                if old.get('directory'):
                    previous = Path(old['directory']).resolve()
                    if not previous.is_relative_to(folder.resolve()):
                        raise ValueError('历史结果路径越界')
                    result = read_json(previous / 'result.json', {})
                    if result.get('identity') != identity:
                        raise ValueError('历史结果配置或依赖改变，请另建结果目录')
                    if not completed(previous / 'result.json', case, previous / 'inference.json'):
                        raise ValueError('最终标签或原始输入完整性校验失败，请另建结果目录')
                    prune_masks(read_json(previous / 'inference.json'), previous)
                    state['reused'] += 1
                    state['cases'].append({'case_id': key, 'status': 'reused', 'attempt': str(previous)})
                else:
                    saved = read_json(folder / 'latest-inference.json', {}).get('checkpoint')
                    if saved:
                        source = Path(saved).resolve()
                        if not source.is_relative_to(folder.resolve()):
                            raise ValueError('推理检查点路径越界')
                        record = checkpoint(source, case)
                        if record['identity'] != identity:
                            raise ValueError('推理检查点的代码或环境身份不符')
                        attempt = source.parent
                    else:
                        attempt = folder / ('attempt-' + uuid.uuid4().hex[:12])
                        state.update(current_case=key, phase='inference_and_export')
                        write_json(output / 'status.json', state)
                        print(f'[{key}] 三模型推理及医生编辑标签导出', flush=True)
                        infer(case, attempt, threads, device, stop, identity)
                    merge(case, attempt / 'inference.json', attempt, 'slicer-seg-v1')
                    state['completed'] += 1
                    state['cases'].append({'case_id': key, 'status': 'completed', 'attempt': str(attempt)})
                state['finished'] += 1
                write_json(output / 'status.json', state)
            state['status'] = 'completed'
        finally:
            if state['status'] == 'running':
                state['status'] = 'stopped' if sys.exc_info()[0] is KeyboardInterrupt else 'failed'
                state['failed'] += int(state['status'] == 'failed')
            state.update(ended_at=stamp(), seconds=time.monotonic() - started, current_case=None, phase='idle')
            write_json(output / 'status.json', state)
    return state
