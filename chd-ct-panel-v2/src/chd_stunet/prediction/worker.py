import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..observability.runlog import RunLog, write_json
from ..training.resources import GPUUnavailable
from ..acceleration.prediction import admit_prediction_device, process_alive
from .export import export_case, valid_result
from .infer import sliding_logits, tune_batch
from .inputs import prepare_case
from .model import file_sha, read_json, restore_network


class PredictionStopped(RuntimeError):
    pass


def run_worker(directory, export=False):
    directory = Path(directory)
    job = read_json(directory/'job.json')
    output = Path(job['output'])
    log = RunLog(directory)
    guard = threading.Lock()
    ended = threading.Event()
    state = {'stage': 'initialization'}

    def parent_guard():
        if not process_alive(job['supervisor']):
            write_json(directory/'finished.json', {'action': 'supervisor_lost'})
            # 只退出当前工作进程；预取线程或插值正在执行时也释放本任务资源。
            os._exit(72)

    def update(stage, **fields):
        parent_guard()
        with guard:
            state.update(stage=stage, **fields)
            log.status('running', **state)

    def heartbeat():
        last = time.monotonic()
        while not ended.wait(2):
            parent_guard()
            if time.monotonic()-last >= 10:
                with guard:
                    log.status('running', **state)
                    log.event('heartbeat', '预测运行中：'+str(state))
                last = time.monotonic()

    def progress(done, total, stage):
        if (directory/'stop.json').exists() or (output/'control/stop.json').exists():
            raise PredictionStopped('收到预测停止请求；未提交病例可恢复')
        update(stage, windows_done=done, windows_total=total)

    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    torch.set_num_threads(job['threads'])
    try:
        parent_guard()
        if export:
            record, model, inferred = job['record'], job['model'], job['inferred']
            if valid_result(job['destination'], job['identity']) is not None:
                write_json(directory/'finished.json', {'action': 'reused'})
                return
            update('export', case_id=record['case_id'])
            if file_sha(inferred['logits']) != inferred['logits_sha256']:
                raise ValueError('GPU 中间结果损坏')
            with threadpool_limits(limits=job['threads']):
                result = export_case(record, inferred['prepared'], model, inferred['logits'],
                                     job['destination'], evaluate=record['evaluate'], threads=job['threads'],
                                     keep_probabilities=job['keep_probabilities'], progress=update)
            result['inference'] = inferred['performance']
            result['preparation_seconds'] = inferred['preparation_seconds']
            write_json(Path(job['destination'])/'result.json', result)
            files = dict(result['file_sha256'], **{'result.json': file_sha(Path(job['destination'])/'result.json')})
            write_json(Path(job['destination'])/'complete.json', {'identity': job['identity'], 'files': files})
            write_json(directory/'finished.json', {'action': 'completed'})
            return
        device, admission = admit_prediction_device(job['device'], job.get('shared_desktop'))
        if job.get('gpu_memory_gib') is not None and device.type == 'cuda':
            admission['usable_bytes'] = min(admission['usable_bytes'], int(job['gpu_memory_gib']*1024**3))
        write_json(directory/'admission.json', admission)
        network = None
        model_key = None
        batch = 1
        sequence = None
        with ThreadPoolExecutor(max_workers=1) as preparation:
            while not (directory/'stop.json').exists() and not (output/'control/stop.json').exists():
                request_path = directory/'request.json'
                if not request_path.exists():
                    time.sleep(.2)
                    continue
                request = read_json(request_path)
                if request['sequence'] == sequence:
                    time.sleep(.2)
                    continue
                sequence = request['sequence']
                model = request['model']
                key = model['weights_sha256']
                if key != model_key:
                    update('load_model', fold=model['fold'])
                    del network
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    network = restore_network(model, device)
                    model_key = key
                    batch = None
                records = request['records']

                def prepare(record):
                    started = time.perf_counter()
                    work = output/'intermediate'/f"fold_{record['fold']}"/record['case_id']
                    prepared = prepare_case(record, model, work, job['threads'])
                    return prepared, time.perf_counter()-started

                future = preparation.submit(prepare, records[0])
                for index, record in enumerate(records):
                    update('prepare', case_id=record['case_id'], fold=record['fold'])
                    prepared, seconds = future.result()
                    future = preparation.submit(prepare, records[index+1]) if index+1 < len(records) else None
                    # prepare 的临时数组已释放；此处映射持久的只读输入。
                    image = np.load(prepared['image'], mmap_mode='c')
                    update('predict', case_id=record['case_id'])
                    if batch is None:
                        batch, capacity = tune_batch(network, image, model['patch_size'], device,
                                                     admission, job['max_batch'], tta=job['tta'], progress=progress)
                        write_json(directory/f"capacity-fold-{model['fold']}.json", capacity)
                        log.event('capacity', f'滑窗批量：{batch}；设备：{device}')
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    logits, performance = sliding_logits(network, image, model['patch_size'], device,
                                                          batch=batch, tta=job['tta'], step=job['tile_step'], progress=progress)
                    performance['device'] = str(device)
                    performance['gpu_uuid'] = admission.get('uuid')
                    performance['peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else 0
                    path = Path(prepared['image']).parent/'logits.npy'
                    np.save(path, logits)
                    del image, logits
                    gc.collect()
                    write_json(path.parent/'inferred.json', dict(identity=job['identity'], prepared=prepared,
                               logits=str(path), logits_sha256=file_sha(path), performance=performance,
                               preparation_seconds=seconds))
                    log.event('case_inferred', 'GPU计算已提交：'+record['case_id'])
                update('idle', windows_done=0, windows_total=0)
                write_json(directory/'response.json', {'sequence': sequence, 'action': 'completed'})
        write_json(directory/'finished.json', {'action': 'stopped'})
    except PredictionStopped:
        write_json(directory/'finished.json', {'action': 'stopped'})
    except (torch.cuda.OutOfMemoryError, MemoryError, GPUUnavailable) as error:
        write_json(directory/'failure.json', {'kind': 'resource', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
    except Exception as error:
        # 记录完整根因后重新抛出，调度器不会把未知计算/数据错误当作容量问题重试。
        write_json(directory/'failure.json', {'kind': 'fatal', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        ended.set()
        pulse.join(timeout=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    run_worker(args.directory, args.export)
