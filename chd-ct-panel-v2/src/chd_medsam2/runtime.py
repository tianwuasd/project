from pathlib import Path
from collections import Counter
import os
import shutil
import subprocess
import time
from uuid import uuid4

import psutil

from chd_stunet.acceleration.prediction import process_alive, process_options, sharing_inventory
from chd_stunet.acceleration.preprocess_cpu import GIB, interpreter_command, snapshot, stop_owned
from chd_stunet.config import ROOT
from chd_stunet.observability.locking import run_lock
from chd_stunet.observability.runlog import RunLog, write_json

from .contracts import check_signature, file_sha, read_json, signed


def freeze_sources(output):
    sources = [ROOT/'run.py', *sorted((ROOT/'src').rglob('*.py')),
               *sorted((ROOT/'configs').rglob('*.json')), ROOT/'configs/models/medsam2.yaml']
    files = {}
    for source in sources:
        name = source.relative_to(ROOT)
        target = output/'code'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[name.as_posix()] = file_sha(target)
    write_json(output/'code_sources.json', files)
    return files


def frozen_job(output):
    job = read_json(output/'job.json')
    check_signature(job)
    check_signature(job['contract'])
    files = read_json(output/'code_sources.json')
    if files != job['contract']['code_files']:
        raise ValueError('MedSAM2 冻结源码清单不一致')
    for name, digest in files.items():
        path = (output/'code'/name).resolve()
        if not path.is_relative_to(output/'code') or file_sha(path) != digest:
            raise ValueError('MedSAM2 冻结源码变化：'+name)
    return job


def _spawn(output, args, filename, *, visible=None, rank_environment=None):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(output/'code/src'))
    if rank_environment is not None:
        env.update(rank_environment)
    if visible is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(visible)
    with (output/filename).open('a', encoding='utf-8') as stream:
        return subprocess.Popen([interpreter_command(env), '-u', str(output/'code/run.py'), *args],
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, **process_options())


def launch(output, contract, *, weights, train_cases, val_cases, validation_frames, requested,
           gpus, max_gpus, cpu_ranks, wait_seconds, shared_desktop, background, resume, stop_after_steps,
           recovery_root):
    output = Path(output)
    with run_lock(output.with_name(output.name+'.launch.lock')):
        launch_file = output/'launch.json'
        if launch_file.is_file() and process_alive(read_json(launch_file)):
            raise RuntimeError('该 MedSAM2 任务已有活动监督器')
        children = [read_json(path) for path in (output/'control').glob('worker-*.json')]
        if any(process_alive(child) for child in children):
            raise RuntimeError('该 MedSAM2 任务还有活动计算进程')
        if resume:
            previous = frozen_job(output)
            if (output/'summary.json').is_file() and read_json(output/'summary.json')['status'] == 'completed':
                return read_json(output/'summary.json')
            previous.pop('identity')
            if requested is not None and requested != previous['requested']:
                raise ValueError('MedSAM2 续跑须保留计算精度对应的设备类型')
            for key, value in (('gpus', gpus), ('max_gpus', max_gpus), ('cpu_ranks', cpu_ranks),
                               ('shared_desktop', shared_desktop), ('wait_seconds', wait_seconds)):
                if value is not None:
                    previous[key] = value
            job = signed(dict(previous, stop_after_steps=stop_after_steps))
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('MedSAM2 训练输出目录非空')
            output.mkdir(parents=True, exist_ok=True)
            contract = signed(dict(contract, code_files=freeze_sources(output)))
            recovery = Path(recovery_root)/f"{output.name}-{contract['identity'][:16]}"
            if any(recovery.is_relative_to(path) or path.is_relative_to(recovery)
                   for path in (output, Path(contract['prepared']))):
                raise ValueError('MedSAM2 本地恢复目录须与数据和输出目录分离')
            recovery.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(recovery).free < 2*GIB or shutil.disk_usage(output).free < 2*GIB:
                raise OSError('MedSAM2 启动需要本地和结果目录各有至少 2 GiB 检查点空间')
            write_json(recovery/'owner.json', {'output': str(output), 'identity': contract['identity']})
            job = signed(dict(contract=contract, weights=str(weights),
                              train_cases=list(train_cases), val_cases=list(val_cases),
                              validation_frames=validation_frames, requested=requested or 'cuda',
                              gpus=gpus or 'auto', max_gpus=max_gpus or 1,
                              cpu_ranks=cpu_ranks or 1, wait_seconds=wait_seconds or 0,
                              shared_desktop=shared_desktop, stop_after_steps=stop_after_steps,
                              recovery=str(recovery)))
            write_json(output/'resolved-config.json', contract)
        (output/'control').mkdir(exist_ok=True)
        for name in ('stop.json', 'pause.json'):
            (output/'control'/name).unlink(missing_ok=True)
        write_json(output/'job.json', job)
        process = _spawn(output, ['_medsam2-supervisor', str(output)], 'launcher.log')
        result = dict(status='launched', output=str(output), log=str(output/'launcher.log'),
                      pid=process.pid, born=psutil.Process(process.pid).create_time())
        write_json(launch_file, result)
    if background:
        return result
    print('MedSAM2 任务日志：'+result['log'], flush=True)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        write_json(output/'control/stop.json', {'requested': True})
        process.wait()
        return {'status': 'stopped', 'output': str(output)}
    if code:
        raise RuntimeError('MedSAM2 任务未完成，查看 '+result['log'])
    return read_json(output/'summary.json') if (output/'summary.json').is_file() else read_json(output/'status.json')


def _cards(output, selection):
    from chd_stunet.training.fleet import query_visible
    cards = query_visible(output/'code/src')
    if selection == 'auto':
        return cards
    requested = selection.split(',')
    result = [card for card in cards if str(card['ordinal']) in requested or card['uuid'] in requested]
    if not result or len(result) != len(requested):
        raise ValueError('--gpus 须使用 auto、可见编号或允许范围内的 UUID')
    return result


def _stop_workers(workers):
    for process, born in workers:
        stop_owned(dict(process=process, born=born))


def _inventory(cards, shared):
    from chd_stunet.training.fleet import query_live
    return sharing_inventory(query_live(cards), shared)


def supervise(output):
    from chd_stunet.training.fleet import eligible

    output = Path(output).resolve(strict=True)
    with run_lock(output/'.medsam2.lock'):
        job = frozen_job(output)
        log = RunLog(output)
        recovery = Path(job.get('recovery', output))
        if recovery != output and read_json(recovery/'owner.json') != {
                'output': str(output), 'identity': job['contract']['identity']}:
            raise ValueError('MedSAM2 本地恢复目录归属不匹配')
        recovery_log = RunLog(recovery)
        cards = _cards(output, job['gpus']) if job['requested']=='cuda' else []
        if job['requested']=='cuda' and not cards:
            raise ValueError('MedSAM2 当前没有可见 CUDA 设备')
        shared = job.get('shared_desktop')
        waiting_since, next_feedback = time.monotonic(), 0.
        capacity_attempts = Counter()
        reserve = max(4*GIB, int(.2*snapshot()['available_memory_bytes']))
        unknown_retries = 0
        while not (output/'control/stop.json').is_file():
            now = time.monotonic()
            storage_error = recovery/'storage-error.json'
            if storage_error.is_file():
                from .checkpoints import CheckpointMirrorError, mirror_checkpoints
                try:
                    mirror_checkpoints(recovery, output)
                except CheckpointMirrorError as error:
                    recovery_log.status('waiting_storage', reason=str(error), recovery=str(recovery))
                    recovery_log.event('training.wait_storage', '本地检查点已提交，等待结果目录恢复')
                    time.sleep(10)
                    continue
                storage_error.unlink()
            facts = snapshot()
            if facts['available_memory_bytes'] < reserve or facts['cpu_budget'] < 1:
                if now >= next_feedback:
                    log.status('waiting_memory', resources=facts, reserve_bytes=reserve)
                    log.event('training.wait_host', '主机资源低于训练保留量，等待可用资源')
                    next_feedback = now+10
                time.sleep(2)
                continue
            if job['requested'] == 'cpu':
                selected, device, visible = [], 'cpu', []
                layout = ()
            else:
                try:
                    live = _inventory(cards, shared)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    if now >= next_feedback:
                        log.status('waiting_gpu', reason='monitor', error=str(error))
                        log.event('training.wait_monitor', 'GPU 监控暂不可用，暂停准入')
                        next_feedback = now+10
                    if job['wait_seconds'] and now-waiting_since >= job['wait_seconds']:
                        raise TimeoutError('GPU 监控未在等待时限内恢复') from error
                    time.sleep(2)
                    continue
                ready = eligible(live)
                candidate_layouts = [ready[:job.get('max_gpus', 1)], *([card] for card in ready)]
                selected, layout = [], ()
                for candidate in candidate_layouts:
                    key = tuple((card['uuid'], card['usable_bytes']//(GIB//2)) for card in candidate)
                    if candidate and capacity_attempts[key] < 2:
                        selected, layout = candidate, key
                        break
                if not selected:
                    if len(ready) == len(cards):
                        log.status('needs_attention', reason='training_capacity', gpu_inventory=live)
                        raise RuntimeError('获准 GPU 均未通过当前训练尺寸容量检查')
                    if job['wait_seconds'] and now-waiting_since >= job['wait_seconds']:
                        log.status('waiting_gpu', reason='timeout', gpu_inventory=live)
                        raise TimeoutError('等待空闲 MedSAM2 GPU 超时，可恢复任务')
                    if now >= next_feedback:
                        log.status('waiting_gpu', gpu_inventory=live)
                        log.event('training.wait_gpu', '等待允许范围内的空闲 GPU')
                        next_feedback = now+10
                    time.sleep(2)
                    continue
                visible, device = [card['uuid'] for card in selected], 'cuda'
            (output/'worker-result.json').unlink(missing_ok=True)
            (output/'control/pause.json').unlink(missing_ok=True)
            for pattern in ('capacity-failure-*.json', 'fatal-*.json'):
                for path in (output/'control').glob(pattern):
                    path.unlink()
            generation = uuid4().hex
            rendezvous = (output/'control'/('collective-'+generation)).as_uri()
            world = len(visible) or job.get('cpu_ranks', 1)
            restore = str(int((recovery/'checkpoints/latest.json').is_file()))
            workers = []
            for rank in range(world):
                rank_env = dict(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank),
                                CHD_MEDSAM2_GENERATION=generation, CHD_MEDSAM2_RENDEZVOUS=rendezvous,
                                CHD_MEDSAM2_RESTORE=restore)
                filename = 'launcher.log' if rank==0 else f'control/rank-{rank}.log'
                process = _spawn(output, ['_medsam2-worker', str(output), device], filename,
                                 visible=visible, rank_environment=rank_env)
                born = psutil.Process(process.pid).create_time()
                workers.append((process, born))
                write_json(output/'control'/f'worker-{rank}.json',
                           {'pid': process.pid, 'born': born, 'generation': generation})
            if selected:
                log.event('training.admission', f'准入 {world} 张 MedSAM2 GPU', cards=selected)
            last_gpu_evidence = time.monotonic()
            last_heartbeat = last_gpu_evidence
            low_memory = 0
            original_failures = []
            while any(process.poll() is None for process, _ in workers):
                original_failures = [process.returncode for process, _ in workers if process.returncode]
                if original_failures:
                    _stop_workers(workers)
                    break
                reason = None
                if selected:
                    owned = {process.pid for process, _ in workers}
                    owned.update(proc.info['pid'] for proc in psutil.process_iter(['pid','ppid'])
                                 if proc.info['ppid'] in owned)
                    try:
                        active = _inventory(selected, shared)
                    except (RuntimeError, subprocess.TimeoutExpired):
                        if time.monotonic()-last_gpu_evidence > 15:
                            reason = 'gpu_monitor_unavailable'
                    else:
                        last_gpu_evidence = time.monotonic()
                        external = {pid for row in active for pid in row['external_pids']} - owned
                        if external:
                            reason = 'external_gpu'
                facts = snapshot()
                low_memory = low_memory+1 if facts['available_memory_bytes'] < reserve else 0
                if low_memory >= 3:
                    reason = 'host_memory_pressure'
                if reason and not (output/'control/pause.json').is_file():
                    write_json(output/'control/pause.json', {'reason': reason})
                    log.event('training.pause', '等待当前更新完成并提交，再重新准入', reason=reason)
                if time.monotonic()-last_heartbeat >= 10:
                    status_path = output/'status.json'
                    status = read_json(status_path) if status_path.is_file() else {}
                    log.event('training.heartbeat', f"阶段 {status.get('phase', '启动')}，已报告更新 {status.get('update', 0)}",
                              worker_status=status, world_size=world, available_memory_bytes=facts['available_memory_bytes'])
                    last_heartbeat = time.monotonic()
                time.sleep(2)
            result_path = output/'worker-result.json'
            result = read_json(result_path) if result_path.is_file() else {}
            state = result.get('state')
            if storage_error.is_file():
                continue
            capacity = [read_json(path) for path in (output/'control').glob('capacity-failure-*.json')]
            if capacity and selected:
                if any(row['reason']=='cuda_oom' for row in capacity):
                    capacity_attempts[layout] += 1
                log.event('training.capacity_failure', '当前 GPU 准入或显存不足，重新选择设备',
                          cards=selected, evidence=capacity)
                continue
            codes = [process.returncode for process, _ in workers]
            if any(codes):
                fatal = sorted((output/'control').glob('fatal-*.json'))
                if not fatal and unknown_retries < 1:
                    unknown_retries += 1
                    log.event('training.recover', '计算进程异常退出，从最后完整检查点重新准入',
                              returncodes=codes, attempt=unknown_retries)
                    continue
                log.status('needs_attention', reason='worker_exit', returncodes=codes,
                           fatal=[str(path) for path in fatal])
                raise RuntimeError('MedSAM2 计算进程失败，查看 launcher.log 中原始异常')
            if state in ('completed', 'stopped'):
                return
            if state != 'paused':
                raise RuntimeError('MedSAM2 计算进程未提交可验证的退出状态')
            waiting_since = time.monotonic()
        log.status('stopped', reason='operator_stop')
