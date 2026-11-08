import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import psutil

from ..acceleration.prediction import budget, can_admit, process_options, reservation, resource_stop_reason, select_visible, stop_owned, sharing_inventory, refresh_budget, admitted_threads, process_alive
from ..acceleration.preprocess_cpu import GIB, interpreter_command
from ..config import ROOT
from ..observability.locking import run_lock
from ..observability.runlog import RunLog, write_json
from ..training.contracts import digest
from .export import valid_result
from .inputs import estimate_case, new_cases, select_cases
from .model import file_sha, freeze_weights, inspect_training, read_json, runtime_versions


def discover_training(root, folds):
    candidates = []
    for path in Path(root).glob('fold_*/*/summary.json'):
        summary = read_json(path)
        run = path.parent
        if (summary.get('status') in ('completed', 'stopped') and
                summary.get('scope') == 'formal' and (run/'checkpoints/best.json').is_file()):
            signature = read_json(run/'signature.json')
            fold = signature['data']['fold']
            if folds != 'all' and fold not in folds:
                continue
            candidates.append(dict(run=str(run), fold=fold, status=summary['status'],
                                   epoch=summary.get('epoch'), attempts=summary.get('attempts', 0),
                                   scientific=digest({'recipe': {k: v for k, v in signature['recipe'].items()
                                                               if k != 'supported_world_size'}, 'data': signature['data']})))
    selected = []
    for fold in sorted({row['fold'] for row in candidates}):
        rows = [row for row in candidates if row['fold'] == fold]
        if len({row['scientific'] for row in rows}) != 1:
            raise ValueError('同折存在不同实验，请用 --training-run 指定：'+str([r['run'] for r in rows]))
        rows.sort(key=lambda row: row['attempts'], reverse=True)
        if len(rows) > 1 and rows[0]['attempts'] == rows[1]['attempts']:
            raise ValueError('同折存在同进度候选，请用 --training-run 指定：'+str([r['run'] for r in rows]))
        selected.append(rows[0]['run'])
    if not selected or (folds != 'all' and set(folds)-{r['fold'] for r in candidates}):
        raise ValueError('没有所选折的可用训练；请检查 --training-run 或训练结果目录')
    return selected


def freeze_prediction_code(output):
    destination = Path(output)/'code'
    files = {}
    for path in [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'configs').rglob('*.json')), ROOT/'run.py', ROOT/'smoke.py']:
        relative = path.relative_to(ROOT)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[relative.as_posix()] = file_sha(target)
    write_json(Path(output)/'code_sources.json', {'files': files})
    return files


def validate_frozen(output):
    output = Path(output)
    for relative, sha in read_json(output/'code_sources.json')['files'].items():
        path = (output/'code'/relative).resolve()
        if not path.is_relative_to((output/'code').resolve()) or file_sha(path) != sha:
            raise ValueError('预测冻结代码改变：'+relative)


def assert_inactive(output):
    launch = Path(output)/'launch.json'
    if launch.is_file():
        identity = read_json(launch)
        for process in psutil.process_iter(['pid', 'create_time']):
            if process.info['pid'] == identity['pid'] and process.info['create_time'] == identity['born']:
                raise RuntimeError('该预测任务仍在运行，不能重复启动或恢复')


def validate_job(job):
    if digest({k: v for k, v in job.items() if k != 'identity'}) != job['identity']:
        raise ValueError('预测任务签名损坏')
    for name, version in runtime_versions().items():
        if job['versions'].get(name) != version:
            raise ValueError('恢复环境版本变化：'+name)


def worker_failure(directory, returncode, *, allow_success=False, supervisor_stopped=False):
    path = Path(directory)/'failure.json'
    failure = read_json(path) if path.is_file() else None
    if failure and failure.get('kind') != 'resource':
        raise RuntimeError('预测 worker 失败：'+str(failure))
    if failure:
        return failure
    if returncode is not None and not supervisor_stopped and not (allow_success and returncode == 0):
        raise RuntimeError(f'预测 worker 未正常退出（{returncode}）：{directory}/worker.log')
    return None


def reap_orphans(output, identity):
    output = Path(output).resolve()
    cleaned = []
    for path in sorted((output/'control').glob('*/owner.json')):
        owner = read_json(path)
        directory = path.parent.resolve()
        if owner['identity'] != identity or Path(owner['directory']).resolve() != directory:
            raise ValueError('工作进程所属任务不一致')
        if not directory.is_relative_to(output/'control'):
            raise ValueError('工作进程目录超出当前任务')
        if not process_alive(owner):
            continue
        try:
            process = psutil.Process(owner['pid'])
            arguments = process.cmdline()
            if (os.path.normcase(process.exe()) != os.path.normcase(owner['executable']) or
                    '--directory' not in arguments or
                    Path(arguments[arguments.index('--directory')+1]).resolve() != directory):
                raise RuntimeError('工作进程身份无法确认，拒绝终止其他进程')
            # 出生时间、解释器和任务目录均匹配后，才清理本任务遗留的进程。
            if process.create_time() != owner['born']:
                raise RuntimeError('工作进程身份变化')
            process.terminate()
        except psutil.NoSuchProcess:
            # 工作进程自身的父进程监测也会退出；并发退出已经达到清理目的。
            continue
        _, alive = psutil.wait_procs([process], timeout=5)
        for item in alive:
            if item.create_time() == owner['born']:
                item.kill()
        _, alive = psutil.wait_procs(alive, timeout=10)
        if alive:
            raise RuntimeError('遗留工作进程未退出，暂停恢复以保护输出')
        cleaned.append(owner)
    write_json(output/'control/orphan-cleanup.json', {'identity': identity, 'cleaned': cleaned})
    return cleaned


def run_prediction(options):
    output = Path(options['output']).resolve()
    if options.get('check_only'):
        return _run_prediction(options)
    # 启动锁覆盖检查和子进程创建，重复命令不会清除正在运行任务的停止请求。
    with run_lock(output.with_name(output.name+'.launch.lock')):
        assert_inactive(output)
        return _run_prediction(options)


def _run_prediction(options):
    output = Path(options['output']).resolve()
    resume = options.get('resume', False)
    if resume:
        validate_frozen(output)
        job = read_json(output/'job.json')
        validate_job(job)
        reap_orphans(output, job['identity'])
        # 恢复时使用已冻结的来源与配置，不跟随新的 best 指针。
        for key in ('tta', 'max_gpus', 'device', 'max_batch', 'checkpoint', 'gpus',
                    'gpu_memory_gib', 'threads', 'prefetch', 'wait_seconds'):
            if key in options and options[key] is not None and options[key] != job['options'][key]:
                raise ValueError('恢复参数变化：'+key)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('预测目录非空；请选择新目录或 --resume')
        skip_producer_audit = options.get('skip_producer_audit', False)
        models = [inspect_training(path, options['checkpoint'], smoke=options['smoke'],
                                   allow_intermediate=options['allow_intermediate'],
                                   skip_producer_audit=skip_producer_audit) for path in options['training_runs']]
        if len({row['fold'] for row in models}) != len(models):
            raise ValueError('每折只能选择一份模型')
        records = new_cases(options['images'], models) if options.get('images') else select_cases(
            options['prepared'], models, options.get('cases', ()))
        if options.get('limit'):
            chosen = sorted({r['case_id'] for r in records})[:options['limit']]
            records = [r for r in records if r['case_id'] in chosen]
        if options.get('check_only'):
            return dict(status='checked', folds=[m['fold'] for m in models],
                        cases=[{'case_id': r['case_id'], 'fold': r['fold']} for r in records],
                        models=[{'run': m['run'], 'epoch': m['epoch'], 'sha256': m['pointer']['sha256']} for m in models],
                        gpu_capacity_checked=False, producer_audit_skipped=skip_producer_audit)
        sources = [Path(r['pair']['image']).resolve() for r in records]
        for source in [*sources, *(Path(m['run']) for m in models), Path(options['prepared'])] if options.get('prepared') else [*sources, *(Path(m['run']) for m in models)]:
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError('预测输出必须与原数据、训练和预处理目录分开')
        output.mkdir(parents=True, exist_ok=True)
        log = RunLog(output)
        log.event('freeze', '冻结来源模型与预测代码；原训练保持只读')
        frozen = [freeze_weights(model, output/'models'/f"fold_{model['fold']}") for model in models]
        code = freeze_prediction_code(output)
        versions = runtime_versions()
        job = dict(schema=1, options=options, models=frozen, records=records, versions=versions, code_sha256=code)
        job['identity'] = digest(job)
        write_json(output/'job.json', job)
        (output/'control').mkdir()
    (output/'control/stop.json').unlink(missing_ok=True)
    launcher = output/'launcher.log'
    env = dict(os.environ, PYTHONUTF8='1')
    with launcher.open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([interpreter_command(env), '-u', str(output/'code/run.py'),
                                    '_predict-supervisor', '--output', str(output)], stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, env=env, **process_options())
    launched = dict(status='launched', pid=process.pid, output=str(output), log=str(launcher),
                    identity=job['identity'])
    write_json(output/'launch.json', dict(launched, born=psutil.Process(process.pid).create_time()))
    if options.get('background'):
        return launched
    print('预测已启动；详细日志：'+str(launcher), flush=True)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        request_stop(output)
        process.wait()
        return dict(status='stopped', output=str(output))
    if code:
        raise RuntimeError(f'预测退出码 {code}；根因见 {launcher}')
    return read_json(output/'summary.json')


def request_stop(output):
    output = Path(output).resolve(strict=True)
    if not (output/'job.json').is_file():
        raise ValueError('指定目录不是预测任务')
    write_json(output/'control/stop.json', {'requested': True, 'time': time.time()})
    return {'status': 'stop_requested', 'output': str(output)}


def wait_expired(outstanding, running, limit, since, now):
    return bool(outstanding and not running and limit and now-since > limit)


def summarize(output, job, results, state, recoveries):
    metrics = {}
    for name in ('LV', 'RV', 'LA', 'RA', 'MYO', 'AO', 'PA'):
        rows = [r['metrics'][name] for r in results if r['metrics'] is not None]
        dice = [r['dice'] for r in rows if r['dice'] is not None]
        hd = [r['hd95_mm'] for r in rows if r['hd95_mm'] is not None]
        metrics[name] = dict(mean_dice=sum(dice)/len(dice) if dice else None,
                             mean_hd95_mm=sum(hd)/len(hd) if hd else None,
                             dice_cases=len(dice), hd95_cases=len(hd),
                             missing_prediction=sum(r['status'] == 'missing_prediction' for r in rows),
                             both_empty=sum(r['status'] == 'both_empty' for r in rows))
    result = dict(status=state, scope='smoke' if job['options']['smoke'] else 'formal',
                  producer_audit_skipped=job['options'].get('skip_producer_audit', False),
                  completed_cases=len(results), requested_cases=len(job['records']), metrics=metrics,
                  folds=sorted({m['fold'] for m in job['models']}), resource_recoveries=recoveries,
                  metrics_scope=('new CT; no reference metrics' if job['options'].get('images')
                                 else 'whole-volume heldout; file-pair independence only'),
                  identity=job['identity'], results=results)
    write_json(Path(output)/'summary.json', result)
    with (Path(output)/'cases.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'fold', 'structure', 'dice', 'hd95_mm', 'status'])
        writer.writeheader()
        for row in results:
            for name, metric in (row['metrics'] or {}).items():
                writer.writerow(dict(case_id=row['case_id'], fold=row['fold'], structure=name,
                                     **{key: metric[key] for key in ('dice', 'hd95_mm', 'status')}))
    return result


def supervise(output):
    output = Path(output).resolve(strict=True)
    with run_lock(output/'.prediction.lock'):
        validate_frozen(output)
        job = read_json(output/'job.json')
        validate_job(job)
        if job['code_sha256'] != read_json(output/'code_sources.json')['files']:
            raise ValueError('冻结代码清单与任务签名不一致')
        options, identity = job['options'], job['identity']
        log = RunLog(output)
        if options.get('skip_producer_audit', False):
            log.event('producer_audit', '本任务显式跳过训练版本兼容名单检查；其他来源和权重核验通过')
        models = {m['fold']: m for m in job['models']}
        tasks = {}
        results = []
        for record in job['records']:
            key = f"fold_{record['fold']}/{record['case_id']}"
            destination = output/'cases'/key
            cached = valid_result(destination, identity)
            if cached:
                for kind, sha in record['source_hashes'].items():
                    if file_sha(record['pair'][kind]) != sha:
                        raise ValueError('恢复时来源改变：'+record['case_id']+'/'+kind)
                if (cached['case_id'] != record['case_id'] or cached['fold'] != record['fold'] or
                        cached['checkpoint_sha256'] != models[record['fold']]['pointer']['sha256']):
                    raise ValueError('已完成病例与指定来源模型不一致')
                results.append(cached)
            else:
                tasks[key] = dict(record=record, state='pending', estimate=estimate_case(record, models[record['fold']]),
                                  destination=str(destination), retries=0)
        mapping = select_visible(output/'code/src', options['gpus'], options['max_gpus']) if options['device'] == 'cuda' and tasks else []
        maximum = options['max_gpus'] if mapping else 1
        case_limit = maximum*3
        estimates = [t['estimate'] for t in tasks.values()] or [GIB]
        baseline = budget(estimates, case_limit, options['threads'])
        write_json(output/'resources-initial.json', baseline)
        workers, exporters = {}, {}
        recoveries = 0
        last_feedback = last_gpu_poll = 0.
        live = sharing_inventory(mapping, options.get('shared_desktop'))
        started_wait = time.monotonic()
        capacity_committed = False

        def spawn(directory, payload, export=False, gpu=None):
            directory.mkdir(parents=True)
            payload = dict(payload, supervisor={'pid': os.getpid(), 'born': psutil.Process().create_time()})
            write_json(directory/'job.json', payload)
            stream = (directory/'worker.log').open('w', encoding='utf-8')
            env = dict(os.environ, PYTHONPATH=str(output/'code/src'), PYTHONUTF8='1')
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[name] = str(payload['threads'])
            if gpu:
                env['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''
            arguments = [interpreter_command(env), '-u', '-m', 'chd_stunet.prediction.worker', '--directory', str(directory)]
            if export:
                arguments.append('--export')
            process = subprocess.Popen(arguments, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, **process_options())
            child = psutil.Process(process.pid)
            born = child.create_time()
            write_json(directory/'owner.json', dict(pid=process.pid, born=born, executable=child.exe(),
                       directory=str(directory), identity=identity, supervisor=payload['supervisor']))
            return dict(process=process, born=born, stream=stream,
                        directory=directory, assigned=[], threads=payload['threads'], gpu=gpu, sequence=None)

        def reservations():
            rows = []
            for entry in workers.values():
                running = [key for key in entry['assigned'] if tasks[key]['state'] == 'infer']
                amount = sum(tasks[key]['estimate'] for key in running)
                rows.append(reservation(entry['process'], entry['born'], amount+512*1024**2,
                                        entry['threads']*len(running)))
            for key, entry in exporters.items():
                rows.append(reservation(entry['process'], entry['born'], tasks[key]['estimate'], entry['threads']))
            return rows

        try:
            while any(t['state'] != 'done' for t in tasks.values()):
                if (output/'control/stop.json').exists():
                    break
                now = time.monotonic()
                fresh = budget(estimates, case_limit, options['threads'])
                if not capacity_committed:
                    # 等待阶段尚未领取资源；在首次启动前采用当前可用容量。
                    baseline = fresh
                if mapping and now-last_gpu_poll >= 5:
                    from ..training.fleet import query_live
                    live = sharing_inventory(query_live(mapping), options.get('shared_desktop'))
                    last_gpu_poll = now
                for token, entry in list(workers.items()):
                    for key in entry['assigned']:
                        marker = output/'intermediate'/key/'inferred.json'
                        if tasks[key]['state'] == 'infer' and marker.is_file():
                            inferred = read_json(marker)
                            if inferred['identity'] != identity:
                                raise ValueError('GPU 中间产物身份不一致')
                            tasks[key].update(state='ready', inferred=inferred)
                    response = entry['directory']/'response.json'
                    if response.is_file() and read_json(response)['sequence'] == entry['sequence']:
                        if any(tasks[key]['state'] == 'infer' for key in entry['assigned']):
                            raise RuntimeError('worker 返回成功但病例未提交')
                        entry['assigned'] = []
                    row = next((r for r in live if entry['gpu'] and r['uuid'] == entry['gpu']['uuid']), None)
                    foreign = row and set(row['external_pids'])-{entry['process'].pid}
                    pressure = resource_stop_reason(fresh, baseline)
                    failed = worker_failure(entry['directory'], entry['process'].poll())
                    reason = 'external_gpu' if foreign else pressure or ('resource' if failed and failed['kind'] == 'resource' else None)
                    if reason:
                        log.event('requeue', f'资源变化 {reason}；保留已完成病例并重新分配')
                        stop_owned(entry)
                        worker_failure(entry['directory'], entry['process'].returncode, supervisor_stopped=True)
                        for key in entry['assigned']:
                            if tasks[key]['state'] == 'infer':
                                marker = output/'intermediate'/key/'inferred.json'
                                if marker.is_file():
                                    inferred = read_json(marker)
                                    if inferred['identity'] != identity:
                                        raise ValueError('GPU 中间产物身份不一致')
                                    tasks[key].update(state='ready', inferred=inferred)
                                    continue
                                tasks[key]['state'] = 'pending'
                                tasks[key]['retries'] += 1
                                if tasks[key]['retries'] > 3:
                                    raise RuntimeError('同一病例资源调整超过3次，保留进度：'+key)
                        recoveries += 1
                        del workers[token]
                        if reason != 'external_gpu':
                            maximum = max(1, maximum-1)
                            options = dict(options, max_batch=max(1, options['max_batch']//2))
                            case_limit = max(1, case_limit//2)
                            baseline = dict(baseline, workers=min(baseline['workers'], case_limit))
                for key, entry in list(exporters.items()):
                    code = entry['process'].poll()
                    failure = worker_failure(entry['directory'], code, allow_success=True)
                    result = valid_result(tasks[key]['destination'], identity)
                    pressure = resource_stop_reason(fresh, baseline)
                    if result is not None:
                        stop_owned(entry)
                        worker_failure(entry['directory'], entry['process'].returncode,
                                       allow_success=True, supervisor_stopped=code is None)
                        results.append(result)
                        tasks[key]['state'] = 'done'
                        del exporters[key]
                        log.event('case_complete', f'病例完成 {len(results)}/{len(job["records"])}：{key}')
                        continue
                    if code == 0 and not failure:
                        raise RuntimeError('导出进程未提交完整病例：'+key)
                    if pressure or failure:
                        stop_owned(entry)
                        worker_failure(entry['directory'], entry['process'].returncode,
                                       allow_success=True, supervisor_stopped=code is None)
                        # 终止信号与完成提交可能相邻；再次检查，已经完成的文件不重写。
                        result = valid_result(tasks[key]['destination'], identity)
                        if result is not None:
                            results.append(result)
                            tasks[key]['state'] = 'done'
                        else:
                            tasks[key].update(state='ready', retries=tasks[key]['retries']+1)
                            if tasks[key]['retries'] > 3:
                                raise RuntimeError('导出连续遭遇内存压力，进度已保留：'+key)
                        del exporters[key]
                        recoveries += 1
                        maximum = case_limit = 1
                        baseline = dict(baseline, workers=1)
                        log.event('export_requeue', 'CPU容量不足，降低并发后重新核定容量：'+key)
                # 后处理尾段释放空闲模型，避免显存和主机内存一直被保留。
                if not any(t['state'] in ('pending', 'infer') for t in tasks.values()):
                    for token, entry in list(workers.items()):
                        if not entry['assigned']:
                            stop_owned(entry)
                            del workers[token]
                busy = bool(any(entry['assigned'] for entry in workers.values()) or exporters)
                if not busy:
                    for token, entry in list(workers.items()):
                        stop_owned(entry)
                        del workers[token]
                    fresh = budget(estimates, case_limit, options['threads'])
                    baseline = refresh_budget(baseline, fresh, busy=False)
                    if capacity_committed:
                        log.event('capacity_refresh', f"空闲阶段重新核定资源：CPU {baseline['cpu_budget']}；"
                                  f"内存 {baseline['memory_budget_bytes']/GIB:.2f} GiB；并发上限 {case_limit}")
                    capacity_committed = False
                # 优先清空已推理病例，防止大体积输出累积挤占后续内存。
                for key, task in tasks.items():
                    threads = admitted_threads(baseline, fresh, reservations(), options['threads'])
                    if task['state'] == 'ready' and can_admit(baseline, fresh, reservations(), task['estimate'], threads):
                        payload = dict(output=str(output), identity=identity, record=task['record'],
                                       model=models[task['record']['fold']], inferred=task['inferred'],
                                       destination=task['destination'], threads=threads,
                                       keep_probabilities=bool(options.get('images') and len(models) > 1))
                        exporters[key] = spawn(output/'control'/('export-'+uuid4().hex[:10]), payload, export=True)
                        capacity_committed = True
                        task['state'] = 'export'
                pending = [key for key, task in tasks.items() if task['state'] == 'pending']
                eligible = [r for r in live if not r['external_pids'] and r['usable_bytes'] > 0]
                eligible.sort(key=lambda r: -r['usable_bytes'])
                available = eligible if mapping else [None]
                for gpu in available:
                    if not pending or len(workers) >= maximum:
                        break
                    key = pending[0]
                    threads = admitted_threads(baseline, fresh, reservations(), options['threads'])
                    if not can_admit(baseline, fresh, reservations(), tasks[key]['estimate']+512*1024**2, threads):
                        continue
                    token = gpu['uuid'] if gpu else 'cpu'
                    if token in workers:
                        continue
                    payload = dict(output=str(output), identity=identity, threads=threads,
                                   device='cuda:0' if gpu else 'cpu', tta=options['tta'], tile_step=options['tile_step'],
                                   max_batch=options['max_batch'], gpu_memory_gib=options.get('gpu_memory_gib'),
                                   shared_desktop=options.get('shared_desktop'))
                    workers[token] = spawn(output/'control'/('gpu-'+uuid4().hex[:10]), payload, gpu=gpu)
                    capacity_committed = True
                for entry in workers.values():
                    if entry['assigned']:
                        continue
                    pending = [key for key, task in tasks.items() if task['state'] == 'pending']
                    if not pending:
                        continue
                    chosen = []
                    for key in pending:
                        if chosen and tasks[key]['record']['fold'] != tasks[chosen[0]]['record']['fold']:
                            continue
                        amount = sum(tasks[k]['estimate'] for k in chosen)+tasks[key]['estimate']
                        if can_admit(baseline, fresh, reservations(), amount, entry['threads']*(len(chosen)+1)):
                            chosen.append(key)
                        if len(chosen) >= (2 if options['prefetch'] else 1):
                            break
                    if chosen:
                        entry['assigned'] = chosen
                        entry['sequence'] = uuid4().hex
                        for key in chosen:
                            tasks[key]['state'] = 'infer'
                            (output/'intermediate'/key/'inferred.json').unlink(missing_ok=True)
                        write_json(entry['directory']/'request.json', dict(sequence=entry['sequence'],
                                   model=models[tasks[chosen[0]]['record']['fold']], records=[tasks[k]['record'] for k in chosen]))
                        started_wait = now
                if now-last_feedback >= 10:
                    active = [entry for entry in workers.values() if entry['assigned']]
                    state = 'running' if active or exporters else 'waiting_resources'
                    details = []
                    for entry in list(workers.values())+list(exporters.values()):
                        path = entry['directory']/'status.json'
                        if path.is_file():
                            details.append(read_json(path))
                    log.status(state, completed_cases=len(results), total_cases=len(job['records']),
                               workers=details, resources={'free_bytes': fresh['available_memory_bytes'],
                               'cpu_budget': fresh['cpu_budget'], 'gpu_count': sum(bool(e['gpu']) for e in workers.values()),
                               'gpu_inventory': live, 'smallest_case_bytes': min(estimates),
                               'memory_budget_bytes': fresh['memory_budget_bytes']})
                    stages = '；'.join(f"{row.get('case_id', '初始化')} {row.get('stage', '')} "
                                      f"{row.get('windows_done', 0)}/{row.get('windows_total', 0)}"
                                      for row in details)
                    log.event('progress', f'预测 {len(results)}/{len(job["records"])}；状态 {state}；'
                              f'计算worker {len(workers)}；CPU导出 {len(exporters)}；{stages}')
                    if state == 'waiting_resources':
                        log.event('waiting', f"可用内存 {fresh['available_memory_bytes']/GIB:.2f} GiB；"
                                  f"病例预估 {min(estimates)/GIB:.2f} GiB；"
                                  f"空闲获准GPU {len(eligible)}；外部占用PID "
                                  f"{sorted({pid for row in live for pid in row['external_pids']})}")
                    last_feedback = now
                running = bool(any(e['assigned'] for e in workers.values()) or exporters)
                outstanding = any(t['state'] != 'done' for t in tasks.values())
                if running or not outstanding:
                    started_wait = now
                if wait_expired(outstanding, running, options['wait_seconds'], started_wait, now):
                    raise TimeoutError('等待获准计算资源超时；已完成结果可续跑')
                time.sleep(.5)
            state = 'stopped' if (output/'control/stop.json').exists() else 'completed'
            for entry in list(workers.values())+list(exporters.values()):
                stop_owned(entry)
            workers.clear()
            exporters.clear()
            if state == 'completed' and options.get('images') and len(models) > 1:
                ensemble_results(output, job, results)
            summary = summarize(output, job, results, state, recoveries)
            log.status(state, completed_cases=len(results), total_cases=len(job['records']))
            log.event('finished', f'预测{state}；完成 {len(results)}/{len(job["records"])}')
            return summary
        except Exception as error:
            write_json(output/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
            summarize(output, job, results, 'failed', recoveries)
            log.status('failed', error=str(error), completed_cases=len(results))
            raise
        finally:
            for entry in list(workers.values())+list(exporters.values()):
                stop_owned(entry)


def ensemble_results(output, job, results):
    import numpy as np
    from .export import save_nifti
    for cid in sorted({r['case_id'] for r in results}):
        rows = [r for r in results if r['case_id'] == cid]
        if len(rows) != len(job['models']):
            raise ValueError('三折集成缺少指定模型结果：'+cid)
        if any(row['shape'] != rows[0]['shape'] or not np.allclose(row['affine'], rows[0]['affine']) for row in rows):
            raise ValueError('多折结果尚未恢复到共同网格')
        arrays = [np.load(row['probabilities'], mmap_mode='r') for row in rows]
        prediction = np.zeros(rows[0]['shape'], dtype=np.uint8)
        for start in range(0, prediction.shape[0], 8):
            block = sum(np.asarray(array[:, start:start+8], dtype=np.float32) for array in arrays)/len(arrays)
            prediction[start:start+8] = block.argmax(0)
        destination = Path(output)/'ensemble'
        destination.mkdir(exist_ok=True)
        path = destination/(cid+'.nii.gz')
        save_nifti(path, prediction, np.asarray(rows[0]['affine']))
        write_json(destination/(cid+'.json'), dict(case_id=cid, folds=[r['fold'] for r in rows],
                   rule='mean probabilities on verified native grid', sha256=file_sha(path),
                   reference_image=rows[0]['reference_image'], identity=job['identity']))
