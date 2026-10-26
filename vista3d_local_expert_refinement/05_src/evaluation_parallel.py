"""Elastic per-case GPU queue, then bounded CPU restore/metrics after GPU exit."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluation_gpu import idle_gpus, load_settings
from gpu_queue import stop_owned_process
from server_data import digest, read_json, write_json
from task_resources import prediction_budget

CODE = Path(__file__).resolve().parents[1]


def verify_environment(work):
    """只比较现有环境记录，不覆盖训练时的证据。"""
    actual = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    expected = (work / '06_configs/pip_freeze_training.txt').read_text(encoding='utf-8')
    if sorted(actual.splitlines()) != sorted(expected.splitlines()):
        raise ValueError('当前 Python 依赖与训练环境记录不符，停止评价；没有覆盖旧记录')


def run_phase(records, sha, settings, work, phase='gpu'):
    records=[dict(r) for r in records]
    path = work / '08_results/evaluation/status.json'
    logdir = work / '08_results/evaluation/queue_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    state = dict(status='selecting_gpus' if phase=='gpu' else 'cpu_tail', started=time.strftime('%F %T'),
                 phase=phase,frozen_sha256=sha, settings=settings, jobs=records)
    pending, active = deque(r for r in records if r['status']!='completed'), {}
    terminating = [False]
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: terminating.__setitem__(0, True))
    stamp = time.time_ns()
    try:
        write_json(path, state)
        while pending or active:
            if terminating[0]:
                raise InterruptedError('收到停止信号，清理本队列子进程，保留已完成病例')
            for gpu, (process, stream, record) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                stream.close()
                record.update(status='completed' if rc == 0 else 'failed', exit_code=rc)
                record['seconds']=max(0,time.monotonic()-record['started_monotonic'])
                del active[gpu]
                print(f"评价 {record['fold']}/{record['method']}: {record['status']}，日志 {record['log']}", flush=True)
                if rc:
                    raise RuntimeError(f"评价失败：{record['log']}；停止队列，已完成病例保留")
            if pending and len(active) < settings['max_gpus']:
                if phase=='cpu':
                    available=[f'CPU-{i}' for i in range(settings['max_gpus'])]
                elif os.environ.get('VISTA_GPU_LEASES'):
                    from cooperative_control import devices
                    available=idle_gpus(settings,candidates=devices(False))
                else:
                    available=idle_gpus(settings)
                available=[g for g in available if g not in active][:settings['max_gpus']-len(active)]
                for gpu in available:
                    if not pending or terminating[0]:
                        break
                    threads=None
                    if phase=='gpu':
                        budget=prediction_budget(work,list(pending)+[r for _,_,r in active.values()],settings['max_gpus'])
                        state['host_budget']=budget
                        if len(active)>=budget['workers']:
                            if not active and budget['capacity_status']=='known_insufficient_memory':
                                raise MemoryError('预测前主机RAM不足以容纳一例原图/重采样/logits估算；未启动GPU病例，请查看 prediction-host-budget.json')
                            break
                        threads=str(budget['threads_per_worker'])
                    record = pending.popleft()
                    env = os.environ.copy()
                    env['WHOLE_HEART_WORKSPACE'] = str(work)
                    env['CUDA_VISIBLE_DEVICES'] = gpu if phase=='gpu' else ''
                    env.pop('WHOLE_HEART_GPU_UUIDS', None)
                    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                        env[key] = threads if phase=='gpu' else '1'
                    if phase=='gpu':env['WHOLE_HEART_CPU_THREADS']=threads
                    if phase=='cpu':env['WHOLE_HEART_CPU_THREADS']='1'
                    logfile = logdir / f"{stamp}_{phase}_{record['fold']}_{record['method']}_{record.get('case_id','all')}.log"
                    stream = logfile.open('w', encoding='utf-8')
                    try:
                        extra = {}
                        if os.environ.get('WHOLE_HEART_MANAGED') == '1':
                            from task_worker import child_options
                            extra = child_options()
                        process = subprocess.Popen(
                            [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                             '--worker', str(record['index']), '--frozen-sha', sha,
                             '--case',record['case_id'],'--phase',phase],
                            cwd=CODE, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True, **extra)
                        if extra:
                            from task_worker import register_child
                            register_child(process, 'predict_'+phase)
                    except BaseException:
                        stream.close()
                        raise
                    active[gpu] = process, stream, record
                    record.update(status='running', gpu_uuid=gpu, pid=process.pid, log=str(logfile))
                    record['started_monotonic']=time.monotonic()
                    print(f"开始评价 {record['fold']}/{record['method']}/{record['case_id']}，{gpu}，日志：{logfile}", flush=True)
                state['status'] = ('running' if phase=='gpu' else 'cpu_tail') if active else 'waiting_for_idle_gpu'
                if not available and not active:
                    print(f"没有空闲GPU，{settings['retry_seconds']}秒后重查。", flush=True)
            write_json(path, state)
            if pending or active:
                # 分段等待便于 SIGTERM 及时收尾；不在无卡时忙循环。
                wait = 1 if active else settings['retry_seconds']
                deadline = time.monotonic() + wait
                while not terminating[0] and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline-time.monotonic())))
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        for process, stream, record in active.values():
            try:
                stop_owned_process(process)
            finally:
                stream.close()
                record['status'] = 'interrupted'
        raise
    finally:
        state['ended'] = time.strftime('%F %T')
        write_json(path, state)
        signal.signal(signal.SIGTERM, previous)


def run_queue(models,sha,settings,work):
    from evaluation_contract import cached
    if len({(m['fold'],m['method']) for m in models})!=len(models):raise ValueError('模型任务重复')
    records=[dict(fold=m['fold'],method=m['method'],index=i,case_id=c,status='pending')
             for i,m in enumerate(models) for c in m['targets']]
    for r in records:
        if cached(work,models[r['index']],r['case_id'],sha,'gpu'):r['status']='completed'
    run_phase(records,sha,settings,work,'gpu')
    if os.environ.get('VISTA_GPU_LEASES'):
        from cooperative_control import release_devices
        from gpu_fleet import verified_keepers
        release_devices(verified_keepers(os.environ['VISTA_GPU_LEASES']),wait=True,final=True)
    print('所有GPU病例进程已退出；开始CPU原网格恢复、压缩、LCC与Dice/HD95。',flush=True)
    for r in records:
        r['status']='completed' if cached(work,models[r['index']],r['case_id'],sha,'cpu') else 'pending'
    workers=max(1,int(os.environ.get('WHOLE_HEART_CPU_TAIL_WORKERS','1')))
    if os.environ.get('WHOLE_HEART_MANAGED')=='1':
        from task_resources import case_budget
        cases=read_json(work/'04_data/manifests/case_inventory.json')['cases'];targets={r['case_id'] for r in records}
        workers=case_budget(work,[c['image'] for c in cases if c['case_id'] in targets])['workers']
    cpu_settings=dict(settings,max_gpus=workers)
    run_phase(records,sha,cpu_settings,work,'cpu')
    from evaluation_summary import summarize
    summarize(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--frozen-sha')
    parser.add_argument('--case')
    parser.add_argument('--phase',choices=['gpu','cpu'],default='gpu')
    args = parser.parse_args()
    from evaluation import ROOT, configure_paths, freeze_models
    configure_paths()
    os.environ['nnUNet_n_proc_DA'] = '0'
    if args.worker is not None:
        frozen = ROOT / '08_results/evaluation/frozen_models.json'
        if digest(frozen) != args.frozen_sha:
            raise ValueError('冻结清单发生变化')
        model = read_json(frozen)['models'][args.worker]
        if digest(model['checkpoint']) != model['checkpoint_sha256']:
            raise ValueError('模型权重发生变化')
        if args.case not in model['targets']:raise ValueError('病例不在冻结目标范围内')
        import torch
        from task_resources import torch_threads
        torch_threads(torch)
        from evaluation_phases import infer,finish
        (infer if args.phase=='gpu' else finish)(ROOT,model,args.case,args.frozen_sha)
        return
    from filelock import FileLock
    # 标准入口已经持有 .run.lock；此锁另外防止两个评价协调进程同时写结果。
    with FileLock(str(ROOT / '.evaluation.lock'), timeout=0):
        status = ROOT / '08_results/evaluation/status.json'
        state = dict(status='checking_all_models', started=time.strftime('%F %T'))
        write_json(status, state)
        try:
            settings = load_settings()
            print('评价GPU参数：' + json.dumps(settings, ensure_ascii=False), flush=True)
            verify_environment(ROOT)
            models, sha = freeze_models()
        except BaseException as error:
            state.update(status='failed', error=str(error), ended=time.strftime('%F %T'))
            write_json(status, state)
            raise
        run_queue(models, sha, settings, ROOT)


if __name__ == '__main__':
    main()
