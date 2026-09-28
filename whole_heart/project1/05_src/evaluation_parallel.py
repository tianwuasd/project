"""最终评价自动使用最多三张空闲卡；一张卡一个模型，全部成功后才汇总。

只有协调进程写队列状态；子进程只写自己的模型目录。预测和指标沿用 evaluation.py。
"""
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

CODE = Path(__file__).resolve().parents[1]


def verify_environment(work):
    """只比较现有环境记录，不覆盖训练时的证据。"""
    actual = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    expected = (work / '06_configs/pip_freeze_training.txt').read_text(encoding='utf-8')
    if sorted(actual.splitlines()) != sorted(expected.splitlines()):
        raise ValueError('当前 Python 依赖与训练环境记录不符，停止评价；没有覆盖旧记录')


def run_queue(models, sha, settings, work):
    records = [dict(fold=m['fold'], method=m['method'], index=i, status='pending')
               for i, m in enumerate(models)]
    if len({(r['fold'], r['method']) for r in records}) != len(records):
        raise ValueError('模型任务重复')
    path = work / '08_results/evaluation/status.json'
    logdir = work / '08_results/evaluation/queue_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    state = dict(status='selecting_gpus', started=time.strftime('%F %T'),
                 frozen_sha256=sha, settings=settings, jobs=records)
    pending, active, chosen = deque(records), {}, None
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
                del active[gpu]
                print(f"评价 {record['fold']}/{record['method']}: {record['status']}，日志 {record['log']}", flush=True)
                if rc:
                    raise RuntimeError(f"评价失败：{record['log']}；停止队列，已完成病例保留")
            if pending and len(active) < settings['max_gpus']:
                candidates = None if chosen is None else [g for g in chosen if g not in active]
                available = idle_gpus(settings, candidates) if candidates is None or candidates else []
                if chosen is None and available:
                    chosen = available[:settings['max_gpus']]
                    available = chosen
                    state['gpu_uuids'] = chosen
                    print(f'评价自动选择 {len(chosen)} 张GPU：{chosen}', flush=True)
                for gpu in available:
                    if not pending or terminating[0]:
                        break
                    record = pending.popleft()
                    env = os.environ.copy()
                    env['WHOLE_HEART_WORKSPACE'] = str(work)
                    env['CUDA_VISIBLE_DEVICES'] = gpu
                    env.pop('WHOLE_HEART_GPU_UUIDS', None)
                    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                        env[key] = '2'
                    logfile = logdir / f"{stamp}_{record['fold']}_{record['method']}.log"
                    stream = logfile.open('w', encoding='utf-8')
                    try:
                        process = subprocess.Popen(
                            [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                             '--worker', str(record['index']), '--frozen-sha', sha],
                            cwd=CODE, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close()
                        raise
                    active[gpu] = process, stream, record
                    record.update(status='running', gpu_uuid=gpu, pid=process.pid, log=str(logfile))
                    print(f"开始评价 {record['fold']}/{record['method']}，GPU={gpu}，日志：{logfile}", flush=True)
                state['status'] = 'running' if active else 'waiting_for_idle_gpu'
                if not available and not active:
                    print(f"没有空闲GPU，{settings['retry_seconds']}秒后重查。", flush=True)
            write_json(path, state)
            if pending or active:
                # 分段等待便于 SIGTERM 及时收尾；不在无卡时忙循环。
                wait = 1 if active else settings['retry_seconds']
                deadline = time.monotonic() + wait
                while not terminating[0] and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline-time.monotonic())))
        state['status'] = 'summarizing'
        write_json(path, state)
        from evaluation_summary import summarize
        summarize(work)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--frozen-sha')
    args = parser.parse_args()
    from evaluation import ROOT, configure_paths, freeze_models, evaluate_model
    configure_paths()
    os.environ['nnUNet_n_proc_DA'] = '0'
    if args.worker is not None:
        frozen = ROOT / '08_results/evaluation/frozen_models.json'
        if digest(frozen) != args.frozen_sha:
            raise ValueError('冻结清单发生变化')
        model = read_json(frozen)['models'][args.worker]
        if digest(model['checkpoint']) != model['checkpoint_sha256']:
            raise ValueError('模型权重发生变化')
        import torch
        torch.set_num_threads(2)
        evaluate_model(model, args.frozen_sha)
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
