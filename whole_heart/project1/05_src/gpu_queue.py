"""多GPU独立实验队列：每个子进程仅可见一张卡，不改变单模型训练设置。"""
from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from gpu_devices import select_gpus, GPUCoolingDown
from server_data import write_json

CODE = Path(__file__).resolve().parents[1]


def check_handoff(gpu, used_before):
    """自己模型退出后的GPU利用率可能滞后；最多等10秒，外部进程仍立即拒绝。"""
    for attempt in range(11 if used_before else 1):
        try:
            select_gpus(1, gpu)
            return
        except GPUCoolingDown:
            if not used_before or attempt == 10:
                raise
            time.sleep(1)


def stop_owned_process(process):
    """只清理本队列启动的进程组，绝不按GPU编号结束其他人的任务。"""
    if process.poll() is not None:
        return
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return  # 轮询后进程恰好正常退出。
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_training_queue(tasks, gpus, work, resume=True):
    """动态分配：空闲卡领取下一模型。失败停止派新任务，运行中模型正常收尾。"""
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('队列需要不重复的GPU UUID列表')
    tasks = list(tasks)
    if len(tasks) != len(set(tasks)):
        raise ValueError('同一实验不能重复排队')
    work = Path(work)
    log_dir = work / '07_experiments/queue_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    records = [dict(fold=f, method=m, status='pending') for f, m in tasks]
    pending, active, failure = deque(records), {}, None
    used_gpus = set()
    state = dict(status='running', gpu_uuids=gpus, jobs=records)
    state_path = work / '00_admin/gpu_queue.json'
    write_json(state_path, state)
    # 信号处理器只设置标志，避免在Popen刚创建但尚未登记进程时抛异常造成遗漏。
    terminating = [False]
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def on_sigterm(signum, frame):
        terminating[0] = True
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        while pending or active:
            if terminating[0]:
                raise InterruptedError('队列收到停止信号，正在清理本队列的训练进程')
            # 先收集所有完成事件，确保同一轮已知失败后不再投递新任务。
            for gpu, (process, stream, record) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                record.update(status='completed' if code == 0 else 'failed', exit_code=code)
                del active[gpu]
                if code != 0:
                    failure = f"{record['fold']}/{record['method']} 失败，日志：{record['log']}"
                    print(f'{failure}；停止派发新任务，已运行模型继续收尾。', flush=True)
                write_json(state_path, state)
            if failure is None:
                for gpu in gpus:
                    if gpu in active or not pending:
                        continue
                    record = pending.popleft()
                    stream = None
                    try:
                        # 数据准备可能耗时较长；真正使用该卡之前再检查，而非只信启动时状态。
                        check_handoff(gpu, gpu in used_gpus)
                        if terminating[0]:
                            raise InterruptedError('队列收到停止信号')
                        env = os.environ.copy()
                        env['CUDA_VISIBLE_DEVICES'] = gpu
                        env.pop('WHOLE_HEART_GPU_UUIDS', None)
                        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                            env[key] = '2'
                        log = log_dir / f"{stamp}_{record['fold']}_{record['method']}.log"
                        stream = log.open('w', encoding='utf-8')
                        command = [sys.executable, '-u', '-B', '-X', 'utf8', str(CODE / '05_src/server_train.py'),
                                   record['fold'], record['method'], *(['--resume'] if resume else [])]
                        process = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        active[gpu] = (process, stream, record)
                        used_gpus.add(gpu)
                        record.update(status='running', gpu_uuid=gpu, pid=process.pid, log=str(log))
                        print(f"启动 {record['fold']}/{record['method']}，GPU={gpu}，日志：{log}", flush=True)
                    except Exception as error:
                        if stream is not None:
                            stream.close()
                        record.update(status='failed', error=str(error))
                        failure = f"无法启动 {record['fold']}/{record['method']}：{error}"
                        break
                    finally:
                        write_json(state_path, state)
            if failure:
                state.update(status='draining' if active else 'failed', error=failure)
                write_json(state_path, state)
                if not active:
                    raise RuntimeError(failure)
            if active:
                time.sleep(1)
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
        signal.signal(signal.SIGTERM, previous_sigterm)
        write_json(state_path, state)
