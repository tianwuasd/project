"""Execute the frozen plan; CPU preparation precedes any GPU wait."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from server_data import digest, read_json, write_json
from task_runtime import identity, alive, load_plan, now


def child_options():
    value = os.environ.get('WHOLE_HEART_LOCK_FD')
    return {'pass_fds': (int(value),)} if value and os.name == 'posix' else {}


def register_child(process, role):
    job = os.environ.get('WHOLE_HEART_JOB')
    if job:
        write_json(Path(job) / f'processes/{process.pid}.json', dict(pid=process.pid, identity=identity(process.pid), role=role))


def run_child(command, env, cwd=None, output=None, timeout=None):
    """Interrupt verified children gently; descriptors remain held if exit is slow."""
    process = None
    cancelled = [False]
    handlers = {}
    def request(*_):
        cancelled[0] = True
    for sig in [signal.SIGINT, signal.SIGTERM]:
        handlers[sig] = signal.signal(sig, request)
    started = time.monotonic()
    record = None
    try:
        process = subprocess.Popen(command, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT if output else None,
                                   start_new_session=True, **child_options())
        record = dict(pid=process.pid, identity=identity(process.pid))
        register_child(process, 'step')
        while process.poll() is None:
            if cancelled[0] or (timeout and time.monotonic() - started > timeout):
                if alive(record):
                    os.kill(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('子进程尚未退出；保持进程记录和继承锁，不强杀或解锁。', flush=True)
                if cancelled[0]:
                    raise KeyboardInterrupt('已请求中断')
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(.2)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        return subprocess.CompletedProcess(command, process.returncode)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def record_environment(python, conda, work, env):
    """Executed inside the training environment, entirely CPU-only."""
    script = '''import importlib.metadata as m, pathlib, subprocess, sys, json
w=pathlib.Path(sys.argv[1]); c=sys.argv[2]
expected={'torch':'2.8.0+cu128','torchvision':'0.23.0+cu128','nnunetv2':'2.8.1'}
actual={k:m.version(k) for k in expected}
if actual != expected: raise ValueError('训练依赖版本不符: '+str(actual))
p=w/'06_configs';p.mkdir(parents=True,exist_ok=True)
values={'pip_freeze_training.txt':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
'conda_explicit_training.txt':subprocess.check_output([c,'list','--prefix',sys.prefix,'--explicit'],text=True)}
for name,text in values.items():
 f=p/name
 if f.exists():
  if sorted(f.read_text(encoding='utf-8').splitlines())!=sorted(text.splitlines()):raise ValueError('原环境记录不符:'+name)
 else:f.write_text(text,encoding='utf-8')
print('依赖版本与CPU环境检查通过；尚未申请GPU。',flush=True)
'''
    run_child([str(python), '-c', script, str(work), str(conda)], env)


def wait_gpu(plan):
    from evaluation_gpu import snapshot, allowed_ids, idle_reason
    settings = dict(allowed_gpus=plan['gpus'], min_free_mib=max(6000, (plan.get('gpu_memory', 5)+.5)*1024),
                    max_utilization_percent=10)
    last = 0
    while True:
        rows = snapshot()
        permitted = allowed_ids(rows, settings)
        if not permitted:
            raise ValueError('本次可见范围与冻结候选GPU没有交集')
        available = [r['uuid'] for r in rows if r['uuid'] in permitted and idle_reason(r, settings) is None]
        if available:
            return available[0]
        if time.monotonic() - last > 15:
            print('等待获准范围内的空闲GPU；CPU前置步骤已经完成。', flush=True)
            last = time.monotonic()
        time.sleep(2)


def execute(job, lock_fd):
    plan = load_plan(job)
    job, work, code = Path(job), Path(plan['workspace']), Path(plan['code'])
    os.environ.update(WHOLE_HEART_JOB=str(job), WHOLE_HEART_LOCK_FD=str(lock_fd), WHOLE_HEART_MANAGED='1')
    from runtime_setup import environment, ensure_conda
    from task_inputs import copy_input, receipt
    env = environment(Path(plan['runtime']))
    env.update(WHOLE_HEART_JOB=str(job), WHOLE_HEART_LOCK_FD=str(lock_fd), WHOLE_HEART_MANAGED='1',
               WHOLE_HEART_WORKSPACE=str(work), WHOLE_HEART_GPU_UUIDS=json.dumps(plan.get('gpus', [])),
               WHOLE_HEART_GPU_LIMIT=str(plan.get('gpu_count', 1)), WHOLE_HEART_GPU_MEMORY=str(plan.get('gpu_memory', 5)))
    # Retain inherited allocation; only individual compute children narrow CUDA.
    env['WHOLE_HEART_EVAL_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '__UNRESTRICTED__')
    state = read_json(job / 'state.json')
    state.update(status='running', pid=os.getpid(), identity=identity(os.getpid()), started=now())
    write_json(job / 'state.json', state)
    def interrupt(*_):
        raise KeyboardInterrupt('收到停止请求')
    old = signal.signal(signal.SIGTERM, interrupt)
    try:
        if plan.get('source') and not (work / '00_admin/input_receipt.json').exists():
            # Validate current raw inputs before any cache fallback or copying.
            from server_data import resolve_cases
            resolve_cases(plan['data'], read_json(Path(plan['source']['workspace']) / '04_data/manifests/case_inventory.json')['cases'])
            copy_input(plan)
        conda = ensure_conda(Path(plan['runtime']), env)
        env['CONDA_EXE'] = str(conda)
        env['WHOLE_HEART_CONDA_EXE'] = str(conda)
        python = Path(env['WHOLE_HEART_ENV_PREFIX']) / 'bin/python'
        # Installation is an actual dependency, not a GPU gate.
        run_child(['bash', str(code / 'bootstrap_server.sh'), '--install-only'], env)
        for step in plan['tasks']:
            current = state['steps'][step]
            if current['status'] == 'completed':
                print('复用已完成步骤：' + step, flush=True)
                continue
            current.update(status='running', started=now())
            state.update(current_step=step)
            write_json(job / 'state.json', state)
            logfile = job / 'logs' / f'{step}.log'
            logfile.parent.mkdir(exist_ok=True)
            current['log'] = str(logfile)
            if step == 'environment':
                current['log'] = str(job / 'launcher.log')
            if step != 'environment':
                # Binding must happen before environment files make a new workspace nonempty.
                from server_data import bind_workspace
                bind_workspace(code, work, plan['data'], plan.get('gpu_memory', 5))
                from model_catalog import save_selection
                save_selection(work, plan['folds'], plan['methods'])
                record_environment(python, conda, work, env)
                mode = {'predict': 'evaluate', 'train': 'resume' if (work / '00_admin/last_queue.json').exists() else 'train'}.get(step, step)
                step_env = dict(env)
                if step == 'smoke':
                    os.environ.update({k: v for k, v in env.items() if k.startswith('WHOLE_HEART_')})
                    step_env['CUDA_VISIBLE_DEVICES'] = wait_gpu(plan)
                    step_env['WHOLE_HEART_GPU_UUIDS'] = json.dumps([step_env['CUDA_VISIBLE_DEVICES']])
                command = [str(python), '-u', '-B', str(code / '05_src/server_launch.py'), plan['data'],
                           '--work-dir', str(work), '--mode', mode, '--folds', ','.join(plan['folds']),
                           '--models', ','.join(plan['methods']), '--gpu-memory', str(plan.get('gpu_memory', 5)),
                           '--yes-train', '--managed']
                with logfile.open('a', encoding='utf-8') as stream:
                    run_child(command, step_env, cwd=code, output=stream)
            receipt(job, step)
            current.update(status='completed', ended=now())
            write_json(job / 'state.json', state)
        state.update(status='completed', ended=now())
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error), ended=now())
        step = state.get('current_step')
        if step and state['steps'][step]['status'] != 'completed':
            state['steps'][step].update(status=state['status'], error=str(error))
        raise
    finally:
        write_json(job / 'state.json', state)
        signal.signal(signal.SIGTERM, old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    execute(args.job, args.lock_fd)
