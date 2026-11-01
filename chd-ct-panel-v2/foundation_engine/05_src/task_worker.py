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
        owner=os.environ.get('VISTA_DDP_RANK_OWNER')
        write_json(Path(job) / f'processes/{process.pid}.json', dict(pid=process.pid, identity=identity(process.pid), role=role,
            ddp_invocation=os.environ.get('WHOLE_HEART_DDP_INVOCATION'),rank_owner=json.loads(owner) if owner else None))


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
    settings = dict(allowed_gpus=plan['gpus'], min_free_mib=6000,planner_gpu_memory_gib=plan.get('gpu_memory',5),
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
    os.environ.update(WHOLE_HEART_JOB=str(job), WHOLE_HEART_LOCK_FD=str(lock_fd), WHOLE_HEART_MANAGED='1',PYTHONFAULTHANDLER='1')
    from runtime_setup import environment, ensure_conda, report_progress
    from task_inputs import copy_input, receipt
    env = environment(Path(plan['runtime']))
    env.update(WHOLE_HEART_JOB=str(job), WHOLE_HEART_LOCK_FD=str(lock_fd), WHOLE_HEART_MANAGED='1',
               WHOLE_HEART_CONTROL_PYTHON=sys.executable,PYTHONFAULTHANDLER='1',
               WHOLE_HEART_WORKSPACE=str(work), WHOLE_HEART_GPU_UUIDS=json.dumps(plan.get('gpus', [])),
               WHOLE_HEART_GPU_LIMIT=str(plan.get('gpu_count', 1)), WHOLE_HEART_GPU_MEMORY=str(plan.get('gpu_memory', 5)))
    if 'global_batch_size' in plan:
        from batch_distribution import validate_global_batch
        env['WHOLE_HEART_GLOBAL_BATCH']=str(validate_global_batch(plan['global_batch_size']))
    if plan.get('gpu_policy')=='automatic':
        env.update(VISTA_GPU_POLICY='automatic',VISTA_RUNTIME=str(plan['runtime']))
    # Retain inherited allocation; only individual compute children narrow CUDA.
    env['WHOLE_HEART_EVAL_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '__UNRESTRICTED__')
    if plan.get('conda') and Path(plan['conda']).is_file():
        env['CONDA_EXE']=plan['conda']
    state = read_json(job / 'state.json')
    state.pop('ended', None)
    state.pop('error', None)
    state.update(status='running', current_step='installation', pid=os.getpid(), identity=identity(os.getpid()), started=now())
    write_json(job / 'state.json', state)
    def interrupt(*_):
        raise KeyboardInterrupt('收到停止请求')
    old = signal.signal(signal.SIGTERM, interrupt)
    fleet=None
    try:
        if plan.get('source') and not (work / '00_admin/input_receipt.json').exists():
            state.update(current_step='input')
            write_json(job / 'state.json', state)
            # Validate current raw inputs before any cache fallback or copying.
            from server_data import resolve_cases
            resolve_cases(plan['data'], read_json(Path(plan['source']['workspace']) / '04_data/manifests/case_inventory.json')['cases'])
            copy_input(plan)
        state.update(current_step='installation')
        write_json(job / 'state.json', state)
        report_progress(env, 'starting', '准备独立计算环境；CPU 前置不等待 GPU。')
        conda = ensure_conda(Path(plan['runtime']), env)
        env['CONDA_EXE'] = str(conda)
        env['WHOLE_HEART_CONDA_EXE'] = str(conda)
        python = Path(env['WHOLE_HEART_ENV_PREFIX']) / 'bin/python'
        # Installation is an actual dependency, not a GPU gate.
        run_child(['bash', str(code / 'bootstrap_server.sh'), '--install-only'], env)
        from task_resources import configure
        budget=configure(job,plan,env)
        print('CPU共享预算：'+json.dumps(budget,ensure_ascii=False),flush=True)
        # CPU-only software smoke is separate from actual-size CUDA preflight.
        run_child([str(python),'-B',str(code/'05_src/software_smoke.py'),str(job/'software-smoke.json')],env)
        for step in plan['tasks']:
            current = state['steps'][step]
            if current['status'] == 'completed':
                print('复用已完成步骤：' + step, flush=True)
                continue
            current.pop('ended', None)
            current.pop('error', None)
            current.update(status='running', started=now())
            state.update(current_step=step)
            write_json(job / 'state.json', state)
            logfile = job / 'logs' / f'{step}.log'
            logfile.parent.mkdir(exist_ok=True)
            current['log'] = str(logfile)
            if step == 'environment':
                current['log'] = str(job / 'launcher.log')
            write_json(job / 'state.json', state)
            print(f'开始步骤：{step}；日志：{current["log"]}', flush=True)
            if step != 'environment':
                # Binding must happen before environment files make a new workspace nonempty.
                from server_data import bind_workspace
                bind_workspace(code, work, plan['data'], plan.get('gpu_memory', 5))
                write_json(work/'00_admin/resource-plan.json',budget)
                from model_catalog import save_selection
                save_selection(work, plan['folds'], plan['methods'])
                record_environment(python, conda, work, env)
                if not (job/'diagnostics/resource-snapshot.json').exists():
                    from runtime_observations import collect
                    collect(job,dict(data=plan['data'],runtime=plan['runtime'],workspace=str(work)))
                mode = {'predict': 'evaluate', 'train': 'resume' if (work / '00_admin/last_queue.json').exists() else 'train'}.get(step, step)
                step_env = dict(env)
                if step == 'smoke':
                    os.environ.update({k: v for k, v in env.items() if k.startswith('WHOLE_HEART_')})
                    state.update(status='waiting_gpu')
                    write_json(job / 'state.json', state)
                    step_env['CUDA_VISIBLE_DEVICES'] = wait_gpu(plan)
                    state.update(status='running')
                    write_json(job / 'state.json', state)
                    step_env['WHOLE_HEART_GPU_UUIDS'] = json.dumps([step_env['CUDA_VISIBLE_DEVICES']])
                if plan.get('gpu_policy')=='automatic' and step in {'train','predict'}:
                    if fleet is None:
                        from gpu_fleet import GPUFleet
                        cap=plan.get('global_batch_size') if step in {'train','assessment'} else None
                        fleet=GPUFleet(job,plan['runtime'],plan['gpu_memory'],python=python,max_training_gpus=cap).start()
                    else:
                        fleet.max_training_gpus=plan.get('global_batch_size') if step in {'train','assessment'} else None
                        fleet.resume_discovery()
                    env['VISTA_GPU_LEASES']=str(fleet.path)
                    step_env['VISTA_GPU_LEASES']=str(fleet.path)
                    # Waiting happens only at a GPU stage, after CPU preprocessing.
                    fleet.devices(wait=True)
                command = [str(python), '-u', '-B', str(code / '05_src/server_launch.py'), plan['data'],
                           '--work-dir', str(work), '--mode', mode, '--folds', ','.join(plan['folds']),
                           '--models', ','.join(plan['methods']), '--gpu-memory', str(plan.get('gpu_memory', 5)),
                           '--yes-train', '--managed']
                if step=='assessment':
                    command=[str(python),'-u','-B',str(code/'05_src/assessment_run.py'),str(job/'plan.json')]
                with logfile.open('a', encoding='utf-8') as stream:
                    run_child(command, step_env, cwd=code, output=stream)
            receipt(job, step)
            current.update(status='completed', ended=now())
            print(f'完成步骤：{step}；产物与核验记录已保存。', flush=True)
            write_json(job / 'state.json', state)
        state.update(status='completed', ended=now())
    except BaseException as error:
        from runtime_observations import failure
        failure(job,state.get('current_step','unknown'),error,workspace=str(work),
            launcher_log=str(job/'launcher.log'),step_log=state.get('steps',{}).get(state.get('current_step'),{}).get('log'))
        from failure_summary import summarize
        current_step=state.get('current_step');step_log=state.get('steps',{}).get(current_step,{}).get('log')
        from datetime import datetime
        try:since=datetime.fromisoformat(state.get('steps',{}).get(current_step,{}).get('started',state.get('started'))).timestamp()
        except (TypeError,ValueError):since=0
        human_error=summarize(work,error,step_log or job/'launcher.log',since)
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=human_error, ended=now())
        step = state.get('current_step')
        if step == 'installation':
            report_progress(env, None, '环境准备未完成；请在总日志查看错误后续跑。', status=state['status'], error=str(error))
        if step in state['steps'] and state['steps'][step]['status'] != 'completed':
            state['steps'][step].update(status=state['status'], error=human_error, ended=state['ended'])
        raise
    finally:
        try:
            if fleet:fleet.close()
        except BaseException as error:
            state.update(status='failed',error='GPU保留进程清理未完成：'+str(error),ended=now())
            from runtime_observations import failure
            failure(job,'gpu_cleanup',error,workspace=str(work))
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
