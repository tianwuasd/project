"""Stdlib deployment/checks with live console and file feedback for each profile.

Uses only the standard library. An explicit --install targets an isolated venv,
never the global interpreter. Preprocessing uses CPU Torch, without weights,
GPU allocation. Training deployment uses its own explicitly selected profile.
"""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
MODULES=("numpy","scipy","nibabel","matplotlib","psutil","threadpoolctl")


def report(message, stream=None):
    """Flush a readable event to the console and, when supplied, its run log."""
    line=f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}'
    print(line,flush=True)
    if stream is not None:
        stream.write(line+'\n');stream.flush()


def _stop_owned_process(process):
    """Stop this launched command/session, never matching external processes."""
    if os.name!='nt':
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
        # A build helper may outlive pip; the private group retains its ID.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    elif process.poll() is None:
        process.terminate()
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:process.kill()
    process.wait()


def run_logged(command, stream, label, *, env=None, timeout=1800, heartbeat=10):
    """Tee a real child command while it runs, with bounded capture and silence feedback.

    Only the reader touches the pipe; the parent owns both console/file writes.
    The bounded queue applies backpressure instead of storing unlimited pip text.
    The full log is durable after each line; only a 64KiB tail is kept in JSON.
    """
    child_env=dict(os.environ if env is None else env)
    child_env['PYTHONUNBUFFERED']='1'
    child_env['PYTHONIOENCODING']='utf-8'
    child_env.setdefault('PIP_PROGRESS_BAR','off')
    child_env.setdefault('PIP_NO_INPUT','1')
    child_env.setdefault('PIP_DISABLE_PIP_VERSION_CHECK','1')
    began=time.monotonic();last_output=began;tail='';messages=queue.Queue(maxsize=256)
    finished=threading.Event();reader_stop=threading.Event()
    report('开始：'+label,stream)
    process=subprocess.Popen(command,env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',errors='replace',bufsize=1,start_new_session=os.name!='nt')
    def read_output():
        try:
            for line in iter(lambda:process.stdout.readline(16384),''):
                while not reader_stop.is_set():
                    try:messages.put(line,timeout=.1);break
                    except queue.Full:continue
                if reader_stop.is_set():break
        finally:finished.set()
    reader=threading.Thread(target=read_output,daemon=True);reader.start()
    timed_out=False
    try:
        while not (finished.is_set() and messages.empty() and process.poll() is not None):
            now=time.monotonic()
            if now-began>=timeout:
                timed_out=True
                report(f'超时：{label}，已等待{timeout:g}秒，正在停止本步骤',stream)
                _stop_owned_process(process);break
            try:line=messages.get(timeout=.1)
            except queue.Empty:
                if now-last_output>=heartbeat:
                    report(f'等待输出：{label}；已运行{now-began:.0f}秒',stream)
                    last_output=now
                continue
            print(line,end='',flush=True);stream.write(line);stream.flush()
            tail=(tail+line)[-65536:];last_output=time.monotonic()
        code=124 if timed_out else process.wait()
    except BaseException:
        _stop_owned_process(process)
        report('已停止：'+label,stream)
        raise
    finally:
        reader_stop.set();reader.join(timeout=3)
        while not messages.empty():
            line=messages.get_nowait()
            print(line,end='',flush=True);stream.write(line);stream.flush()
            tail=(tail+line)[-65536:]
        if not reader.is_alive():process.stdout.close()
    elapsed=time.monotonic()-began
    report(f'{"完成" if code==0 else "失败"}：{label}；退出码{code}；耗时{elapsed:.1f}秒',stream)
    return subprocess.CompletedProcess(command,code,stdout=tail,stderr='')


def run_waiting(action,stream,label,heartbeat=10):
    """Report blocking in-process work without claiming a measured percentage."""
    began=time.monotonic();stop=threading.Event()
    report('开始：'+label,stream)
    def notify():
        while not stop.wait(heartbeat):report(f'等待：{label}；已运行{time.monotonic()-began:.0f}秒',stream)
    thread=threading.Thread(target=notify,daemon=True);thread.start()
    try:return action()
    except BaseException as error:
        report(f'失败：{label}；{type(error).__name__}: {error}',stream)
        raise
    finally:stop.set();thread.join()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--plan",action="store_true")
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--install",action="store_true")
    parser.add_argument("--profile",choices=["analysis","preprocess","training"],default="preprocess")
    parser.add_argument('--cpu',action='store_true',help='training installation uses CPU Torch instead of CUDA128')
    parser.add_argument('--download-weights',action='store_true',help='with training --install, fetch/check pinned author weights')
    parser.add_argument('--weights',help='optional author weight cache path')
    parser.add_argument("--python")
    parser.add_argument("--venv")
    parser.add_argument('--proxy',help='optional temporary pip proxy, never saved globally')
    args=parser.parse_args()
    local=ROOT/"project.local.json"
    saved=json.loads(local.read_text(encoding="utf-8")) if local.is_file() else {}
    interpreter_key={'training':'training_python','preprocess':'preprocess_python','analysis':'analysis_python'}[args.profile]
    if args.download_weights and (args.profile!='training' or not args.install):
        parser.error('--download-weights requires --profile training --install')
    python=args.python or saved.get(interpreter_key) or saved.get("analysis_python") or sys.executable
    venv=Path(args.venv or ROOT/('.venv_training' if args.profile=='training' else '.venv_preprocess' if args.profile=='preprocess' else '.venv_analysis')).resolve()
    vpython=venv/("Scripts/python.exe" if sys.platform=="win32" else "bin/python")
    modules=MODULES+('nnunetv2','blosc2','torch','skimage') if args.profile in ('preprocess','training') else MODULES
    requirement_file={'analysis':'requirements-analysis.txt','preprocess':'requirements-preprocess.txt','training':'requirements-training.txt'}[args.profile]
    record=dict(profile=args.profile,source_python=python,target_venv=str(venv),
        requirements=str(ROOT/requirement_file),modules=modules,
        action="install" if args.install else "check" if args.check else "plan",
        scope=('training software and optional pinned weight cache; GPU admission is a separate train-stage probe'
               if args.profile=='training' else 'CPU preprocessing and analysis; no GPU, model weights or training deployment'
               if args.profile=='preprocess' else 'read-only analysis'))
    import hashlib
    record['requirements_sha256']=hashlib.sha256(Path(record['requirements']).read_bytes()).hexdigest()
    torch_index='https://download.pytorch.org/whl/cu128' if args.profile=='training' and not args.cpu else 'https://download.pytorch.org/whl/cpu'
    record['network']=dict(packages='PyPI',torch=dict(version='2.8.0',index=torch_index) if args.profile!='analysis' else None,
                           temporary_proxy_requested=bool(args.proxy))
    logroot=ROOT/"outputs/environment"
    logroot.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    log_path=logroot/(stamp+'.'+record['action']+'.log')
    record['log_path']=str(log_path)
    with log_path.open('w',encoding='utf-8') as stream:
        report(f'环境阶段：{record["action"]}；配置：{args.profile}',stream)
        report('完整日志：'+str(log_path),stream)
        report('执行解释器：'+str(python),stream)
        if args.install:report('部署位置：'+str(venv),stream)
    if args.install:
        if venv.exists() and not (venv/"pyvenv.cfg").is_file():
            raise ValueError("Target exists and is not a recognized venv; refusing to overwrite")
        with log_path.open("a",encoding="utf-8") as stream:
            env=dict(os.environ)
            if args.proxy:
                env.update(HTTP_PROXY=args.proxy,HTTPS_PROXY=args.proxy)
            commands=[]
            if not vpython.is_file():
                commands.append([python,'-m','venv',str(venv)])
            if args.profile in ('preprocess','training'):
                commands.append([str(vpython),'-m','pip','install','torch==2.8.0','--index-url',torch_index])
            commands.append([str(vpython),'-m','pip','install','-r',record['requirements']])
            record['steps']=[]
            for number,command in enumerate(commands,1):
                began=time.monotonic()
                step=dict(command=command,started_at=datetime.now().isoformat())
                record['steps'].append(step)
                stream.write(json.dumps(step)+'\n');stream.flush()
                purpose=('创建隔离环境' if command[1:3]==['-m','venv'] else
                         '安装PyTorch' if any(arg.startswith('torch==') for arg in command) else '安装项目依赖')
                try:
                    code=run_logged(command,stream,f'[{number}/{len(commands)}] {purpose}',env=env).returncode
                    step.update(exit_code=code,elapsed_seconds=time.monotonic()-began)
                except KeyboardInterrupt:
                    step.update(error='interrupted',exit_code=130,elapsed_seconds=time.monotonic()-began)
                    code=130
                except Exception as error:
                    step.update(error=str(error),exit_code=1,elapsed_seconds=time.monotonic()-began)
                    report(f'步骤异常：{type(error).__name__}: {error}',stream)
                    code=1
                if code:
                    record['passed']=False
                    (logroot/(stamp+'_install_failed.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                    report('环境部署未完成，完整日志：'+str(log_path))
                    return code if code==130 else 1
        python=str(vpython)
    if args.check or args.install:
        # nnunetv2 has no __version__; query distributions and exercise the APIs
        # that this adapter actually calls instead of trusting pip's exit code.
        code=("import importlib,importlib.metadata as m,json,sys; names="+repr(modules)+";"
              "print(json.dumps({'python':sys.executable,'version':sys.version,"
              "'modules':{n:(importlib.import_module(n) and m.version('scikit-image' if n=='skimage' else n)) for n in names}}))")
        if args.profile=='preprocess':
            code += ";from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor;from nnunetv2.training.dataloading.nnunet_dataset import comp_blosc2_params;from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner;assert m.version('nnunetv2')=='2.8.1';assert hasattr(DefaultPreprocessor,'run_case_npy');print('preprocessing APIs OK; CUDA not requested')"
        if args.profile=='training':
            code += ";from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer;from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader;import torch;assert m.version('nnunetv2')=='2.8.1';assert hasattr(nnUNetTrainer,'get_training_transforms');assert hasattr(nnUNetDataLoader,'get_bbox');print('training APIs OK; CUDA build='+str(torch.version.cuda)+'; no GPU capacity claim')"
        with log_path.open('a',encoding='utf-8') as stream:
            result=run_logged([python,"-X","utf8","-c",code],stream,'检查依赖版本与实际接口',timeout=120)
        record.update(passed=result.returncode==0,actual=result.stdout.strip(),
                      error=result.stdout.strip() if result.returncode else '')
    if args.install and record.get('passed'):
        # Existing private paths stay in place; only this stage's interpreter
        # changes. The CLI will automatically use it in subsequent commands.
        saved[interpreter_key]=str(vpython)
        if args.profile=='training' and args.download_weights:
            sys.path.insert(0,str(ROOT/'src'))
            from chd_stunet.training.weights import ensure_weights
            paths=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8')).get(sys.platform,{})
            target=Path(args.weights or saved.get('weights_path') or paths.get('weights_path') or ROOT/'runtime/weights/stunet-small/small_ep4k.model').resolve()
            with log_path.open('a',encoding='utf-8') as stream:
                record['weights']=run_waiting(lambda:ensure_weights(target,proxy=args.proxy),
                                             stream,'校验或下载作者权重：'+str(target))
                report('作者权重校验完成',stream)
            saved['weights_path']=str(target)
        local.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
    output=logroot/(stamp+"_"+record["action"]+".json")
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False,indent=2))
    message=('环境计划已生成' if record['action']=='plan' else
             '环境部署与检查完成' if args.install and record['passed'] else
             '环境检查通过' if record['passed'] else '环境检查失败')
    with log_path.open('a',encoding='utf-8') as stream:report(message+'；记录：'+str(output),stream)
    return 0 if record.get("passed",True) else 1


if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
