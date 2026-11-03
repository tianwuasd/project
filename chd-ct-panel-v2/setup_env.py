"""Standalone environment plan/check/install for analysis and preprocessing.

Uses only the standard library. An explicit --install targets an isolated venv,
never the global interpreter. Preprocessing uses CPU Torch, without weights,
GPU allocation or training setup.
"""
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
MODULES=("numpy","scipy","nibabel","matplotlib","psutil","threadpoolctl")


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
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    if args.install:
        if venv.exists() and not (venv/"pyvenv.cfg").is_file():
            raise ValueError("Target exists and is not a recognized venv; refusing to overwrite")
        with (logroot/(stamp+".install.log")).open("w",encoding="utf-8") as stream:
            import os,time
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
            for command in commands:
                began=time.monotonic()
                step=dict(command=command,started_at=datetime.now().isoformat())
                record['steps'].append(step)
                stream.write(json.dumps(step)+'\n');stream.flush()
                try:
                    code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800).returncode
                    step.update(exit_code=code,elapsed_seconds=time.monotonic()-began)
                except Exception as error:
                    step.update(error=str(error),elapsed_seconds=time.monotonic()-began)
                    code=1
                if code:
                    record['passed']=False
                    (logroot/(stamp+'_install_failed.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                    print('环境部署失败，完整日志：'+str(logroot/(stamp+'.install.log')))
                    return 1
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
        result=subprocess.run([python,"-X","utf8","-c",code],capture_output=True,text=True,encoding="utf-8",timeout=120)
        record.update(passed=result.returncode==0,actual=result.stdout.strip(),error=result.stderr.strip())
    if args.install and record.get('passed'):
        # Existing private paths stay in place; only this stage's interpreter
        # changes. The CLI will automatically use it in subsequent commands.
        saved[interpreter_key]=str(vpython)
        if args.profile=='training' and args.download_weights:
            sys.path.insert(0,str(ROOT/'src'))
            from chd_stunet.training.weights import ensure_weights
            paths=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8')).get(sys.platform,{})
            target=Path(args.weights or saved.get('weights_path') or paths.get('weights_path') or ROOT/'runtime/weights/stunet-small/small_ep4k.model').resolve()
            record['weights']=ensure_weights(target,proxy=args.proxy)
            saved['weights_path']=str(target)
        local.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
    output=logroot/(stamp+"_"+record["action"]+".json")
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False,indent=2))
    print("环境记录："+str(output))
    return 0 if record.get("passed",True) else 1


if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
