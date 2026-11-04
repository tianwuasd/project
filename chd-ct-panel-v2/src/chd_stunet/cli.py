"""Local analysis CLI. Heavy scientific imports are deferred until analysis begins."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
import subprocess
from pathlib import Path
import sys

from .config import ROOT,read_config,resolve_path,prepare_paths,training_paths,default_prepared


def positive(text):
    value=int(text)
    if value<1: raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative(text):
    value=int(text)
    if value<0:raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description="CHD: data analysis and selectable preprocessing")
    sub=parser.add_subparsers(dest="command",required=True)
    analyze=sub.add_parser("analyze",help="analyze native data; no repairs or cohort exclusion")
    analyze.add_argument("--input")
    analyze.add_argument("--descriptor")
    analyze.add_argument("--output")
    analyze.add_argument("--workers",type=positive)
    analyze.add_argument("--max-workers",type=positive)
    analyze.add_argument("--readers",type=positive)
    analyze.add_argument("--components",choices=("selected","all","none"))
    analyze.add_argument("--limit",type=positive)
    analyze.add_argument("--cases",nargs="+")
    analyze.add_argument("--resume",action="store_true")
    analyze.add_argument("--tag",default="analysis")
    prepare=sub.add_parser('prepare',help='select a versioned preprocessing method; no training')
    prepare.add_argument('--method',default='stunet-s')
    prepare.add_argument('--list-methods',action='store_true')
    prepare.add_argument('--input')
    prepare.add_argument('--descriptor',help='optional previous descriptor; not required to bootstrap')
    prepare.add_argument('--output',help='explicit run directory; relative paths are anchored to the project')
    prepare.add_argument('--workers',type=positive,help='optional upper limit; CPU/RAM may lower it')
    prepare.add_argument('--readers',type=positive,help='optional read/decompression limit, separate from compute')
    prepare.add_argument('--fold',nargs='+',choices=['0','1','2','all'],default=['all'])
    prepare.add_argument('--cases',nargs='+',help='preprocess these cases; audit and train-only fitting remain full scope')
    prepare.add_argument('--cohort',choices=('all','main','auxiliary'),default='all',
                         help='limit outputs only; auxiliary requires stunet-s-partial; fit always uses main train')
    prepare.add_argument('--limit',type=positive,help='limit output cases only; NEVER fit on a shortened training subset')
    prepare.add_argument('--audit-only',action='store_true')
    prepare.add_argument('--resume',action='store_true')
    train=sub.add_parser('train',help='STU-Net-S CPU/single-CUDA fine-tuning with case-specific supervision')
    train.add_argument('--prepared',help='verified preprocessing run; defaults to saved stage path')
    train.add_argument('--output')
    train.add_argument('--fold',type=int,choices=(0,1,2),default=0)
    train.add_argument('--weights',help='pinned author small_ep4k.model; required only for a new run')
    train.add_argument('--device',default='cuda:0')
    train.add_argument('--workers',type=nonnegative,help='CPU supply upper limit; 0 is explicit serial mode')
    train.add_argument('--smoke',action='store_true',help='explicit 64-cube/B1 short profile, not a formal experiment')
    train.add_argument('--train-cases',nargs='+')
    train.add_argument('--val-cases',nargs='+')
    train.add_argument('--epochs',type=positive)
    train.add_argument('--train-steps',type=positive)
    train.add_argument('--val-steps',type=positive)
    train.add_argument('--batch-size',type=positive)
    train.add_argument('--patch-size',nargs=3,type=positive)
    train.add_argument('--seed',type=int)
    train.add_argument('--no-amp',dest='amp',action='store_false',default=None)
    train.add_argument('--without-auxiliary',dest='include_auxiliary',action='store_false',default=None)
    train.add_argument('--check-only',action='store_true',help='validate inputs/recipe without allocating a GPU')
    train.add_argument('--stop-after-steps',type=positive,help='pause safely at this total train-attempt count; omit on resume')
    train.add_argument('--resume',action='store_true')
    status=sub.add_parser("status",help="read an existing run status without scientific imports")
    status.add_argument("--output",required=True)
    sub.add_parser("check",help="show configured paths; does not install anything")
    args=parser.parse_args(argv)
    if args.command=='prepare' and args.list_methods:
        from .preprocessing.registry import list_methods
        print(json.dumps(list_methods(),ensure_ascii=False,indent=2));return 0
    if args.command=="status":
        print((resolve_path(args.output)/"status.json").read_text(encoding="utf-8"));return 0
    public,local=read_config()
    if args.command=="check":
        print(json.dumps(dict(project_root=str(ROOT),local_config_present=(ROOT/"project.local.json").is_file(),
            raw_root=local.get("raw_root"),raw_exists=Path(local.get("raw_root","/__missing__")).is_dir(),
            preprocessing_paths=prepare_paths(local),
            implementation="analysis + selectable preprocessing + STU-Net-S CPU/single-CUDA training; no whole-volume prediction"),ensure_ascii=False,indent=2));return 0
    selected_python=local.get('training_python') if args.command=='train' else local.get('preprocess_python') if args.command=='prepare' else None
    selected_python=selected_python or local.get('analysis_python')
    if not selected_python:
        venv=ROOT/('.venv_training' if args.command=='train' else '.venv_preprocess' if args.command=='prepare' else '.venv_analysis')
        candidate=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if candidate.is_file():selected_python=str(candidate)
    # A Linux venv Python may be a symlink to the system binary. Resolving
    # both paths erases the environment selection even though sys.prefix and
    # installed packages differ. Preserve the executable's invocation path.
    if selected_python:selected_python=os.path.abspath(os.path.expanduser(selected_python))
    if selected_python and os.path.normcase(selected_python)!=os.path.normcase(os.path.abspath(sys.executable)):
        if os.environ.get("CHD_ANALYSIS_REEXEC")=="1":
            raise RuntimeError("Selected interpreter did not match the configured environment")
        if not Path(selected_python).is_file():
            raise FileNotFoundError("Configured interpreter is unavailable: "+selected_python)
        env=dict(os.environ,CHD_ANALYSIS_REEXEC="1",PYTHONUTF8="1")
        arguments=list(argv) if argv is not None else sys.argv[1:]
        print('切换执行环境：'+selected_python,flush=True)
        return subprocess.call([selected_python,"-u",str(ROOT/"run.py"),*arguments],env=env)
    if args.command=='train':
        paths=training_paths(local)
        prepared=resolve_path(args.prepared) if args.prepared else default_prepared(paths)
        weights=args.weights or paths.get('weights_path')
        weights=resolve_path(weights) if weights else None
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        output=resolve_path(args.output) if args.output else resolve_path(paths.get('training_output_root','outputs/training'))/f'fold_{args.fold}'/stamp
        from .training.engine import run_train
        overrides=dict(epochs=args.epochs,train_steps_per_epoch=args.train_steps,
            validation_steps_per_epoch=args.val_steps,batch_size=args.batch_size,patch_size=args.patch_size,
            amp=args.amp,include_auxiliary=args.include_auxiliary,seed=args.seed)
        result=run_train(prepared,output,fold=args.fold,weights=weights,device=args.device,smoke=args.smoke,
            train_cases=args.train_cases or (),val_cases=args.val_cases or (),overrides=overrides,
            workers=args.workers,resume=args.resume,stop_after_steps=args.stop_after_steps,check_only=args.check_only)
        print(json.dumps(dict(output=str(output),**result),ensure_ascii=False,indent=2))
        return 0
    if args.command=='prepare':
        paths=prepare_paths(local)
        raw=resolve_path(args.input or paths.get('raw_root'))
        descriptor=args.descriptor or paths.get('descriptor')
        descriptor=resolve_path(descriptor) if descriptor else None
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        from .preprocessing.registry import get_method
        method=get_method(args.method)
        output=resolve_path(args.output) if args.output else resolve_path(paths.get('preprocess_output_root','outputs/preprocessing'))/args.method/method['protocol']/stamp
        folds=(0,1,2) if 'all' in args.fold else tuple(sorted(set(map(int,args.fold))))
        # Runtime leaves limit native libraries before importing nnU-Net. The
        # parent is a supervisor, not an extra multi-threaded compute worker.
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key]='1'
        from threadpoolctl import threadpool_limits
        from .preprocessing.run import run_prepare
        with threadpool_limits(limits=1):
            result=run_prepare(raw,output,method_name=args.method,descriptor=descriptor,folds=folds,
                selected_cases=args.cases or (),limit=args.limit,workers=args.workers,
                readers=args.readers or public.get('readers',2),
                resume=args.resume,audit_only=args.audit_only,cohort=args.cohort)
        print(json.dumps(dict(output=str(output),**result),ensure_ascii=False,indent=2))
        if result.get('protocol') and not result['failed_cases'] and not args.audit_only:
            from .observability.runlog import write_json
            write_json(ROOT/'outputs/latest_preprocessing.json',dict(output=str(output),method=args.method))
        return 1 if result['failed_cases'] else 0
    raw=resolve_path(args.input or local.get("raw_root"))
    descriptor=args.descriptor or local.get("descriptor")
    descriptor=resolve_path(descriptor) if descriptor else None
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output=resolve_path(args.output) if args.output else resolve_path(local.get("output_root","outputs"))/args.tag/stamp
    from .analysis_run import run_analysis
    try:
        summary=run_analysis(raw,output,descriptor=descriptor,workers=args.workers,
            cap=args.max_workers or public["max_workers"],readers=args.readers or public["readers"],
            components=args.components or public["components"],
            component_cases=local.get("component_cases",[]),snapshot_cases=local.get("snapshot_cases",[]),
            limit=args.limit,selected_cases=args.cases or (),resume=args.resume)
        from .observability.runlog import write_json
        write_json(ROOT/"outputs/latest.json",dict(output=str(output),report=str(output/"report.md")))
        print(json.dumps(dict(report=str(output/"report.md"),analyzed=summary["analyzed_pairs"],
                              failed=summary["failed_pairs"]),ensure_ascii=False))
        return 0 if not summary["failed_pairs"] else 1
    except KeyboardInterrupt:
        print("用户中断；已完成逐例结果保留，使用相同配置和--resume继续。",file=sys.stderr);return 130
    except Exception as error:
        print(f"分析失败：{error}；检查输出目录的状态和日志。",file=sys.stderr)
        raise


if __name__=="__main__":
    raise SystemExit(main())
