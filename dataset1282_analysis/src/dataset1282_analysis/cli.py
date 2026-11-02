"""Read-only local dataset analysis command line."""
import argparse
import json
import os
from pathlib import Path
from . import __version__
from .config import resolve_config
from .contracts import read_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='store_true')
    p.add_argument('--config',help='Task configuration; overrides project.local.json')
    commands=p.add_subparsers(dest='command')
    commands.add_parser('inventory')
    for name in ('analyze','resume','status','report'):
        sub=commands.add_parser(name)
        sub.add_argument('--run',required=name in ('resume','status','report'))
        if name=='analyze':
            sub.add_argument('--stage',choices=['metadata','headers','full'],default='full')
            sub.add_argument('--case',action='append',dest='cases')
    args=p.parse_args(argv)
    root=Path(os.environ.get('DATASET1282_CODE_ROOT',Path(__file__).resolve().parents[2]))
    home=Path(os.environ.get('DATASET1282_HOME',root))
    if args.version:
        print(json.dumps({'version':__version__,'code_root':str(root),'default_config':str(root/'configs/default.json'),
                          'docs':str(root/'docs'),'home':str(home)},ensure_ascii=False,indent=2));return 0
    config=resolve_config({},home/'project.local.json',root,args.config)
    if not args.command: p.print_help();return 0
    if args.command=='inventory':
        from .inputs import load_cases,inventory
        result=inventory(load_cases(config))
    elif args.command=='status':
        result=read_json(Path(config['work_root'])/'runs'/args.run/'status.json')
    elif args.command=='report':
        from .reporting import build_report
        result=build_report(Path(config['work_root'])/'runs'/args.run,home=home)
    else:
        from .engine.runner import run_analysis
        result=run_analysis(config,root,home,run_id=args.run,stage=getattr(args,'stage','full'),
                            case_ids=getattr(args,'cases',None),resume=args.command=='resume')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result.get('failed',0)==0 else 2
