import argparse
import json

from .contracts import load_settings, verify_engine


def main(argv=None):
    parser = argparse.ArgumentParser(description="助手3独立小样本完整流程与计时")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="只读检查配置和生产依赖")
    check.add_argument("--config", required=True)
    execute = commands.add_parser("run", help="运行已批准的2/1/1小样本流程")
    execute.add_argument("--config", required=True)
    execute.add_argument("--run-id", required=True)
    deliver = commands.add_parser("deliver", help="按私有规格生成源码包和新增/复用清单")
    deliver.add_argument("--spec", required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", required=True)
    worker.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        settings = load_settings(args.config)
        result = {"settings": settings, "engine": verify_engine(settings["engine_root"], settings["engine_commit"])}
    elif args.command == "run":
        from .runner import run
        result = run(args.config, args.run_id)
    elif args.command == "deliver":
        from .delivery import from_spec
        result = from_spec(args.spec)
    else:
        from .worker import execute
        result = execute(args.request, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
