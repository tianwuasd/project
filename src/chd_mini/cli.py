import argparse
import json

from .contracts import load_settings, verify_engine


def main(argv=None):
    parser = argparse.ArgumentParser(description="助手3独立小样本完整流程与计时")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="只读检查配置和生产依赖")
    check.add_argument("--config", required=True)
    execute = commands.add_parser("run", help="运行已批准的1–2例三模型技术流程")
    execute.add_argument("--config", required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--approval", required=True)
    inspect = commands.add_parser("inspect", help="只读检查三模型编排配置")
    inspect.add_argument("--config", required=True)
    review = commands.add_parser("review", help="共享阶段3：生成明确预测路线或执行零额外训练验收")
    review.add_argument("--handoff", required=True)
    review.add_argument("--execute", action="store_true")
    attempt = commands.add_parser("attempt", help="生产训练输入构建前登记或结算累计尝试")
    attempt.add_argument("--ledger", required=True)
    attempt.add_argument("--stage", required=True)
    attempt.add_argument("--outcome", choices=("successful", "skipped", "failed", "interrupted"))
    attempt.add_argument("--number", type=int)
    deliver = commands.add_parser("deliver", help="按私有规格生成源码包和新增/复用清单")
    deliver.add_argument("--spec", required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", required=True)
    worker.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        settings = load_settings(args.config)
        result = {"settings": settings, "engine": verify_engine(settings["engine_root"], settings["engine_commit"])}
    elif args.command == "inspect":
        from .model_routes import load_acceptance, build_steps
        settings = load_acceptance(args.config)
        result = {"settings": settings, "steps": build_steps(settings),
                  "engine": verify_engine(settings["engine_root"], settings["engine_commit"])}
    elif args.command == "review":
        from .review import build_review, execute_review
        from .contracts import read_json
        result = execute_review(args.handoff) if args.execute else build_review(read_json(args.handoff))
    elif args.command == "attempt":
        from .attempts import AttemptLedger
        ledger = AttemptLedger(args.ledger)
        if args.outcome is None:
            if args.number is not None:
                raise ValueError("领取尝试时不能指定序号")
            result = {"number": ledger.claim(args.stage)}
        else:
            result = ledger.finish(args.stage, args.number, args.outcome)
    elif args.command == "run":
        raise ValueError("本轮训练由助手2统一执行；助手3使用review阶段3零额外训练入口")
    elif args.command == "deliver":
        from .delivery import from_spec
        result = from_spec(args.spec)
    else:
        from .worker import execute
        result = execute(args.request, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
