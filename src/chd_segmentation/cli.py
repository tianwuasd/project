import argparse
import json
from pathlib import Path

from . import __version__
from .artifacts.store import read_json
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="先心分割处理")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--prepared", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after", type=int)
    predict = commands.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--inputs")
    predict.add_argument("--case", action="append")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--prepared", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    status = commands.add_parser("status")
    status.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        result = read_json(Path(args.run) / "summary.json")
    elif args.command == "prepare":
        from .preprocessing.prepare import prepare
        result = prepare(load_config(args.config), args.output)
        result = {"identity": result["identity"], "scope": result["scope"], "cases": len(result["manifest"]["cases"])}
    elif args.command == "train":
        from .training.engine import train
        result = train(load_config(args.config), args.prepared, args.resume, args.stop_after)
    elif args.command == "predict":
        from .prediction.infer import predict
        result = predict(load_config(args.config), args.output, args.inputs, args.case)
    else:
        from .prediction.metrics import evaluate
        from .preprocessing.prepare import load_prepared
        result = evaluate(args.predictions, load_prepared(args.prepared), args.output, load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
