import argparse
import json
from pathlib import Path

from . import __version__
from .artifacts.store import read_json
from .config import load_config
from .observability.events import event, log_directory, stage


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
    train.add_argument("--device", choices=("cpu", "cuda"))
    train.add_argument("--gpus", type=int, choices=range(1, 5), help="显卡数量")
    train.add_argument("--gpu-ids", help="物理编号或 UUID，逗号分隔，例如 0,2")
    train.add_argument("--batch", type=int, help="全局 batch")
    train.add_argument("--batch-layout", help="手动每卡 batch，例如 2,1,1")
    train.add_argument("--epochs", type=int)
    train.add_argument("--workers", help="全组供数进程上限：auto 或整数")
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
    test = commands.add_parser("test", help="使用训练冻结计划运行内部或外部测试")
    test.add_argument("--config", required=True)
    test.add_argument("--set", dest="test_set", required=True, choices=("internal", "external"))
    test.add_argument("--group", choices=("target_primary", "target_secondary"))
    test.add_argument("--case", action="append")
    status = commands.add_parser("status")
    status.add_argument("--run", required=True)
    bind = commands.add_parser("bind-inputs", help="将私有迁移包绑定到本机已有数据，不复制原始数据")
    bind.add_argument("--directory", required=True)
    bind.add_argument("--root", required=True)
    bind.add_argument("--chd-root")
    bind.add_argument("--image-root")
    bind.add_argument("--pretrained", help="已有权重的实际位置")
    for command in (prepare, train, predict, evaluate, test):
        command.add_argument("--split", help="旧工程或手动选择的划分文件")
        command.add_argument("--split-format", choices=("auto", "imagechd", "imagechd_roles", "chd1282_cohort", "folds", "roles"))
        command.add_argument("--split-id-field", help="清单中的精确映射字段，默认 case_id")
        command.add_argument("--fold", type=int)
    args = parser.parse_args(argv)
    overrides = {}
    if args.command not in ("status", "bind-inputs"):
        split = {k: v for k, v in {"path": args.split, "format": args.split_format,
                                   "id_field": args.split_id_field}.items() if v is not None}
        if split:
            overrides["split"] = split
        if args.fold is not None:
            overrides["fold"] = args.fold
    if args.command == "train":
        resources, training = {}, {}
        for key, value in (("device", args.device), ("gpu_count", args.gpus),
                           ("workers", "auto" if args.workers == "auto" else int(args.workers) if args.workers else None)):
            if value is not None:
                resources[key] = value
        if args.gpu_ids:
            resources["gpu_ids"] = args.gpu_ids.split(",")
            resources["gpu_count"] = len(resources["gpu_ids"])
            if args.gpus is not None and args.gpus != resources["gpu_count"]:
                parser.error("--gpus 与 --gpu-ids 数量不同")
        if args.batch_layout:
            resources["batch_layout"] = [int(value) for value in args.batch_layout.split(",")]
        for key, value in (("batch", args.batch), ("epochs", args.epochs)):
            if value is not None:
                training[key] = value
        overrides.update(resources=resources, training=training)
    if args.command not in ("status", "bind-inputs"):
        config = load_config(args.config, overrides)
        event(log_directory(config), "command_ready", command=args.command, dataset=config["dataset"],
              fold=config["fold"], config=str(Path(args.config).resolve()),
              run_dir=config["run_dir"], work_root=config["work_root"])
    if args.command == "bind-inputs":
        from .datasets.transfer import bind_inputs
        roots = {name: value for name, value in (("chd1282", args.chd_root), ("imagechd", args.image_root)) if value is not None}
        result = bind_inputs(args.directory, args.root, roots, args.pretrained)
    elif args.command == "status":
        result = read_json(Path(args.run) / "summary.json")
    elif args.command == "prepare":
        from time import perf_counter
        with stage(config, "load_prepare_modules"):
            from .preprocessing.prepare import prepare
        existed = Path(config["plan"]).is_file()
        started = perf_counter()
        result = prepare(config, args.output)
        result = {"identity": result["identity"], "scope": result["scope"], "cases": len(result["manifest"]["cases"]),
                  "plan": config["plan"], "planning_mode": config["planning"]["mode"],
                  "plan_reused": existed, "fit_ids": result["plan"].get("fit_ids"),
                  "seconds": perf_counter() - started}
    elif args.command == "train":
        with stage(config, "load_training_modules"):
            from .training.engine import train
        result = train(config, args.prepared, args.resume, args.stop_after)
    elif args.command == "predict":
        with stage(config, "load_prediction_modules"):
            from .prediction.infer import predict
        result = predict(config, args.output, args.inputs, args.case)
    elif args.command == "test":
        with stage(config, "load_test_modules"):
            from .prediction.testing import run_test
        result = run_test(config, args.test_set, args.group, args.case)
    else:
        with stage(config, "load_evaluation_modules"):
            from .prediction.metrics import evaluate
            from .preprocessing.prepare import load_prepared
        result = evaluate(args.predictions, load_prepared(args.prepared), args.output, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
