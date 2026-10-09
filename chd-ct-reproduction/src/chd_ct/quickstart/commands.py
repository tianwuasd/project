"""Dependency-light task arguments shared by desktop and server launchers."""

from ..paths import config_path

TASKS = ("environment", "preprocess", "train", "predict", "evaluate", "demo")


def add_task_arguments(parser):
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--dataset", help="已解压 ImageCHD 目录；预测预处理可选单个 NIfTI")
    parser.add_argument("--prepared", help="预处理输出目录 / 训练预测输入目录")
    parser.add_argument("--models", help="完整六阶段模型目录或 models.json")
    parser.add_argument("--predictions", help="独立预测结果目录，供 evaluate 使用")
    parser.add_argument("--config", help="六阶段配置，默认 configs/chd.yaml")
    parser.add_argument("--mode", choices=["check", "smoke", "preflight", "train"], default="smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--size", type=int, default=0, help="预处理：0 保留原始网格；正数生成缩小缓存")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-file")
    parser.add_argument("--for-prediction", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")


def choose_task(args, interactive):
    if args.task:
        return args.task
    if not interactive:
        raise ValueError("非交互运行请指定 --task environment/preprocess/train/predict/evaluate/demo")
    print("请选择一个独立功能：0 环境检查；1 预处理；2 训练/短测；3 预测；4 评估；5 合成全流程测试")
    value = input("功能 [0]：").strip() or "0"
    if value not in {str(i) for i in range(6)}:
        raise ValueError("请选择 0 至 5")
    return TASKS[int(value)]


def build_command(args, root, python, output, device, mode=None):
    task = args.task
    if task == "demo":
        return [python, "-m", "chd_ct.imagechd.demo", "--output", output, "--device", device]
    command = [python, "-m", "chd_ct.imagechd." + task]
    if task == "preprocess":
        if not args.dataset:
            raise ValueError("预处理需要 --dataset")
        command += [
            "--input",
            args.dataset,
            "--output",
            args.prepared or output / "prepared",
            "--size",
            str(args.size),
            "--seed",
            str(args.seed),
        ]
        if args.for_prediction:
            command += ["--for-prediction"]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        if args.split_file:
            command += ["--split-file", args.split_file]
        return command
    if not args.prepared:
        raise ValueError("需要 --prepared；请先单独运行预处理")
    command += ["--prepared", args.prepared, "--output", output]
    if task == "train":
        command += [
            "--config",
            args.config or config_path("chd.yaml"),
            "--mode",
            mode or args.mode,
            "--device",
            device,
        ]
    elif task == "predict":
        if not args.models:
            raise ValueError("预测需要 --models 完整模型目录")
        command += ["--models", args.models, "--device", device]
        for flag, value in (("--case-id", args.case_id), ("--split", args.split)):
            if value:
                command += [flag, value]
        if args.allow_smoke:
            command += ["--allow-smoke"]
    elif task == "evaluate":
        if not args.predictions:
            raise ValueError("评估需要 --predictions")
        command += ["--predictions", args.predictions, "--split", args.split or "test"]
    else:
        raise ValueError("未知任务")
    return command
