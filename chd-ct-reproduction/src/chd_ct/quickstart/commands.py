"""Dependency-light task arguments shared by desktop and server launchers."""

from ..paths import config_path
from .diagnosis import DIAGNOSIS_TASKS, add_arguments, build_diagnosis_command

TASKS = ("environment", "preprocess", "train", "predict", "evaluate", "demo", "test") + DIAGNOSIS_TASKS


def add_task_arguments(parser):
    parser.add_argument("--task", choices=TASKS)
    add_arguments(parser)
    parser.add_argument("--dataset", help="已解压 ImageCHD 目录；预测预处理可选单个 NIfTI")
    parser.add_argument("--prepared", help="预处理输出目录 / 训练预测输入目录")
    parser.add_argument("--models", help="完整六阶段模型目录或 models.json")
    parser.add_argument("--predictions", help="独立预测结果目录，供 evaluate 使用")
    parser.add_argument("--config", help="六阶段配置，默认 configs/chd.yaml")
    parser.add_argument("--mode", choices=["check", "smoke", "preflight", "train"], default="smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--size", type=int, default=0, help="预处理：0 保留重采样网格；正数生成缩小缓存")
    parser.add_argument("--spacing-excel", help="预处理影像间距表；默认自动读取数据目录内的影像信息 Excel")
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1, 1, 1),
        metavar=("X", "Y", "Z"),
        help="预处理目标间距，默认 1 1 1 mm",
    )
    parser.add_argument("--no-resample", action="store_true", help="预处理保留旧行为，不按 Excel 重采样")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-file")
    parser.add_argument("--for-prediction", action="store_true")
    parser.add_argument("--resume", action="store_true", help="预处理/训练/预测/测试在原输出目录续跑")
    parser.add_argument("--case-id")
    parser.add_argument("--split", choices=["auto", "cv", "train", "val", "test"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")


def choose_task(args, interactive):
    if args.task:
        return args.task
    if not interactive:
        raise ValueError("非交互运行请指定 --task environment/preprocess/train/predict/evaluate/demo/test")
    print(
        "请选择一个独立功能：0 环境检查；1 预处理；2 训练/短测；3 预测；4 评估；5 合成分割测试；6 测试集预测和评估；7 导入诊断标签；8 提取诊断特征；9 训练诊断树；10 诊断与解释；11 诊断评估；12 合成诊断测试"
    )
    value = input("功能 [0]：").strip() or "0"
    if value not in {str(i) for i in range(len(TASKS))}:
        raise ValueError("请选择 0 至 12")
    return TASKS[int(value)]


def build_command(args, root, python, output, device, mode=None):
    task = args.task
    if task in DIAGNOSIS_TASKS:
        return build_diagnosis_command(args, python, output)
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
        if getattr(args, "spacing_excel", None):
            command += ["--spacing-excel", args.spacing_excel]
        if getattr(args, "target_spacing", None) is not None:
            command += ["--target-spacing", *map(str, args.target_spacing)]
        if getattr(args, "no_resample", False):
            command += ["--no-resample"]
        if getattr(args, "resume", False):
            command += ["--resume"]
        if getattr(args, "reuse_failure_report", None):
            command += ["--reuse-failure-report", args.reuse_failure_report]
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
        if getattr(args, "gpu_ids", None):
            command += ["--gpu-ids", ",".join(args.gpu_ids)]
        if getattr(args, "gpu_limit", None) is not None:
            command += ["--gpu-limit", str(args.gpu_limit)]
    elif task in {"predict", "test"}:
        if task == "test" and args.case_id:
            raise ValueError("test 评估完整 split；单病例请使用 predict")
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
        command += ["--predictions", args.predictions, "--split", args.split or "auto"]
    else:
        raise ValueError("未知任务")
    if getattr(args, "resume", False) and task in {"train", "predict", "test"}:
        command += ["--resume"]
    return command
