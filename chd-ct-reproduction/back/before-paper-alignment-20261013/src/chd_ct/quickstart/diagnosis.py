"""Dependency-light diagnosis arguments, settings and command builder."""

from pathlib import Path

DIAGNOSIS_TASKS = (
    "diagnosis-labels",
    "diagnosis-features",
    "diagnosis-train",
    "diagnose",
    "diagnosis-evaluate",
    "diagnosis-demo",
)
INPUTS = {
    "diagnosis-labels": (("diagnosis_source", "诊断表 XLSX / CSV", "diagnosis_source"),),
    "diagnosis-features": (
        ("prepared", "预处理目录", "prepared"),
        ("predictions", "分割预测目录（训练分类器需含 train/val）", "last_predictions"),
    ),
    "diagnosis-train": (
        ("features", "解剖特征目录", "last_features"),
        ("diagnosis_labels", "诊断标签目录", "last_diagnosis_labels"),
    ),
    "diagnose": (
        ("features", "解剖特征目录", "last_features"),
        ("classifier", "诊断分类器目录", "last_classifier"),
    ),
    "diagnosis-evaluate": (
        ("features", "解剖特征目录", "last_features"),
        ("diagnosis_labels", "诊断标签目录", "last_diagnosis_labels"),
        ("diagnoses", "诊断预测目录", "last_diagnoses"),
    ),
    "diagnosis-demo": (),
}
OUTPUT_KEYS = {
    "diagnosis-labels": "last_diagnosis_labels",
    "diagnosis-features": "last_features",
    "diagnosis-train": "last_classifier",
    "diagnose": "last_diagnoses",
}


def add_arguments(parser):
    for flag in (
        "diagnosis-source",
        "features",
        "diagnosis-labels",
        "classifier",
        "diagnoses",
        "evidence",
        "rules",
    ):
        parser.add_argument("--" + flag)
    parser.add_argument("--blank-policy", choices=["negative", "unknown"], default="negative")
    parser.add_argument("--method", choices=["tree", "rules", "both"], default="both")
    parser.add_argument("--diagnosis-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--tree-depth", type=int, default=4)
    parser.add_argument("--min-leaf", type=int, default=2)


def resolve_inputs(args, saved, choose, interactive, defaults=None):
    defaults = defaults or {}
    for name, label, key in INPUTS[args.task]:
        if name == "classifier" and args.method == "rules":
            continue
        value = choose(getattr(args, name, None), saved.get(key), defaults.get(name, ""), label, interactive)
        if not value:
            raise ValueError("需要 --" + name.replace("_", "-"))
        setattr(args, name, str(Path(value).expanduser().resolve()))
    args.device = "cpu"


def build_diagnosis_command(args, python, output):
    operation = "predict" if args.task == "diagnose" else args.task.removeprefix("diagnosis-")
    command = [python, "-m", "chd_ct.diagnosis.cli", operation, "--output", output]
    mapping = {"diagnosis_source": "--input", "diagnosis_labels": "--labels", "diagnoses": "--predictions"}
    for name, _, _ in INPUTS[args.task]:
        if name == "classifier" and args.method == "rules":
            continue
        value = getattr(args, name, None)
        if not value:
            raise ValueError("需要 --" + name.replace("_", "-"))
        command += [mapping.get(name, "--" + name.replace("_", "-")), value]
    if operation == "labels":
        command += ["--blank-policy", args.blank_policy]
    if operation == "train":
        command += [
            "--max-depth",
            str(args.tree_depth),
            "--min-leaf",
            str(args.min_leaf),
            "--seed",
            str(args.seed),
        ]
    if operation == "predict":
        command += ["--method", args.method]
        for name in ("evidence", "rules"):
            if getattr(args, name, None):
                command += ["--" + name, getattr(args, name)]
    if operation in {"predict", "evaluate"}:
        split = args.diagnosis_split
        if split == "all" and operation == "evaluate":
            raise ValueError("诊断评估须选择 train/val/test 中一个集合")
        command += ["--split", split]
    if operation in {"train", "predict"} and args.allow_smoke:
        command += ["--allow-smoke"]
    return command
