"""Independent diagnosis tasks; no segmentation is started implicitly."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立诊断：标签、特征、训练、预测解释、评估、合成短测")
    subs = parser.add_subparsers(dest="task", required=True)
    for task in ("labels", "features", "train", "predict", "evaluate", "demo"):
        p = subs.add_parser(task)
        p.add_argument("--output", required=True, help="新的输出目录，已有目录不会覆盖")
        if task == "labels":
            p.add_argument("--input", required=True)
            p.add_argument("--blank-policy", choices=["negative", "unknown"], default="negative")
        if task == "features":
            p.add_argument("--prepared", required=True)
            p.add_argument("--predictions", required=True)
        if task in {"train", "predict", "evaluate"}:
            p.add_argument("--features", required=True)
        if task in {"train", "evaluate"}:
            p.add_argument("--labels", required=True)
        if task in {"train", "predict"}:
            p.add_argument("--allow-smoke", action="store_true")
        if task == "train":
            p.add_argument("--max-depth", type=int, default=4)
            p.add_argument("--min-leaf", type=int, default=2)
            p.add_argument("--seed", type=int, default=42)
        if task == "predict":
            p.add_argument("--classifier")
            p.add_argument("--method", choices=["tree", "rules", "both"], default="both")
            p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
            p.add_argument("--evidence", help="经复核的临床解剖证据 JSON，供独立规则使用")
            p.add_argument("--rules", help="候选规则 YAML")
        if task == "evaluate":
            p.add_argument("--predictions", required=True)
            p.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = vars(parser.parse_args(argv))
    task = args.pop("task")
    try:
        if task == "labels":
            from .labels import import_labels

            args["source"] = args.pop("input")
            result = import_labels(**args)
        elif task == "features":
            from .features import extract

            result = extract(**args)
        elif task == "train":
            from .classifier import train_classifier

            result = train_classifier(**args)
        elif task == "predict":
            from .predict import diagnose

            result = diagnose(**args)
        elif task == "evaluate":
            from .evaluate import evaluate_diagnosis

            result = evaluate_diagnosis(**args)
        else:
            from .demo import run_demo

            result = run_demo(**args)
        if result.get("status") == "not_trainable":
            print("没有疾病满足训练条件；查看 trainability.json。", file=sys.stderr)
            return 2
        print(f"诊断步骤 {task} 已完成：{args['output']}")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print("诊断步骤未通过：" + str(error), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
