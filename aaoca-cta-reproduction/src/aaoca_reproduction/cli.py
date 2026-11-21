import argparse
import json
from pathlib import Path
import sys

from .audit import check_training, load_frozen, save_audit
from .inputs import bind_inputs, export_inputs
from .prepare import prepare_case
from .records import load_config


def main():
    parser = argparse.ArgumentParser(description="AAOCA论文复现：输入核查与明确病例定位裁剪")
    parser.add_argument("action", choices=("audit", "prepare", "prepare-classification", "check-training", "export-inputs", "bind-inputs",
                                          "export-confirmed-labels", "train", "predict"))
    parser.add_argument("--config", default="project.local.json")
    parser.add_argument("--case")
    parser.add_argument("--inputs")
    parser.add_argument("--server-root")
    parser.add_argument("--dependency-dir")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "audit":
        _, result = save_audit(config)
    elif args.action == "bind-inputs":
        if not args.inputs:
            parser.error("bind-inputs需要--inputs指定已迁移的相对路径清单")
        result = bind_inputs(config, args.inputs)
    else:
        manifest = load_frozen(config)
        result = None
    if args.action in ("prepare-classification", "train", "predict") and args.dependency_dir:
        sys.path.insert(0, str(Path(args.dependency_dir).resolve(strict=True)))
    if args.action in ("prepare", "prepare-classification"):
        matches = [row for row in manifest["cases"] if row["case_id"] == args.case]
        if len(matches) != 1:
            parser.error("--case需要明确指定冻结清单中的一例")
        if args.action == "prepare-classification":
            from .preprocessing import prepare_classification
            result = prepare_classification(matches[0], config, config["data_root"] / "aaoca-classification" / args.case)
        else:
            result = prepare_case(matches[0], config, config["project_root"] / "work" / "prepared" / args.case)
    elif args.action == "check-training":
        result = check_training(config, manifest)
    elif args.action == "export-inputs":
        result = export_inputs(config, manifest, args.server_root)
    elif args.action == "export-confirmed-labels":
        from .labels import export_confirmed_labels
        if not args.output:
            parser.error("需要--output指定private内的新确认标签CSV")
        result = export_confirmed_labels(config, manifest, args.output)
    elif args.action == "train":
        from .training import train
        result = train(config, manifest, resume=args.resume)
    elif args.action == "predict":
        from .prediction import predict
        result = predict(config, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
