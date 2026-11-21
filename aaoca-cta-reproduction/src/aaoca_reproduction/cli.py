import argparse
import json

from .audit import check_training, load_frozen, save_audit
from .inputs import bind_inputs, export_inputs
from .prepare import prepare_case
from .records import load_config


def main():
    parser = argparse.ArgumentParser(description="AAOCA论文复现：输入核查与明确病例定位裁剪")
    parser.add_argument("action", choices=("audit", "prepare", "check-training", "export-inputs", "bind-inputs"))
    parser.add_argument("--config", default="project.local.json")
    parser.add_argument("--case")
    parser.add_argument("--inputs")
    parser.add_argument("--server-root")
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
    if args.action == "prepare":
        matches = [row for row in manifest["cases"] if row["case_id"] == args.case]
        if len(matches) != 1:
            parser.error("--case需要明确指定冻结清单中的一例")
        result = prepare_case(matches[0], config, config["project_root"] / "work" / "prepared" / args.case)
    elif args.action == "check-training":
        result = check_training(config, manifest)
    elif args.action == "export-inputs":
        result = export_inputs(config, manifest, args.server_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
