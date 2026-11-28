import argparse
import json
from pathlib import Path
import sys

from .audit import check_training, load_frozen, read_ct_registry, save_audit
from .inputs import bind_inputs, export_inputs
from .prepare import prepare_case
from .records import load_config


def main():
    parser = argparse.ArgumentParser(description="AAOCA论文复现：输入核查与明确病例定位裁剪")
    parser.add_argument("action", choices=("audit", "prepare", "prepare-classification", "prepare-dataset", "check-training", "check-label-registry", "organize-two-tasks", "bind-dataset", "export-inputs", "bind-inputs",
                                          "export-confirmed-labels", "generate-run-configs", "reset-preprocessing", "train", "predict", "predict-best",
                                          "audit-sources", "build-cohort", "bind-cohort", "prepare-author-weights", "generate-experiments",
                                          "predict-ensemble", "finetune-ensemble", "audit-shortcuts"))
    parser.add_argument("--config", default="project.local.json")
    parser.add_argument("--case")
    parser.add_argument("--inputs")
    parser.add_argument("--server-root")
    parser.add_argument("--dependency-dir")
    parser.add_argument("--output")
    parser.add_argument("--diagnoses")
    parser.add_argument("--runtime-root")
    parser.add_argument("--data-root")
    parser.add_argument("--aaoca-root")
    parser.add_argument("--imagechd-root")
    parser.add_argument("--chd1282-root")
    parser.add_argument("--author-archive")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-output-root")
    args = parser.parse_args()
    if args.action in {"audit-sources", "build-cohort"} and args.dependency_dir:
        sys.path.append(str(Path(args.dependency_dir).resolve(strict=True)))
    if args.check and args.action not in {"bind-dataset", "prepare-dataset", "generate-run-configs", "reset-preprocessing", "predict-best",
                                         "build-cohort", "bind-cohort", "prepare-author-weights", "generate-experiments", "predict-ensemble"}:
        parser.error("--check仅适用于绑定、预处理计划、配置生成、清理计划和最佳模型预测计划")
    config = load_config(args.config)
    if any((args.runtime_root, args.data_root, args.aaoca_root, args.imagechd_root, args.chd1282_root)):
        if args.action not in {"bind-dataset", "bind-cohort"}:
            parser.error("数据及运行根覆盖只适用于bind-dataset，不修改源配置")
        if args.runtime_root:
            config["project_root"] = config["course_root"] = Path(args.runtime_root).resolve()
        for key in ("data_root", "aaoca_root", "imagechd_root", "chd1282_root"):
            if getattr(args, key):
                config[key] = Path(getattr(args, key)).resolve()
    if args.author_archive:
        config["author_archive"] = Path(args.author_archive).resolve(strict=True)
    if args.action == "audit":
        _, result = save_audit(config)
    elif args.action == "audit-sources":
        from .multisource import source_audit
        result = source_audit(config)
    elif args.action == "build-cohort":
        from .multisource import build_cohort
        if not args.diagnoses or not args.output:
            parser.error("build-cohort需要新CSV的--diagnoses及正式--output目录")
        result = build_cohort(config, args.diagnoses, args.output, check=args.check)
    elif args.action == "bind-cohort":
        from .multisource import bind_cohort
        if not args.inputs or not args.output or not config.get("chd1282_root"):
            parser.error("bind-cohort需要--inputs、--output和实际--chd1282-root")
        result = bind_cohort(config, args.inputs, args.output, check=args.check)
    elif args.action == "prepare-author-weights":
        from .experiments import prepare_author_weights
        result = prepare_author_weights(config, check=args.check)
    elif args.action == "generate-experiments":
        from .experiments import generate_experiments
        if not args.output:
            parser.error("generate-experiments需要--output指定独立配置目录")
        result = generate_experiments(config, args.output, device=args.device, check=args.check)
    elif args.action == "generate-run-configs":
        from .workflow import generate_run_configs
        if not args.output:
            parser.error("生成配置须用--output指定独立目录，原配置不改")
        result = generate_run_configs(config, args.config, args.output, gpus=args.gpu_count,
                    batch_size=args.batch_size, epochs=args.epochs, device=args.device, check=args.check)
    elif args.action == "bind-dataset":
        from .migration import bind_dataset
        if not args.inputs or not args.diagnoses or not args.output:
            parser.error("bind-dataset需要--inputs集中目录、--diagnoses工作簿及--output独立运行目录")
        result = bind_dataset(config, args.inputs, args.diagnoses, args.output,
                              Path(args.config).resolve().parent, check=args.check)
    elif args.action == "bind-inputs":
        if not args.inputs:
            parser.error("bind-inputs需要--inputs指定已迁移的相对路径清单")
        result = bind_inputs(config, args.inputs)
    else:
        manifest = load_frozen(config)
        result = None
    if args.action in ("prepare-classification", "prepare-dataset", "train", "predict", "predict-best", "finetune-ensemble", "predict-ensemble") and args.dependency_dir:
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
    elif args.action == "prepare-dataset":
        from .preprocessing_batch import prepare_dataset
        result = prepare_dataset(config, manifest, check=args.check)
    elif args.action == "check-training":
        result = check_training(config, manifest)
    elif args.action == "reset-preprocessing":
        from .reset_preprocessing import reset_preprocessing
        result = reset_preprocessing(config, manifest, confirm_root=args.confirm_output_root, check=args.check)
    elif args.action == "organize-two-tasks":
        from .datasets import organize_two_tasks
        if not args.inputs or not args.output:
            parser.error("organize-two-tasks需要--inputs指定正式诊断表、--output指定独立正式数据集目录")
        result = organize_two_tasks(config, manifest, args.inputs, args.output)
    elif args.action == "check-label-registry":
        from .labels import read_label_rows, summarize_registry_labels
        if not config.get("diagnoses"):
            parser.error("check-label-registry需要diagnoses指定实际英文诊断表")
        result = summarize_registry_labels(read_label_rows(config["diagnoses"], config.get("diagnosis_sheet")),
                                           read_ct_registry(config), manifest, aaoca_root=config["aaoca_root"])
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
    elif args.action == "predict-best":
        from .best_prediction import predict_best
        result = predict_best(config, manifest, output=args.output, device=args.device, check=args.check)
    elif args.action == "predict-ensemble":
        from .experiments import predict_ensemble
        result = predict_ensemble(config, manifest, split=args.split, device=args.device, check=args.check)
    elif args.action == "finetune-ensemble":
        from .experiments import finetune_ensemble
        result = finetune_ensemble(config, manifest, resume=args.resume)
    elif args.action == "audit-shortcuts":
        from .experiments import shortcut_audit
        result = shortcut_audit(config, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
