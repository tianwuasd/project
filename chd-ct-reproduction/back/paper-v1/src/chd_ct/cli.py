"""Command line interface: see `chd-ct --help` and README for complete workflows."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .config import load_config
from .data import aligned_label, load_volume, read_manifest
from .labels import STAGES


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parser():
    p = argparse.ArgumentParser(description="Independent CHD CT research reproduction")
    commands = p.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="Create synthetic engineering-test phantoms")
    demo.add_argument("--output", required=True)
    validate = commands.add_parser("validate-data", help="Validate patient splits and NIfTI label grids")
    validate.add_argument("--manifest", required=True)
    train = commands.add_parser("train", help="Train one stage, or all stages in dependency order")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--max-steps", type=int)
    infer = commands.add_parser(
        "infer", help="Seven-network fusion; recurrent checkpoint bundles its encoder"
    )
    infer.add_argument("--image", required=True)
    infer.add_argument("--models", nargs="+", required=True, help="One or three complete model directories")
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--allow-smoke", action="store_true")
    infer.add_argument("--rules", help="Optional rule YAML for research candidate findings")
    features = commands.add_parser("features", help="Extract physical anatomy features from a label volume")
    features.add_argument("--segmentation", required=True)
    features.add_argument("--initial")
    features.add_argument("--spine")
    features.add_argument("--contact-mm", type=float, default=1.5)
    features.add_argument("--output", required=True)
    diagnose = commands.add_parser("diagnose", help="Evaluate explicit three-valued candidate rules")
    diagnose.add_argument("--features", required=True)
    diagnose.add_argument("--rules", required=True)
    diagnose.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate-seg", help="Compute per-class Dice on matching grids")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--output", required=True)
    labels = commands.add_parser(
        "evaluate-diagnosis", help="Evaluate JSON matrices with -1 for unknown predictions"
    )
    labels.add_argument("--truth", required=True)
    labels.add_argument("--prediction", required=True)
    labels.add_argument("--output", required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "make-demo":
            from .synthetic import make_demo_dataset

            print(make_demo_dataset(args.output))
        elif args.command == "validate-data":
            rows = read_manifest(args.manifest)
            for row in rows:
                v = load_volume(row["image"])
                aligned_label(row["label"], v)
                if row.get("initial_label"):
                    aligned_label(row["initial_label"], v, 7)
            print(
                json.dumps(
                    {
                        "cases": len(rows),
                        "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
                        "valid": True,
                    }
                )
            )
        elif args.command == "train":
            from .training.engine import train_stage

            config = load_config(args.config)
            for stage in STAGES if args.stage == "all" else [args.stage]:
                train_stage(config, args.manifest, stage, args.output, args.device, args.max_steps)
        elif args.command == "infer":
            from .features import extract_features
            from .inference.pipeline import run_inference

            seg, initial, volume, metadata = run_inference(
                args.image, args.models, args.output, args.device, args.allow_smoke
            )
            features = extract_features(seg, volume.spacing, initial=initial, affine=volume.affine)
            write_json(Path(args.output) / "features.json", features)
            if args.rules:
                from .diagnosis import evaluate_rules

                findings = evaluate_rules(features, load_yaml(args.rules))
                findings["inference_profile"] = metadata["profile"]
                write_json(Path(args.output) / "diagnosis.json", findings)
            print(f"Wrote research outputs to {args.output}")
        elif args.command == "features":
            from .features import extract_features

            v = load_volume(args.segmentation)
            initial = aligned_label(args.initial, v, 7) if args.initial else None
            spine = aligned_label(args.spine, v, 2) if args.spine else None
            write_json(
                args.output, extract_features(v.data, v.spacing, initial, spine, v.affine, args.contact_mm)
            )
        elif args.command == "diagnose":
            from .diagnosis import evaluate_rules

            features = json.loads(Path(args.features).read_text(encoding="utf-8"))
            write_json(args.output, evaluate_rules(features, load_yaml(args.rules)))
        elif args.command == "evaluate-seg":
            from .evaluation import segmentation_metrics

            v = load_volume(args.prediction)
            write_json(args.output, segmentation_metrics(v.data, aligned_label(args.target, v)))
        elif args.command == "evaluate-diagnosis":
            from .evaluation import multilabel_metrics

            truth = np.asarray(json.loads(Path(args.truth).read_text(encoding="utf-8")))
            prediction = np.asarray(json.loads(Path(args.prediction).read_text(encoding="utf-8")))
            write_json(args.output, multilabel_metrics(truth, prediction))
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
