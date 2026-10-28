"""Heavy work runs separately from the dependency-light launcher."""

import argparse
import json
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def smoke(root, manifest, output, device, complete):
    import yaml

    from ..config import load_config
    from ..data import load_volume, normalize_ct, read_manifest, resize, save_prediction
    from ..diagnosis import evaluate_rules
    from ..features import extract_features
    from ..inference.pipeline import predict_stage, run_inference
    from ..labels import STAGES
    from ..training.engine import train_stage

    config = load_config(root / "configs/smoke.yaml")
    stages = list(STAGES) if complete else [s for s in STAGES if not s.startswith("init")]
    for number, stage in enumerate(stages):
        print(f"试跑训练 {number + 1}/{len(stages)}：{stage}", flush=True)
        train_stage(config, manifest, stage, output / "models", device=device, max_steps=1)
    rows = read_manifest(manifest)
    inference = next((r for r in rows if r["split"] == "test"), next(r for r in rows if r["split"] == "val"))
    print("试跑推理与输出保存……", flush=True)
    if complete:
        seg, initial, volume, _ = run_inference(
            inference["image"], [output / "models"], output / "prediction", device=device, allow_smoke=True
        )
        features = extract_features(seg, volume.spacing, initial=initial, affine=volume.affine)
        save_json(output / "prediction/features.json", features)
        rules = yaml.safe_load((root / "configs/rules.yaml").read_text(encoding="utf-8"))
        save_json(output / "prediction/diagnosis.json", evaluate_rules(features, rules))
    else:
        volume = load_volume(inference["image"])
        pred, _ = predict_stage(
            normalize_ct(volume.data, config["window"]), "all128", [output / "models"], device
        )
        save_prediction(
            resize(pred, volume.data.shape, labels=True), volume, output / "prediction/baseline.nii.gz"
        )
    result = {
        "status": "passed" if complete else "partial",
        "trained_stages": stages,
        "inference": "seven_network_pipeline" if complete else "all128_baseline_only",
        "inference_split": inference["split"],
        "device": device,
        "clinical_validation": False,
        "message": "全部流程试跑通过"
        if complete
        else "基础训练和单模型推理通过；缺 initial_label，未运行 init 两阶段及完整融合诊断。",
    }
    save_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "synthetic", "selected"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from .datasets import prepare_sample, validate_dataset

    if args.command == "validate":
        save_json(output / "data_validation.json", validate_dataset(args.manifest))
    elif args.command == "synthetic":
        from ..synthetic import make_demo_dataset

        manifest = make_demo_dataset(output / "data")
        smoke(root, manifest, output, args.device, True)
    else:
        # Validate the entire dataset in the earlier step, not only its sampled rows.
        info = json.loads((output.parent / "data_validation.json").read_text(encoding="utf-8"))
        sample = prepare_sample(args.manifest, output / "sample")
        save_json(output / "sample_info.json", sample)
        smoke(root, sample["manifest"], output, args.device, info["initial_complete"])


if __name__ == "__main__":
    main()
