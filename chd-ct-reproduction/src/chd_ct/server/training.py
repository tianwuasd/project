"""Full-size preflight is isolated from final checkpoints and limited to one epoch."""

import argparse
from pathlib import Path

import yaml

from ..config import load_config
from ..labels import STAGES
from ..training.engine import train_stage


def run(config_path, manifest, output, preflight=False):
    config = load_config(config_path)
    if config["profile"] == "smoke":
        raise ValueError("服务器正式训练/预检不能使用 smoke 配置。")
    if preflight:
        for spec in config["stages"].values():
            spec["epochs"] = 1
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "effective-config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for stage in STAGES:
        train_stage(config, manifest, stage, output, device="cuda", max_steps=1 if preflight else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.config, args.manifest, args.output, args.preflight)


if __name__ == "__main__":
    main()
