"""Run a fully local CPU engineering demo without external images or weights."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, "-m", "chd_ct", *map(str, args)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/demo")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output exists; choose a new directory to preserve previous experiments")
    run("make-demo", "--output", output / "data")
    run("validate-data", "--manifest", output / "data/manifest.csv")
    run(
        "train",
        "--config",
        ROOT / "configs/smoke.yaml",
        "--manifest",
        output / "data/manifest.csv",
        "--output",
        output / "models",
        "--stage",
        "all",
        "--max-steps",
        "2",
    )
    run(
        "infer",
        "--image",
        output / "data/synthetic_03_image.nii.gz",
        "--models",
        output / "models",
        "--output",
        output / "prediction",
        "--allow-smoke",
        "--rules",
        ROOT / "configs/rules.yaml",
    )
    print(f"Engineering demo complete: {output}. Synthetic outputs have no clinical meaning.")


if __name__ == "__main__":
    main()
