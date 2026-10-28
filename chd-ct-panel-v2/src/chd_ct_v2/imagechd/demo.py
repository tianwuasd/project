"""Explicit synthetic full workflow for software testing only."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from ..paths import config_path
from .common import write_json
from .evaluate import evaluate
from .predict import predict
from .preprocess import prepare
from .train import train


def run(output, device="cpu"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "synthetic"
    raw.mkdir()
    target = np.zeros((18, 16, 16), np.uint8)
    for label in range(1, 8):
        target[2 * label : 2 * label + 2, 3:13, 3:13] = label
    report = {"status": "running", "synthetic": True, "clinical_validation": False}
    try:
        for i in range(4):
            image = np.random.default_rng(i).normal(100, 20, target.shape).astype(np.float32) + target * 10
            nib.save(nib.Nifti1Image(image, np.eye(4)), raw / f"ct_{i}_image.nii.gz")
            nib.save(nib.Nifti1Image(target, np.eye(4)), raw / f"ct_{i}_label.nii.gz")
        prepare(raw, output / "prepared", size=16)
        train(
            output / "prepared",
            output / "models",
            config_path("chd-smoke.yaml"),
            "smoke",
            device,
        )
        predict(
            output / "prepared",
            output / "models",
            output / "predictions",
            device,
            split="test",
            allow_smoke=True,
        )
        evaluate(output / "prepared", output / "predictions", output / "evaluation")
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHD 六阶段合成测试；不代表医学精度")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    run(args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
