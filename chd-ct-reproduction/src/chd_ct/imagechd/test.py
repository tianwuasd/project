"""Independent held-out prediction plus evaluation; no training or preprocessing."""

import argparse
from pathlib import Path

from .common import read_prepared, write_json
from .evaluate import evaluate
from .predict import predict


def run_test(prepared, models, output, device="cpu", split="test", allow_smoke=False):
    _, data = read_prepared(prepared)
    if data.get("for_prediction") or not any(r["split"] == split for r in data["cases"]):
        raise ValueError("测试需要带真值且含所选 split 的缓存；无标签数据请使用 predict")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "task": "test", "split": split, "models": str(models)}
    try:
        predict(prepared, models, output / "predictions", device=device, split=split, allow_smoke=allow_smoke)
        metrics = evaluate(prepared, output / "predictions", output / "evaluation", split=split)
        report.update(
            status="passed",
            cases_evaluated=metrics["cases_evaluated"],
            predictions=str(output / "predictions"),
            evaluation=str(output / "evaluation/evaluation.json"),
        )
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "test-report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args(argv)
    run_test(args.prepared, args.models, args.output, args.device, args.split, args.allow_smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
