"""Internal stage worker; never preprocesses data or publishes a full collection."""

import argparse
import json
from pathlib import Path

from .common import read_prepared, write_json
from .config import STAGES
from .train import file_hash, train_stage, training_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--mode", choices=["smoke", "preflight", "train"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    config = json.loads((output / "effective-config.json").read_text(encoding="utf-8"))
    if args.check_environment:
        from ..quickstart.environment import probe

        # This worker is already isolated. Probe in-process so cancellation or
        # another stage failing cannot orphan a nested CUDA probe process.
        report = probe()
        write_json(output / (args.stage + ".environment.json"), report)
        if not report.get("ok") or not report.get("cuda"):
            raise RuntimeError("该阶段 GPU 环境检查未通过：" + str(report))
    directory, data = read_prepared(args.prepared)
    info = train_stage(
        directory,
        training_rows(data),
        args.stage,
        config,
        output,
        args.mode,
        file_hash(directory / "dataset.json"),
        args.resume,
    )
    write_json(output / (args.stage + ".result.json"), info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
