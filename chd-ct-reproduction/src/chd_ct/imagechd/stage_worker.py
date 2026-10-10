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
    args = parser.parse_args(argv)
    output = Path(args.output)
    config = json.loads((output / "effective-config.json").read_text(encoding="utf-8"))
    directory, data = read_prepared(args.prepared)
    info = train_stage(
        directory,
        training_rows(data),
        args.stage,
        config,
        output,
        args.mode,
        file_hash(directory / "dataset.json"),
    )
    write_json(output / (args.stage + ".result.json"), info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
