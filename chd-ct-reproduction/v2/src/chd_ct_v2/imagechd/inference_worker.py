"""One case, one authorized visible GPU; the coordinator owns the report."""

import os
import sys
from pathlib import Path

import torch

from ..recovery import read_json
from .predict import infer_case


def main():
    context = read_json(Path(sys.argv[1]))
    row = next(row for row in context.pop("rows") if row["case_id"] == sys.argv[2])
    torch.set_num_threads(int(os.environ.get("CHD_CPU_THREADS", "2")))
    infer_case(**context, row=row)


if __name__ == "__main__":
    main()
