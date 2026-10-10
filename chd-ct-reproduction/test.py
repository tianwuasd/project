"""Standalone primary task; input/output responsibility is explicit."""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    os.environ["PYTHONPATH"] = str(root / "src")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from chd_ct.imagechd.test import main

    raise SystemExit(main())
