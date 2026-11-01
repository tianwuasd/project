"""Private regression harness for inherited CT renderers (not a panel entry)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chd_ct_v2.server.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
