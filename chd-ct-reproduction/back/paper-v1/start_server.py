"""Linux entry; bootstrap needs only the standard library."""

import sys
from pathlib import Path

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from chd_ct.server.launcher import main

    raise SystemExit(main())
