"""Dependency-light entry point; works before installing this project as a package."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info < (3, 11):
        print("需要 Python 3.11 或以上。请安装后重新双击 start.bat。")
        return 1
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    from chd_ct.quickstart.wizard import main as wizard

    return wizard()


if __name__ == "__main__":
    raise SystemExit(main())
