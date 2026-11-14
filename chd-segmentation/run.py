from pathlib import Path
import json
import subprocess
import sys

from update import code_guard


if __name__ == "__main__":
    with code_guard(__file__) as repo:
        if sys.argv[1:] == ["--code-info"]:
            sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
            status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain=v1"], check=True, stdout=subprocess.PIPE).stdout
            print(json.dumps({"sha": sha, "modified": bool(status), "code": str(Path(__file__).resolve().parent), "python": sys.executable}, ensure_ascii=False))
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
            from chd_segmentation.cli import main
            main()
