from pathlib import Path
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aaoca_reproduction.cli import main

if __name__ == "__main__":
    if sys.argv[1:] == ["--code-info"]:
        root = Path(__file__).resolve().parent
        def git(*arguments):
            return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                                  capture_output=True, text=True).stdout.strip()
        repository = Path(git("rev-parse", "--show-toplevel"))
        relative = root.relative_to(repository).as_posix()
        print(json.dumps({"sha": git("rev-parse", "HEAD"), "code": str(root),
                          "modified": bool(git("status", "--porcelain", "--untracked-files=no", "--", relative)),
                          "python": sys.executable}, ensure_ascii=False))
    else:
        main()
