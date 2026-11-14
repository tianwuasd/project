from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from time import perf_counter

from update import code_guard


if __name__ == "__main__":
    started = perf_counter()
    business = bool(sys.argv[1:]) and sys.argv[1] in ("prepare", "train", "predict", "evaluate", "test", "bind-inputs")
    if business:
        print(json.dumps({"event": "command_starting", "command": sys.argv[1], "pid": os.getpid(),
                          "time_utc": datetime.now(timezone.utc).isoformat(),
                          "code": str(Path(__file__).resolve().parent), "python": sys.executable,
                          "message": "正在取得代码读取锁并加载命令入口"}, ensure_ascii=False), flush=True)
    with code_guard(__file__) as repo:
        if sys.argv[1:] == ["--code-info"]:
            sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
            scope = Path(__file__).resolve().parent.relative_to(repo).as_posix()
            status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain=v1", "--", scope], check=True, stdout=subprocess.PIPE).stdout
            print(json.dumps({"sha": sha, "modified": bool(status), "code": str(Path(__file__).resolve().parent), "python": sys.executable}, ensure_ascii=False))
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
            if business:
                print(json.dumps({"event": "code_ready", "command": sys.argv[1],
                                  "time_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                                  "seconds": perf_counter() - started}, ensure_ascii=False), flush=True)
            from chd_segmentation.cli import main
            main()
