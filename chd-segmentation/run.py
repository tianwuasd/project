from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
from time import perf_counter
import runpy

from code_runtime import pin, info


if __name__ == "__main__":
    code = Path(__file__).resolve().parent
    selected = pin(code)
    if selected != code:
        sys.argv[0] = str(selected / 'run.py')
        sys.path.insert(0, str(selected))
        runpy.run_path(sys.argv[0], run_name='__main__')
        raise SystemExit(0)
    started = perf_counter()
    business = bool(sys.argv[1:]) and sys.argv[1] in ("prepare", "preprocess", "train", "predict", "evaluate", "test", "bind-inputs")
    if business:
        print(json.dumps({"event": "command_starting", "command": sys.argv[1], "pid": os.getpid(),
                          "time_utc": datetime.now(timezone.utc).isoformat(),
                          "code": str(Path(__file__).resolve().parent), "python": sys.executable,
                          "message": "正在从固定代码快照加载命令入口"}, ensure_ascii=False), flush=True)
    if sys.argv[1:] == ["--code-info"]:
        print(json.dumps(info(code), ensure_ascii=False))
    else:
        sys.path.insert(0, str(code / 'src'))
        if business:
            print(json.dumps({"event": "code_ready", "command": sys.argv[1],
                              "time_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                              "seconds": perf_counter() - started, **info(code)}, ensure_ascii=False), flush=True)
        from chd_segmentation.cli import main
        main()
