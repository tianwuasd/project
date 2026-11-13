import json
from pathlib import Path
import subprocess
import sys

from update import deployment_lock, inspect_release, read_json


def main():
    code = Path(__file__).resolve().parent
    pointer = read_json(code / "current.json")
    with deployment_lock(pointer["runtime"], shared=True):
        # 获得共享代码锁后重读，避免启动过程跨越一次版本切换。
        pointer = read_json(code / "current.json")
        release, record = inspect_release(code, pointer)
        if sys.argv[1:] == ["--code-info"]:
            print(json.dumps({"sha": pointer["sha"], "code": str(release), "runtime": pointer["runtime"],
                              "python": pointer["python"], "version": record["version"]}, ensure_ascii=False, indent=2))
            return 0
        print("代码提交：%s；实际入口：%s" % (pointer["sha"], release / "run.py"), file=sys.stderr, flush=True)
        return subprocess.call([pointer["python"], "-B", str(release / "run.py"), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
