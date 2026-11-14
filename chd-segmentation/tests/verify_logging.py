import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="最多两例真实病例的实时阶段日志短测")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    check = TestCase()
    # 先独立检查 CLI，再由短测取得代码锁，保持 Windows 代码锁的正常约束。
    with subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "run.py"), "prepare",
                           "--config", args.spec, "--output", str(Path(args.work_root) / "unused.json")],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8") as child:
        first = json.loads(child.stdout.readline())
        flushed = child.poll() is None
        remainder, _ = child.communicate()
        check.assertNotEqual(child.returncode, 0)
        check.assertIn("配置缺少", remainder)
    check.assertEqual(first["event"], "command_starting")
    check.assertTrue(flushed)
    from smoke import run
    run(args.spec, args.work_root, args.report, check_logging=True,
        startup_receipt={"first_event": first["event"], "flushed_before_exit": flushed})
