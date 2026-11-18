import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from time import perf_counter
import traceback
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filelock import FileLock

from chd_mini.contracts import TOOL_ROOT, confined, git, load_settings, read_json, write_json
from chd_mini.runner import inventory, RunWorkspace


def verify(config_path):
    settings = load_settings(config_path)
    run_id = "runtime-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = confined(TOOL_ROOT / ".runs" / run_id, TOOL_ROOT)
    if root.exists():
        raise FileExistsError(root)
    work, logs = root / "work", root / "logs"
    checks = unittest.TestCase()
    started = perf_counter()
    work.mkdir(parents=True)
    logs.mkdir()
    record = {"state": "RUNNING", "samples": 0, "training_updates": 0, "scope": "维护及失败清理检查", "commands": []}
    with FileLock(str(TOOL_ROOT / ".runs" / ".tool.lock"), timeout=0):
        try:
            failure_id = run_id + "-failure"
            with checks.assertRaises(subprocess.CalledProcessError):
                with RunWorkspace(settings, failure_id) as task:
                    task.stage("invalid-stage")
            failure = read_json(TOOL_ROOT / ".runs" / failure_id / "logs" / "cleanup.json")
            checks.assertEqual(failure["state"], "completed")
            checks.assertTrue(failure["work_removed"])
            failure_summary = read_json(TOOL_ROOT / ".runs" / failure_id / "logs" / "summary.json")
            checks.assertEqual(failure_summary["stages"][-1]["state"], "failed")
            checks.assertGreater(failure_summary["stages"][-1]["wall_seconds"], 0)
            checkout = work / "tool-checkout"
            command = ["git", "clone", "--no-hardlinks", str(TOOL_ROOT), str(checkout)]
            record["commands"].append(command)
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
            pin = git(TOOL_ROOT, "rev-parse", "HEAD")
            before = git(checkout, "rev-parse", "HEAD")
            checks.assertEqual(before, pin)
            command = [settings["python"], "-B", str(checkout / "update.py"), "--check"]
            record["commands"].append(command)
            check_output = subprocess.check_output(command, text=True, encoding="utf-8")
            command = [settings["python"], "-B", str(checkout / "update.py"), "--remote", "origin",
                       "--ref", "codex/chd-segmentation-mini-workflow"]
            record["commands"].append(command)
            completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
            checks.assertEqual(git(checkout, "rev-parse", "HEAD"), pin)
            checks.assertEqual(git(checkout, "status", "--porcelain"), "")
            record.update(state="PASS", failure_cleanup=failure, failure_log=str(task.logs),
                          update_check_output=check_output, update_stdout=completed.stdout, update_stderr=completed.stderr,
                          verified_tool_commit=pin)
        finally:
            failure_info = sys.exc_info()
            if failure_info[0] is not None:
                record.update(state="FAIL", error="".join(traceback.format_exception(*failure_info)))
                if isinstance(failure_info[1], subprocess.CalledProcessError):
                    record.update(failed_stdout=failure_info[1].stdout, failed_stderr=failure_info[1].stderr)
            rows = inventory(work)
            record.update(seconds=perf_counter() - started, files=rows,
                          peak_work_bytes=sum(r["bytes"] for r in rows), peak_work_files=len(rows),
                          ram_peak="未测量", vram_peak="未测量")
            write_json(logs / "maintenance.json", record)
            confined(work, root)
            for row in rows:
                path = confined(work / row["path"], work)
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
            shutil.rmtree(work)
            write_json(logs / "cleanup.json", {"state": "completed", "deleted_files": len(rows),
                       "released_bytes": sum(r["bytes"] for r in rows), "work_removed": not work.exists(),
                       "all_subprocesses_waited": True})
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(verify(args.config))
