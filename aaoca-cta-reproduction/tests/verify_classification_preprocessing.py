import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TASK = Path("D:/projects/back/t17/aaoca-assistant4/20261121-preprocess")
REPORT = ROOT / "reports" / "classification-preprocessing-20261121.json"


def worker(args):
    from aaoca_reproduction.audit import load_frozen
    from aaoca_reproduction.records import load_config, write_json

    if args.dependency_dir:
        sys.path.insert(0, str(Path(args.dependency_dir).resolve(strict=True)))
    from aaoca_reproduction.preprocessing import prepare_classification

    config = load_config(args.config)
    manifest = load_frozen(config)
    config.update(data_root=TASK / "data", course_root=TASK / "course",
                  classification_recipe="author_native_index", cpu_threads=2,
                  max_work_bytes=256 * 2**20, max_work_files=20)
    keys = config["smoke_cases"]
    if len(keys) != 2 or len(set(keys)) != 2:
        raise ValueError("许可仅覆盖两例已选中的真实输入")
    cases = [row for key in keys for row in manifest["cases"] if row["case_id"] == key]
    if len(cases) != 2 or {row["dataset"] for row in cases} != {"aaoca", "imagechd"}:
        raise ValueError("需要原冻结配置各数据集一例")
    results = []
    for case in cases:
        write_json(TASK / "course" / "progress.json", {"selected_case_ids": keys, "current_case_id": case["case_id"],
                                                      "completed": results, "phase": "preprocessing"})
        directory = TASK / "data" / case["case_id"]
        result = prepare_classification(case, config, directory, verify_author=True)
        reassigned = dict(case, role="predict" if case["role"] != "predict" else "candidate_train",
                          aliases=list(reversed(case["aliases"])))
        reused = prepare_classification(reassigned, config, directory)
        if not reused["reused"] or reused["outputs"] != result["outputs"]:
            raise AssertionError("预处理没有按相同身份复用")
        if result["method"]["formal_mixed_classification_ready"]:
            raise AssertionError("作者参考方向尚未确认，不能放行混合正式分类")
        result["role_and_alias_change_reused"] = True
        results.append(result)
        write_json(TASK / "course" / "progress.json", {"selected_case_ids": keys, "completed": results, "phase": "case_completed"})
    print(json.dumps({"cases": results, "training_updates": 0, "gpu_used": False}, ensure_ascii=False), flush=True)


def main(args):
    from aaoca_reproduction.records import load_config, read_json, write_json

    config = load_config(args.config)
    dependency_dir = Path(args.dependency_dir).resolve(strict=True) if args.dependency_dir else None
    if TASK.exists():
        raise FileExistsError("批准测试根已存在，拒绝覆盖或清理既有文件")
    if REPORT.exists():
        raise FileExistsError("本批报告已经存在，重试须核对原授权范围")
    TASK.mkdir(parents=True)
    (TASK / ".gitignore").write_text("*\n", encoding="utf-8")
    course = TASK / "course"
    course.mkdir()
    (course / "tmp").mkdir()
    log = course / "worker.log"
    started = monotonic()
    started_utc = datetime.now(timezone.utc)
    deadline_utc = started_utc + timedelta(seconds=600)
    code_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    modified = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    print(json.dumps({"event": "approved_preprocessing_started", "code_sha": code_sha,
                      "modified_tracked_files": modified, "started_utc": started_utc.isoformat(),
                      "started_beijing": started_utc.astimezone(timezone(timedelta(hours=8))).isoformat(),
                      "deadline_utc": deadline_utc.isoformat(),
                      "deadline_beijing": deadline_utc.astimezone(timezone(timedelta(hours=8))).isoformat()}, ensure_ascii=False), flush=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--batch", args.batch,
               "--config", str(Path(args.config).resolve())]
    if dependency_dir is not None:
        command += ["--dependency-dir", str(dependency_dir)]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TEMP=str(course / "tmp"),
                       TMP=str(course / "tmp"), TMPDIR=str(course / "tmp"), OMP_NUM_THREADS="2",
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    process = None
    error = None
    peak_rss = None
    result = None
    success = False
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                memory = psutil.virtual_memory()
                if memory.available < memory.total * (1 - config["host_memory_fraction"]):
                    error = "host_memory_budget_exceeded"
                elif monotonic() - started >= 570:
                    error = "time_budget_exceeded_cleanup_reserve_30_seconds"
                if error is not None:
                    process.kill()
                    break
                sleep(0.05)
            process.wait()
        if error is None and process.returncode != 0:
            error = "worker_exit_" + str(process.returncode)
        if error is None:
            lines = log.read_text(encoding="utf-8").splitlines()
            result = json.loads(lines[-1])
            peak_rss = max(row["memory_guard"]["peak_rss_bytes"] for row in result["cases"])
            success = len(result["cases"]) == 2 and result["training_updates"] == 0
            if not success:
                raise AssertionError("测试回执没有覆盖获准两例及零更新")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        failure = sys.exc_info()[1]
        if failure is not None:
            error = str(failure)
        progress = read_json(course / "progress.json") if (course / "progress.json").exists() else None
        files = [p for p in TASK.rglob("*") if p.is_file()]
        inventory = []
        for path in files:
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(TASK.resolve(strict=True)):
                raise RuntimeError("发现不属于本任务的文件，拒绝清理")
            stamp = path.stat()
            inventory.append({"path": str(path), "bytes": stamp.st_size, "links": stamp.st_nlink})
        size = sum(x["bytes"] for x in inventory if x["links"] == 1)
        within_budget = len(files) <= 20 and size <= 256 * 2**20
        record = {"state": "passed" if success and within_budget and failure is None else "failed",
                  "scope": "two_real_cases_author_native_index_preprocessing_only", "result": result,
                  "selected_case_ids": config["smoke_cases"], "progress": progress,
                  "completed_cases": len(progress["completed"]) if progress else 0,
                  "training_updates": 0, "gpu_used": False, "peak_rss_bytes": peak_rss,
                  "rss_scope": "worker self-sampling inside preprocessing; imports excluded; unavailable after worker failure",
                  "vram_peak": "未测量；无GPU计算", "seconds_before_cleanup": monotonic() - started,
                  "started_utc": started_utc.isoformat(), "deadline_utc": deadline_utc.isoformat(),
                  "code_sha": code_sha, "modified_tracked_files": modified,
                  "worker_stop_utc": (started_utc + timedelta(seconds=570)).isoformat(),
                  "inventory_before_cleanup": inventory, "within_disk_budget": within_budget,
                  "error": error, "worker_tail": log.read_text(encoding="utf-8")[-10000:] if log.exists() else ""}
        write_json(REPORT, record)
        for path in files:
            path.unlink()
        for path in sorted((p for p in TASK.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if path.is_symlink() or not path.resolve().is_relative_to(TASK.resolve()):
                raise RuntimeError("发现外部目录连接，拒绝清理")
            path.rmdir()
        TASK.rmdir()
        seconds = monotonic() - started
        record.update(seconds=seconds, speed_cases_per_second=(len(result["cases"]) / seconds if result else 0),
                      cleanup={"deleted_files": len(files), "released_bytes": size, "task_absent": not TASK.exists()})
        write_json(REPORT, record)
        if REPORT.stat().st_size > 2 * 2**20:
            raise OSError("正式报告超出2MiB预算")
    print(json.dumps({"state": record["state"], "report": str(REPORT), "error": error}, ensure_ascii=False))
    if record["state"] != "passed":
        raise RuntimeError("真实预处理短流程未通过，详见正式报告")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="已批准两例分类预处理；零更新；监督预算并自动清理")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dependency-dir")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch", choices=("original", "r2"), default="original")
    arguments = parser.parse_args()
    if arguments.batch == "r2":
        TASK = Path("D:/projects/back/t17/aaoca-assistant4/20261121-preprocess-r2")
        REPORT = ROOT / "reports" / "classification-preprocessing-r2-20261121.json"
    worker(arguments) if arguments.worker else main(arguments)
