import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aaoca_reproduction.audit import check_training, save_audit
from aaoca_reproduction.prepare import prepare_case
from aaoca_reproduction.records import check_work, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description="两例真实定位与裁剪；零分类训练；结束自动清理")
    parser.add_argument("--config", default="project.local.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["project_root"]
    task = (root / "work" / "smoke-20261121").resolve()
    if task.exists():
        raise FileExistsError("本轮短流程目录已存在，请先核实并清理本任务遗留产物")
    manifest, audit = save_audit(config)
    cases = {row["case_id"]: row for row in manifest["cases"]}
    if len(config["smoke_cases"]) != 2 or len(set(config["smoke_cases"])) != 2:
        raise ValueError("本轮只允许两例明确且不同的真实病例")
    selected = [cases[key] for key in config["smoke_cases"]]
    if {row["dataset"] for row in selected} != {"aaoca", "imagechd"}:
        raise ValueError("本轮需要AAOCA与ImageCHD各一例")
    task.mkdir(parents=True)
    start = perf_counter()
    results = []
    success = False
    try:
        for case in selected:
            directory = task / case["case_id"]
            result = prepare_case(case, config, directory, verify_center=True)
            reused = prepare_case(case, config, directory)
            if not reused["reused"] or reused["output_stat"] != result["output_stat"]:
                raise AssertionError("重复运行没有精确复用本次裁剪")
            results.append(result)
        training = check_training(config, manifest)
        if training["ready"] or training["training_updates"] != 0:
            raise AssertionError("没有医生标签时训练入口错误放行")
        success = True
    finally:
        failure = sys.exc_info()[1]
        files = [p for p in task.rglob("*") if p.is_file()]
        inventory = [{"path": str(p), "bytes": p.stat().st_size,
                      "hardlink_count": p.stat().st_nlink} for p in files]
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(task):
                raise RuntimeError("发现不属于本轮的文件连接，停止自动清理")
        footprint = check_work(config)
        for path in files:
            path.unlink()
        for path in sorted((p for p in task.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.rmdir()
        task.rmdir()
        record = {"state": "passed" if success else "failed", "scope": "two real localization-and-crop cases",
                  "cases": results, "audit": audit, "training_updates": 0,
                  "gpu_used": False, "vram_peak": "未测量；本轮没有GPU计算",
                  "classification_prediction": False, "seconds": perf_counter() - start,
                  "speed_cases_per_second": len(results) / max(perf_counter() - start, 1e-9),
                  "inventory_before_cleanup": inventory, "peak_work_footprint_observed": footprint,
                  "cleanup": {"deleted_files": len(files),
                              "released_new_bytes": sum(x["bytes"] for x in inventory if x["hardlink_count"] == 1),
                              "task_directory_absent": not task.exists()},
                  "error": str(failure) if failure is not None else None}
        write_json(root / "reports" / "smoke.json", record)
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
