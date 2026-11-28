import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import psutil

from windows_job_runner import WindowsJobRunner


PROJECT = Path(__file__).resolve().parents[1]
if sys.argv[1:2] == ["--worker"]:
    sys.path.insert(0, str(PROJECT / "src"))
    sys.path.insert(0, os.environ["AAOCA_SMOKE_DEPENDENCY_DIR"])
    if sys.argv[2] in {"check-statistics", "check-selection"}:
        from aaoca_reproduction.records import load_config
        from aaoca_reproduction.audit import load_frozen
        from aaoca_reproduction.experiments import prepared_statistics, validation_threshold, patient_confidence_intervals
        from aaoca_reproduction.data import prepared_receipt, load_array
        from aaoca_reproduction.preprocessing import array_statistics
        cfg = load_config(sys.argv[4])
        current = load_frozen(cfg)
        if sys.argv[2] == "check-statistics":
            for case in [row for row in current["cases"] if row["dataset"] in {"aaoca", "imagechd"}]:
                receipt = prepared_receipt(case, cfg)
                expected = array_statistics(load_array(receipt, training=False))
                # 直接调用兼容输入允许的无生产元数据合同，读取的仍是本批真实数组。
                actual = prepared_statistics({"directory": receipt["directory"], "producer_provenance": None}, cfg)
                assert actual == expected
            output = {"real_existing_arrays": 2, "equal": True, "historical_receipt_tested": False}
        else:
            rows = json.loads((Path(cfg["classification"]["run_directory"]) / "test-ensemble/predictions.json").read_text(encoding="utf-8"))["results"]
            selected = [next(row for row in rows if row["truth"] == value) for value in (0, 1)]
            threshold = validation_threshold(selected)
            intervals = patient_confidence_intervals(selected, threshold)
            output = {"real_scored_cases": 2, "threshold_finite": 0 <= threshold <= 1,
                      "bootstrap_replicates": intervals["replicates"], "test_only": True}
        print(json.dumps(output, indent=2), flush=True)
        sys.exit(0)
    sys.argv = [str(PROJECT / "run.py"), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--cases", nargs=3, required=True)
parser.add_argument("--dependency-dir", type=Path, default=PROJECT / ".deps/torchio0196")
parser.add_argument("--resume-existing", action="store_true")
parser.add_argument("--author-only", action="store_true")
args = parser.parse_args()
if args.author_only and args.resume_existing:
    raise ValueError("作者补测须使用独立的新批次")
MAX_FILES = 50 if args.author_only else 120
MAX_BYTES = (5 if args.author_only else 20) * 2**30
MAX_SECONDS = 900 if args.author_only else 1800
if args.root.is_symlink():
    raise ValueError("测试根目录不能是符号链接")
ROOT = args.root.resolve()
if (ROOT.exists() != args.resume_existing) or not ROOT.parent.is_dir() or ROOT == PROJECT or PROJECT.is_relative_to(ROOT):
    raise ValueError("测试目录必须独立、尚不存在且位于已批准父目录")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


base = read(args.base)
manifest = read(Path(base["input_manifest"]))
manifest["cases"] = [row for row in manifest["cases"] if row["case_id"] in args.cases]
if len(manifest["cases"]) != 3 or {row["dataset"] for row in manifest["cases"]} != {"aaoca", "chd1282", "imagechd"}:
    raise ValueError("必须为三个来源各一名真实病例")
for row in manifest["cases"]:
    row["role"] = {"aaoca": "candidate_train", "chd1282": "candidate_validation", "imagechd": "predict"}[row["dataset"]]
sources = [Path(row[key]) for row in manifest["cases"] for key in ("image", "label")]


def stamps():
    return {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in sources}


original = {str(Path(row[key])): [row["source_stat"][key]["bytes"], row["source_stat"][key]["mtime_ns"]]
            for row in manifest["cases"] for key in ("image", "label")}
owner_path = ROOT / "test-owner.json"
expected_owner = {"schema": 1, "project": str(PROJECT), "cases": sorted(args.cases), "source_stats": original}
if args.resume_existing:
    owner = read(owner_path)
    if any(owner.get(key) != value for key, value in expected_owner.items()):
        raise ValueError("现存目录缺少本批来源与病例一致的测试归属")
ROOT.mkdir(exist_ok=args.resume_existing)
if not args.resume_existing:
    (ROOT / ".gitignore").write_text("*\n", encoding="ascii")
    write(owner_path, {**expected_owner, "started_unix": time.time()})
result = {"state": "failed", "phases": [], "cases": [{"case_id": row["case_id"], "dataset": row["dataset"]} for row in manifest["cases"]],
    "gpu_used": False, "train_updates_limit_per_member": 0 if args.author_only else 2,
    "train_updates_total_limit": 0 if args.author_only else 10, "author_only": args.author_only,
    "smoke_input_shape": [64, 64, 64], "full_preprocessing_shape": [215, 215, 85], "peak_job_committed_bytes": 0,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
if args.report.exists():
    result["previous_attempt"] = read(args.report)
elapsed_before_restart = time.time() - owner["started_unix"] if args.resume_existing else 0
started = time.monotonic() - elapsed_before_restart
result["resumed_existing_batch"] = args.resume_existing
result["elapsed_before_restart_seconds"] = elapsed_before_restart


def phase(action, config, *extra):
    before = time.monotonic()
    path = ROOT / "console.log"
    offset = path.stat().st_size if path.exists() else 0
    command = [sys.executable, "-u", "-B", "-X", "utf8", __file__, "--worker", action, "--config", str(config), *map(str, extra)]
    with path.open("ab", buffering=0) as log:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                           AAOCA_SMOKE_DEPENDENCY_DIR=str(args.dependency_dir.resolve(strict=True)))
        limit = 8 * 2**30 - psutil.Process().memory_info().rss - 64 * 2**20
        job = WindowsJobRunner(command, log, PROJECT, environment, limit)
        try:
            while job.poll() is None:
                result["peak_job_committed_bytes"] = max(result["peak_job_committed_bytes"], job.memory())
                if time.monotonic() - started > MAX_SECONDS:
                    raise TimeoutError("达到已批准批次时限")
                host = psutil.virtual_memory()
                if host.available < host.total * .2:
                    raise MemoryError("整机可用内存低于20%")
                # 容量由固定5成员/3病例及检查点结构给定上界，退出后统计稳定文件清单。
                if sum(path.is_file() for path in ROOT.rglob("*")) > MAX_FILES or os.fstat(log.fileno()).st_size > 16 * 2**20:
                    raise OSError("文件数或日志容量超过批准范围")
                time.sleep(.25)
        finally:
            if job.poll() is None:
                job.kill()
            code = job.wait()
            result["peak_job_committed_bytes"] = max(result["peak_job_committed_bytes"], job.memory())
            assert job.active() == 0
            job.close()
            record = {"action": action, "extra": list(map(str, extra)), "exit_code": code,
                      "seconds": round(time.monotonic() - before, 3)}
            result["phases"].append(record)
            print(json.dumps(record), flush=True)
    if code:
        raise RuntimeError("真实流程失败：" + action)
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    if sum(p.stat().st_size for p in files) > MAX_BYTES:
        raise OSError("稳定产物清单超过批准容量")
    with path.open("rb") as log:
        log.seek(offset)
        text = log.read().decode("utf-8").replace("\r\n", "\n")
    start = text.rfind("\n{\n")
    return json.loads(text[start + 1:] if start >= 0 else text)


def author_prediction(manifest):
    author = read(ROOT / "configs/project.author.json")
    author.update(run_purpose="smoke", smoke_case_limit=3)
    author["classification"].update(smoke_input_shape=[64, 64, 64], max_update_attempts=2)
    direct_manifest = deepcopy(manifest)
    for row in direct_manifest["cases"]:
        row["role"] = "predict"
    write(ROOT / "author-manifest.json", direct_manifest)
    author["input_manifest"] = str(ROOT / "author-manifest.json")
    write(ROOT / "configs/project.author.json", author)
    result["author_prediction"] = phase("predict-ensemble", ROOT / "configs/project.author.json", "--device", "cpu")
    assert result["author_prediction"]["members"] == 5 and result["author_prediction"]["test_only"]
    report = read(Path(result["author_prediction"]["output"]))
    assert len(report["results"]) == 3
    for row in report["results"]:
        assert len(row["members"]) == 5
        assert abs(row["probability"] - sum(member["probability"] for member in row["members"]) / 5) < 1e-12
    result["selection_interface"] = phase("check-selection", ROOT / "configs/project.author.json")


try:
    base.update(state_root=str(ROOT / "course"), course_root=str(ROOT / "course"), data_root=str(ROOT / "data"),
        input_manifest=str(ROOT / "manifest.json"), max_memory_bytes=8 * 2**30, max_work_bytes=MAX_BYTES,
        max_work_files=MAX_FILES, cpu_threads=2, prepared_inputs={})
    base["classification"].update(device="cpu")
    write(ROOT / "manifest.json", manifest)
    write(ROOT / "base.json", base)
    if not args.resume_existing:
        phase("prepare-dataset", ROOT / "base.json", "--check")
        prepared = phase("prepare-dataset", ROOT / "base.json")
        reused = phase("prepare-dataset", ROOT / "base.json")
        assert prepared["completed"] == 3 and prepared["skipped"] == 0 and reused["reused"] == 3
        result["preprocessing"] = {"completed": 3, "reused": 3, "seconds": prepared["seconds"],
                                   "cases_per_second": 3 / prepared["seconds"]}
    receipts = [read(path) for path in (ROOT / "data/aaoca-classification").glob("*/classification.json")]
    assert len(receipts) == 3 and all(row["geometry"]["round_trip_checked"] and row["geometry"]["target_axis_codes"] == ["R", "A", "S"] for row in receipts)
    result["geometry"] = [{"case_id": row["case_id"], "target_axis_codes": row["geometry"]["target_axis_codes"],
                           "source_binding": row["crop"].get("spatial_binding")} for row in receipts]
    if not args.resume_existing:
        phase("prepare-author-weights", ROOT / "base.json", "--check")
        result["author_weights"] = phase("prepare-author-weights", ROOT / "base.json")
        phase("generate-experiments", ROOT / "base.json", "--output", ROOT / "configs", "--device", "cpu")
    if args.author_only:
        author_prediction(manifest)
        result["state"] = "passed"
        sys.exit(0)
    fine = read(ROOT / "configs/project.finetune_robust.json")
    fine.update(run_purpose="smoke", smoke_case_limit=3)
    fine["classification"].update(smoke_input_shape=[64, 64, 64], epochs=2,
        max_update_attempts=2 if args.resume_existing else 1, checkpoint_interval_updates=1, artifact_budget_bytes=12 * 2**30)
    fine["classification"]["fine_tuning"]["warmup_epochs"] = 1
    write(ROOT / "configs/project.finetune_robust.json", fine)
    if not args.resume_existing:
        first = phase("finetune-ensemble", ROOT / "configs/project.finetune_robust.json")
        assert first["state"] == "stopped"
    finals = []
    for i in range(5):
        if args.resume_existing:
            break
        local = deepcopy(fine)
        settings = local["classification"]
        settings.update(member_index=i, seed=fine["classification"]["seed"] + i,
            pretrained_detection=fine["classification"]["author_members"][i], max_update_attempts=2,
            run_directory=str(Path(fine["classification"]["run_directory"]) / f"member_{i + 1}"))
        path = ROOT / f"train-member-{i + 1}.json"
        write(path, local)
        phase("train", path, *( ["--resume"] if i == 0 else []))
        final = phase("train", path, "--resume")
        assert final["state"]["complete"] and final["state"]["successful_updates"] == 2 and final["state"]["test_only"]
        finals.append({key: final["state"][key] for key in ("member_index", "successful_updates", "epoch", "complete", "test_only")})
    fine["classification"]["max_update_attempts"] = 2
    write(ROOT / "configs/project.finetune_robust.json", fine)
    resumed = phase("finetune-ensemble", ROOT / "configs/project.finetune_robust.json", "--resume")
    assert resumed["state"] == "completed" and resumed["completed_members"] == list(range(5))
    if args.resume_existing:
        finals = resumed["member_states"]
        assert all(row["successful_updates"] == 2 and row["complete"] for row in finals)
    result["training"] = {"members": finals, "total_real_updates": sum(row["successful_updates"] for row in finals), "ensemble_resume": resumed}
    result["statistics_compatibility"] = phase("check-statistics", ROOT / "base.json")
    predicted = phase("predict-ensemble", ROOT / "configs/project.finetune_robust.json", "--device", "cpu", "--check")
    assert predicted["members"] == 5 and predicted["cases"] == 1
    result["fine_tuned_prediction"] = phase("predict-ensemble", ROOT / "configs/project.finetune_robust.json", "--device", "cpu")
    author_prediction(manifest)
    result["state"] = "passed"
finally:
    error = sys.exc_info()[1]
    result["error"] = None if error is None or isinstance(error, SystemExit) and error.code == 0 else str(error)
    result["seconds_total"] = round(time.monotonic() - started, 3)
    result["source_unchanged"] = stamps() == original
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    assert all(path.resolve().is_relative_to(ROOT) and not path.is_symlink() for path in files)
    result["files"] = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size} for path in files]
    result["cleaned_files"] = len(files)
    result["cleaned_bytes"] = sum(row["bytes"] for row in result["files"])
    result["processes_exited"] = True
    if (ROOT / "console.log").exists():
        result["console_tail"] = (ROOT / "console.log").read_text(encoding="utf-8")[-24000:]
    write(args.report, result)
    assert args.report.stat().st_size <= 2 * 2**20
    shutil.rmtree(ROOT)
    result["root_absent"] = not ROOT.exists()
    write(args.report, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"console_tail", "files", "previous_attempt"}}, ensure_ascii=False), flush=True)
