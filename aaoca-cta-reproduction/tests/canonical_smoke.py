import argparse
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
    if os.environ.get("AAOCA_SMOKE_EXTRA_SITE"):
        sys.path.append(os.environ["AAOCA_SMOKE_EXTRA_SITE"])
    sys.argv = [str(PROJECT / "run.py"), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--cases", nargs=2, required=True)
parser.add_argument("--dependency-dir", type=Path, default=PROJECT / ".deps/torchio0196")
parser.add_argument("--extra-site-dir", type=Path)
args = parser.parse_args()
ROOT = args.root.resolve()
if ROOT.exists() or not ROOT.parent.is_dir() or ROOT == PROJECT or PROJECT.is_relative_to(ROOT):
    raise ValueError("测试目录须独立、尚不存在且父目录已批准")
ROOT.mkdir()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

(ROOT / ".gitignore").write_text("*\n", encoding="ascii")
previous = read(args.report)
result = {"attempt": previous["attempt"] + 1, "previous_attempt": previous, "state": "failed", "phases": [],
          "peak_job_committed_bytes": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "gpu_used": False, "test_model_input_shape": [64, 64, 64], "preprocessing_shape": [215, 215, 85], "model": "full_SENet154"}
started = time.monotonic()
base = read(args.base)
manifest = read(args.base.parent / base["input_manifest"])
manifest["cases"] = [r for r in manifest["cases"] if r["case_id"] in args.cases]
assert len(manifest["cases"]) == 2
result["cases"] = [{"case_id": r["case_id"], "role": r["role"]} for r in manifest["cases"]]
sources = [Path(r[k]) for r in manifest["cases"] for k in ("image", "label")]

def stamps():
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sources}

original = stamps()

def phase(action, config, *extra):
    before = time.monotonic()
    path = ROOT / "console.log"
    offset = path.stat().st_size if path.exists() else 0
    command = [sys.executable, "-u", "-B", "-X", "utf8", __file__, "--worker", action, "--config", str(config), *map(str, extra)]
    with path.open("ab", buffering=0) as log:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                           AAOCA_SMOKE_DEPENDENCY_DIR=str(args.dependency_dir.resolve(strict=True)))
        if args.extra_site_dir:
            environment["AAOCA_SMOKE_EXTRA_SITE"] = str(args.extra_site_dir.resolve(strict=True))
        job = WindowsJobRunner(command, log, PROJECT, environment, 8 * 2**30)
        try:
            while job.poll() is None:
                result["peak_job_committed_bytes"] = max(result["peak_job_committed_bytes"], job.memory())
                if time.monotonic() - started > 1800:
                    raise TimeoutError("达到30分钟测试时限")
                host = psutil.virtual_memory()
                if host.available < host.total * .2:
                    raise MemoryError("整机可用内存低于20%")
                if len([p for p in ROOT.rglob("*") if p.is_file()]) > 30 or os.fstat(log.fileno()).st_size > 8 * 2**20:
                    raise OSError("文件数量或日志容量超限")
                time.sleep(.25)
        finally:
            if job.poll() is None:
                job.kill()
            code = job.wait()
            result["peak_job_committed_bytes"] = max(result["peak_job_committed_bytes"], job.memory())
            assert job.active() == 0
            job.close()
            result["phases"].append({"action": action, "extra": list(map(str, extra)), "exit_code": code,
                                     "seconds": round(time.monotonic() - before, 3)})
            print(json.dumps(result["phases"][-1]), flush=True)
    if code:
        raise RuntimeError("真实流程失败：" + action)
    with path.open("rb") as log:
        log.seek(offset)
        text = log.read().decode("utf-8").replace("\r\n", "\n")
    start = text.rfind("\n{\n")
    return json.loads(text[start + 1:] if start >= 0 else text)

try:
    write(ROOT / "manifest.json", manifest)
    base.update(state_root=str(ROOT / "course"), course_root=str(ROOT / "course"), data_root=str(ROOT / "data"),
                input_manifest=str(ROOT / "manifest.json"), max_memory_bytes=8 * 2**30, max_work_bytes=12 * 2**30,
                max_work_files=30, cpu_threads=2, prepared_inputs={})
    write(ROOT / "base.json", base)
    phase("generate-run-configs", ROOT / "base.json", "--output", ROOT / "configs", "--device", "cpu", "--epochs", 2)
    prep = ROOT / "configs/project.aaoca_present.preprocess.json"
    train = ROOT / "configs/project.aaoca_present.train.json"
    phase("prepare-dataset", prep, "--check")
    prepared = phase("prepare-dataset", prep)
    reused = phase("prepare-dataset", prep)
    assert prepared["completed"] == 2 and reused["reused"] == 2
    result["preprocessing"] = {"completed": 2, "reused": 2, "seconds": prepared["seconds"], "cases_per_second": 2 / prepared["seconds"]}
    receipts = [read(p) for p in (ROOT / "data/aaoca-classification").glob("*/classification.json")]
    assert len(receipts) == 2 and all(r["geometry"]["target_axis_codes"] == ["R", "A", "S"] and r["geometry"]["round_trip_checked"] for r in receipts)
    result["geometry"] = [{k: r["geometry"][k] for k in ("target_axis_codes", "target_shape_xyz", "round_trip_checked")} for r in receipts]
    config = read(train)
    config["run_purpose"] = "smoke"
    config["classification"].update(smoke_input_shape=[64, 64, 64], max_update_attempts=1, artifact_budget_bytes=12 * 2**30, checkpoint_interval_updates=1)
    write(train, config)
    first = phase("train", train)
    assert first["state"]["successful_updates"] == 1
    config["classification"]["max_update_attempts"] = 2
    write(train, config)
    second = phase("train", train, "--resume")
    assert second["state"]["successful_updates"] == 2
    final = phase("train", train, "--resume")
    assert final["state"]["complete"] and final["state"]["epoch"] == 2 and final["state"]["attempted_updates"] == 2
    result["training"] = {k: final["state"][k] for k in ("epoch", "complete", "attempted_updates", "successful_updates", "test_only")}
    result["training"]["peak_rss_bytes"] = max(v["memory"]["peak_rss_bytes"] for v in (first, second, final))
    run = Path(config["classification"]["run_directory"])
    assert (run / "best.pt").is_file() and (run / "latest.pt").is_file()
    result["checkpoint_bytes"] = {p.name: p.stat().st_size for p in run.glob("*.pt")}
    validation = next(r["case_id"] for r in manifest["cases"] if r["role"] == "candidate_validation")
    config["classification"].update(ensemble_mode="single_model", prediction_case_ids=[validation], prediction_directory=str(ROOT / "course/prediction"), weights=[{
        "path": str(run / "latest.pt"), "task": "aaoca_present", "labels": {"0": "normal_origin", "1": "AAOCA"},
        "author_release": "b134e621943f60f20caf6a9272aa85ceeeb441f8", "source": "two-real-case-authorized-smoke",
        "format": "local_complete_checkpoint", "strategy": "local_from_scratch", "member_index": 0, "test_only": True}])
    write(ROOT / "predict.json", config)
    predicted = phase("predict", ROOT / "predict.json")
    assert predicted["test_only"] and len(predicted["results"]) == 1 and 0 <= predicted["results"][0]["probability"] <= 1
    result["prediction"] = {"count": 1, "test_only": True, "probability": predicted["results"][0]["probability"]}
    result["reset_check"] = phase("reset-preprocessing", prep, "--check")
    result["reset"] = phase("reset-preprocessing", prep, "--confirm-output-root", ROOT / "data/aaoca-classification")
    assert not any((ROOT / "data/aaoca-classification").glob("*/classification.json"))
    result["state"] = "passed"
finally:
    result["error"] = str(sys.exc_info()[1]) if sys.exc_info()[1] else None
    result["seconds_total"] = round(time.monotonic() - started, 3)
    result["source_unchanged"] = stamps() == original
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    assert ROOT == args.root.resolve() and all(p.resolve().is_relative_to(ROOT) for p in files)
    result["files"] = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in files]
    result["cleaned_files"] = len(files)
    result["cleaned_bytes"] = sum(p["bytes"] for p in result["files"])
    assert len(files) <= 30 and result["cleaned_bytes"] <= 12 * 2**30
    result["processes_exited"] = True
    if (ROOT / "console.log").exists():
        result["console_tail"] = (ROOT / "console.log").read_text(encoding="utf-8")[-24000:]
    write(args.report, result)
    shutil.rmtree(ROOT)
    result["root_absent"] = not ROOT.exists()
    write(args.report, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"previous_attempt", "console_tail", "files"}}, ensure_ascii=False), flush=True)
