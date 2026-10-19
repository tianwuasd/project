"""Small synthetic server acceptance case, beside chd-ct-reproduction in project."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def fixture(repo, output):
    import nibabel as nib
    import numpy as np
    import yaml

    raw = output / "raw"
    raw.mkdir()
    target = np.zeros((18, 16, 16), np.uint8)
    for label in range(1, 8):
        target[2 * label : 2 * label + 2, 3:13, 3:13] = label
    for index in range(6):
        image = np.random.default_rng(index).normal(100, 20, target.shape).astype(np.float32)
        nib.save(nib.Nifti1Image(image, np.eye(4)), raw / f"ct_{index}_image.nii.gz")
        nib.save(nib.Nifti1Image(target, np.eye(4)), raw / f"ct_{index}_label.nii.gz")
    config = yaml.safe_load((repo / "configs/chd-smoke.yaml").read_text(encoding="utf-8"))
    config["cross_validation"] = dict(folds=3, development_scope="all")
    (output / "tiny.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(job, device, cap):
    state = read(job / "job_state.json")
    checks = {"all_steps_passed": state["status"] == "passed"}
    steps = {r["name"]: r for r in state["steps"]}
    for name in ("preprocess", "test", "predict", "evaluate"):
        output = Path(steps[name]["output"])
        checks[f"{name}_cpu_record"] = (output / "cpu-allocation.json").is_file()
        checks[f"{name}_progress"] = (output / "progress.json").is_file()
    checks["training_tuning_unchanged"] = not (
        Path(steps["smoke"]["output"]) / "cpu-allocation.json"
    ).exists()
    predicted = Path(steps["predict"]["output"]) / "predict"
    report = read(predicted / "prediction-report.json")
    checks["six_predictions"] = len(report["cases"]) == 6
    checks["out_of_fold_protocol"] = all(r["inference_protocol"] == "out_of_fold" for r in report["cases"])
    checks["six_evaluated"] = (
        read(Path(steps["evaluate"]["output"]) / "evaluate/evaluation.json")["cases_evaluated"] == 6
    )
    actual_gpus = set()
    peak_active = 0
    if device == "cuda":
        for name, child in (("test", "test/predictions"), ("predict", "predict")):
            queue = read(Path(steps[name]["output"]) / child / "queue.json")
            peak_active = max(peak_active, queue.get("peak_active_gpus", 0))
            actual_gpus.update(r.get("gpu_uuid") for r in queue["stages"].values() if r.get("gpu_uuid"))
            checks[f"{name}_queue_cap"] = queue["status"] == "complete" and queue["gpu_limit"] <= cap
        release = state.get("gpu_released_at")
        checks["released_before_cpu_tail"] = bool(release) and all(
            steps[name]["started_at"] >= release for name in ("test", "predict", "evaluate")
        )
        hold = Path(state["gpu_hold_directory"])
        from chd_ct.server.gpu_hold import alive, records

        checks["no_live_holders"] = not any(alive(record) for record in records(hold))
    return dict(
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        device=device,
        gpu_cap=cap,
        actual_gpu_uuids=sorted(actual_gpus),
        multiple_gpus_observed=len(actual_gpus) > 1,
        peak_active_gpus=peak_active,
        simultaneous_multi_gpu_verified=peak_active > 1,
        workflow=str(job),
        synthetic=True,
        note="Software acceptance only; single available GPU cannot verify multi-GPU throughput.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1] / "chd-ct-reproduction"
    )
    parser.add_argument("--python", type=Path, help="Existing CHD environment Python; no installation")
    parser.add_argument("--work-root", type=Path, default=Path.cwd() / "resource-smoke-runs")
    parser.add_argument("--gpu-count", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--timeout", type=int, default=900, help="Total seconds, including GPU waiting")
    parser.add_argument("--_fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args._fixture:
        fixture(repo, args._fixture)
        return 0
    if sys.platform != "linux":
        parser.error("Run this server acceptance case on Linux; fixture/unit checks can run locally.")
    python = (args.python or Path(sys.executable)).resolve()
    if not python.is_file() or not (repo / "start_server.py").is_file():
        parser.error("Check --python and --repo paths")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    output = args.work_root.resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONUNBUFFERED="1")
    print("独立合成验收目录：" + str(output), flush=True)
    subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--repo", str(repo), "--_fixture", str(output)],
        env=env,
        check=True,
    )
    command = [
        str(python),
        str(repo / "start_server.py"),
        "--tasks",
        "environment,preprocess,smoke,test,predict,evaluate",
        "--name",
        "资源调度合成验收",
        "--runtime",
        str(output / "runtime"),
        "--results",
        str(output / "results"),
        "--dataset",
        str(output / "raw"),
        "--prepared",
        str(output / "prepared"),
        "--config",
        str(output / "tiny.yaml"),
        "--size",
        "16",
        "--no-resample",
        "--allow-smoke",
        "--python",
        str(python),
        "--device",
        args.device,
        "--gpu-count",
        str(args.gpu_count),
        "--gpu",
        args.gpu,
        "--non-interactive",
        "--foreground",
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, env=env)
    try:
        code = process.wait(timeout=args.timeout)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            subprocess.run(
                [
                    str(python),
                    str(repo / "start_server.py"),
                    "--runtime",
                    str(output / "runtime"),
                    "--stop",
                    "--non-interactive",
                ],
                env=env,
                timeout=45,
            )
        print("验收已中断或超时，检查该目录日志后可续跑；正式任务未更改。", flush=True)
        return 130
    if code:
        print("验收流程失败，输出保留在：" + str(output), flush=True)
        return code
    sys.path.insert(0, str(repo / "src"))
    job = Path(read(output / "runtime/last-job.json")["directory"])
    report = verify(job, args.device, args.gpu_count)
    report["elapsed_seconds"] = time.monotonic() - started
    (output / "acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, passed in report["checks"].items():
        print(("PASS " if passed else "FAIL ") + name)
    print("验收报告：" + str(output / "acceptance.json"))
    print("详细日志：" + str(job / "launcher.log"))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
