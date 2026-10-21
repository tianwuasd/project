"""Small CPU-only verification acceptance case, without CT data or ML packages."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def check_modes(run, cache, cache_record):
    from chd_ct.server import jobs, launcher
    from chd_ct.server.resume import build_resume_plan

    job = run / "workflow"
    job.mkdir()
    steps, snapshots = [], []
    for index, task in enumerate(("preprocess", "diagnosis-labels", "diagnosis-features", "diagnosis-train")):
        output = job / "steps" / f"{index + 1:02d}_{task}"
        arguments = dict(task=task, device="cpu")
        if task == "preprocess":
            arguments["prepared"] = str(cache)
            snapshot = cache_record
        elif task != "diagnosis-train":
            target = output / task / ("labels.json" if task == "diagnosis-labels" else "features.json")
            target.parent.mkdir(parents=True)
            target.write_text("{}", encoding="utf-8")
            snapshot = dict(
                cache_record,
                outputs={str(target): hashlib.sha256(b"{}").hexdigest()},
                output_roots=[str(target.parent)],
            )
        else:
            arguments.update(
                features=str(Path(steps[2]["output"]) / "diagnosis-features"),
                diagnosis_labels=str(Path(steps[1]["output"]) / "diagnosis-labels"),
            )
            snapshot = None
        steps.append(dict(name=task, output=str(output), arguments=arguments))
        snapshots.append(snapshot)
    plan = dict(
        schema_version=1,
        name="Synthetic resume choices",
        directory=str(job),
        runtime=str(run / "runtime"),
        log=str(job / "launcher.log"),
        steps=steps,
    )
    state = jobs.initial_state(plan)
    state["status"] = "failed"
    for step, record in zip(state["steps"], snapshots):
        step.update(status="passed" if record else "failed")
        if record:
            step["snapshot"] = record
    for name, data in (("plan.json", plan), ("job_state.json", state)):
        (job / name).write_text(json.dumps(data), encoding="utf-8")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in job.rglob("*") if p.is_file()}
    checks = {}
    for mode in ("full", "simple", "none"):
        args = launcher.make_parser().parse_args(["--resume-job", str(job), "--resume-check", mode])
        selected = build_resume_plan(args, dict(runtime=plan["runtime"]), False)
        checks[mode] = selected["resume_verification"]
        assert all(s.get("reuse") for s in selected["steps"][:-1])
        (run / f"selected-{mode}.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    assert checks["full"]["files"] == 8 and checks["full"]["bytes"] == 48 * 1024**2 + 4
    assert checks["simple"]["files"] == 2 and checks["simple"]["bytes"] == 4
    assert checks["none"]["bytes"] == 0 and checks["none"]["status"] == "skipped"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in job.rglob("*") if p.is_file()}
    return checks


def main():
    parser = argparse.ArgumentParser(description="校验读取快速验收：合成数据，不训练、不用 GPU")
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chd-ct-reproduction/src"))
    from chd_ct.server.verification import verify_snapshots

    args.work_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="resume-read-", dir=args.work_root)).resolve()
    cache = run / "cache"
    cache.mkdir()
    expected, before = {}, {}
    for i in range(6):
        path = cache / f"case{i}.bin"
        content = bytes([i]) * (8 * 1024**2)
        path.write_bytes(content)
        expected[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        before[path] = path.stat().st_mtime_ns
    record = dict(inputs=expected, outputs={}, input_roots=[str(cache)], output_roots=[])
    start = time.monotonic()
    result = verify_snapshots(
        [(name, record) for name in ("preprocess", "smoke", "preflight", "train", "test")]
    )
    result["seconds"] = time.monotonic() - start
    assert result["files"] == 6 and result["references"] == 30 and result["bytes"] == 48 * 1024**2
    for path, mtime in before.items():
        assert path.stat().st_mtime_ns == mtime
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[str(path.resolve())]
    result["resume_modes"] = check_modes(run, cache, record)
    path = cache / "case0.bin"
    stamp = path.stat()
    with path.open("r+b") as stream:
        stream.write(b"changed")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    try:
        verify_snapshots([("intentional-change", record)], show_progress=False)
    except ValueError:
        result["changed_content_rejected"] = True
    else:
        raise AssertionError("修改内容后校验未拒绝")
    result["status"] = "passed"
    (run / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS：6 个文件跨 5 步只读取 48 MiB，内容修改能被发现，旧文件校验时保持原样。\n报告：{run}")
    print("三种续跑选择通过：全面读取缓存和依赖；简单仅读取特征/标签 4 字节；不检查读取 0 字节。")
    print("耗时仅代表本次合成读取；不代表正式 NAS 任务的加速倍数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
