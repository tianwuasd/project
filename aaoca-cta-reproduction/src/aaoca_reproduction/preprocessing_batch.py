from datetime import datetime, timezone
from contextlib import contextmanager
from math import prod
import os
from pathlib import Path
from time import perf_counter

import psutil

from .labels import validate_manifest
from .exclusions import SKIP_FILE
from .records import CLASSIFICATION_TARGET, event, read_json, signature, write_json


@contextmanager
def batch_lock(path, shared=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBRLCK if shared else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def recover_partial(directory, case, key, previous):
    if not directory.exists() or (directory / "classification.json").exists() or (directory / SKIP_FILE).exists():
        return {"files": 0, "bytes": 0}
    paths = list(directory.iterdir())
    if not paths:
        return {"files": 0, "bytes": 0}
    expected = {"case_id": case["case_id"], "identity": key, "directory": str(directory)}
    if expected not in previous.get("recovery_records", []) or directory.is_symlink():
        raise ValueError("未完成目录缺少当前源/方法的批次归属，须人工核对：" + str(directory))
    for name in ("image", "label"):
        if signature(case[name]) != case["source_stat"][name]:
            raise ValueError("未完成病例源输入已变化，拒绝清理")
    allowed = {SKIP_FILE + ".pending", "normalized.pending.nii.gz", "normalized.nii.gz", "training_xyz.pending.npy",
               "training_xyz.npy", "classification.json.pending", "source-label.seg.nrrd"}
    for path in paths:
        if path.name not in allowed or path.is_symlink() or not path.is_file():
            raise ValueError("未完成病例含未知文件或链接，拒绝清理：" + str(path))
        if path.name == "source-label.seg.nrrd" and not os.path.samefile(path, case["label"]):
            raise ValueError("标签入口与原标签不是同一文件，拒绝清理")
    result = {"files": len(paths), "bytes": sum(path.stat().st_size for path in paths if path.name != "source-label.seg.nrrd")}
    for path in paths:
        path.unlink()
    return result


def prepare_dataset(config, manifest, check=False):
    cases = validate_manifest(manifest)
    if not cases:
        raise ValueError("分类预处理清单为空")
    voxels = prod(CLASSIFICATION_TARGET)
    # NIfTI按未压缩float32及压缩开销估算，另计float16 NPY和小回执。
    per_case = (voxels * 4 * 1001 + 999) // 1000 + voxels * 2 + 1024 + 256 + 65536
    data_root = Path(config["data_root"]).resolve()
    root = (data_root / "aaoca-classification").resolve()
    if not root.is_relative_to(data_root):
        raise ValueError("共享分类预处理目录超出data_root")
    progress_path = Path(config["course_root"]).resolve() / "preprocessing/progress.json"
    plan = {"cases": len(cases), "output_root": str(root), "progress": str(progress_path),
            "batch_lock": str(root / ".batch.lock"),
            "target_xyz": list(CLASSIFICATION_TARGET), "files_per_case": 3,
            "formal_files_with_progress_lock_and_console_log": len(cases) * 3 + 3,
            "estimated_output_bytes": len(cases) * per_case,
            "console_log_allowance_bytes": 32 * 2**20,
            "estimated_one_case_temporary_bytes": 25 * 2**20,
            "temporary_estimate_excludes_accumulated_formal_outputs": True,
            "source_copies": 0, "training_updates": 0, "arrays_read": 0,
            "check_only": check, "ram_peak_measured": False}
    if check:
        return plan
    with batch_lock(root / ".batch.lock"):
        return run_batch(config, cases, root, progress_path, plan)


def run_batch(config, cases, root, progress_path, plan):
    from .preprocessing import method, prepare_classification, preprocessing_identity

    started = perf_counter()
    process = psutil.Process()
    previous = read_json(progress_path) if progress_path.exists() else {}
    if previous.get("state") == "running" and previous.get("pid") != process.pid and psutil.pid_exists(previous["pid"]):
        if psutil.Process(previous["pid"]).create_time() == previous["process_created"]:
            raise RuntimeError("前次批次进程仍存在，停止后再接续")
    recipe = method(config)
    recovery_records = list(previous.get("recovery_records", []))
    if previous.get("inflight") is not None and previous["inflight"] not in recovery_records:
        recovery_records.append(previous["inflight"])
    progress = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "pid": process.pid, "process_created": process.create_time(), "plan": plan,
                "completed": [], "skipped": [], "skipped_count": 0, "current_case": None, "reused_count": 0, "remaining": len(cases),
                "inflight": None, "recovery_records": recovery_records}
    for case in cases:
        directory = root / case["case_id"]
        key = preprocessing_identity(case, recipe)
        cleaned = recover_partial(directory, case, key, progress)
        if cleaned["files"]:
            event("classification_partial_cleaned", case_id=case["case_id"], **cleaned)
        progress.update(current_case=case["case_id"], completed_count=len(progress["completed"]),
                        inflight={"case_id": case["case_id"], "identity": key, "directory": str(directory)})
        write_json(progress_path, progress)
        began = perf_counter()
        result = prepare_classification(case, config, directory)
        recovery_records[:] = [record for record in recovery_records if record != progress["inflight"]]
        item = {"case_id": case["case_id"], "reused": result["reused"],
                "seconds": perf_counter() - began, "identity": result["identity"]}
        skipped = result.get("state") == "skipped"
        if skipped:
            item["reason"] = result["reason"]
            progress["skipped"].append(item)
        else:
            progress["completed"].append(item)
        progress.update(completed_count=len(progress["completed"]), seconds=perf_counter() - started,
                        skipped_count=len(progress["skipped"]),
                        reused_count=progress["reused_count"] + int(item["reused"] and not skipped),
                        remaining=len(cases) - len(progress["completed"]) - len(progress["skipped"]), inflight=None)
        write_json(progress_path, progress)
        event("classification_dataset_case_skipped" if skipped else "classification_dataset_case_completed",
              **item, completed=progress["completed_count"], skipped=progress["skipped_count"],
              remaining=progress["remaining"], total=len(cases))
    progress.update(state="completed", current_case=None, seconds=perf_counter() - started)
    write_json(progress_path, progress)
    return {"state": "completed", "cases": len(cases), "completed": len(progress["completed"]),
            "skipped": len(progress["skipped"]), "reused": sum(item["reused"] for item in progress["completed"]),
            "seconds": progress["seconds"], "output_root": str(root), "progress": str(progress_path),
            "training_updates": 0}
