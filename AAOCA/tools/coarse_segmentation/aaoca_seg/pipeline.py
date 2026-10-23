"""Isolated attempts, original-grid QC and hash-checked case-level reuse."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from . import __version__
from .images import combine, inspect_image
from .schema import LABELS, TASKS

CODE_ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_device(device):
    if not re.fullmatch(r"cpu|gpu(?::[0-9]+)?", device):
        raise ValueError("设备必须是 cpu、gpu 或 gpu:可见编号")
    return device


def validate_output(root, cases):
    root = Path(root).resolve()
    if root == CODE_ROOT or root.is_relative_to(CODE_ROOT):
        raise ValueError("运行结果必须放在粗分割源码目录之外")
    for row in cases:
        image = Path(row["image"]).resolve()
        if root == image.parent or root.is_relative_to(image.parent) or image.is_relative_to(root):
            raise ValueError("结果目录必须与输入影像目录分开")
    return root


def environment():
    versions = {}
    for package in ("TotalSegmentator", "numpy", "nibabel", "torch", "nnunetv2"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = Path(os.environ.get("TOTALSEG_HOME_DIR", str(Path.home() / ".totalsegmentator"))) / "config.json"
    license_present = False
    try:
        license_present = bool(json.loads(config.read_text(encoding="utf-8")).get("license_number"))
    except (OSError, ValueError):
        pass
    return {"python": sys.version.split()[0], "versions": versions,
            "license_configured": license_present,
            "license_note": "仅检查是否配置，许可有效性由 TotalSegmentator 验证",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "未设置；需遵守服务器分配"),
            "pipeline_version": __version__}


def require_backend():
    info = environment()
    if info["versions"]["TotalSegmentator"] != "2.18.0":
        raise ValueError("请在独立环境安装 requirements-inference.txt（TotalSegmentator 2.18.0）")
    if not info["license_configured"]:
        raise ValueError("心腔和冠脉任务需要先配置 TotalSegmentator 非商业许可，见 README")
    return info


def commands(image, attempt, device="gpu:0", threads=2):
    validate_device(device)
    if not 1 <= threads <= 64:
        raise ValueError("threads 必须在 1–64 之间")
    result = []
    for task in TASKS:
        cmd = [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
               "-i", str(image), "-o", str(Path(attempt) / task), "--task", task,
               "--device", device, "--nr_thr_resamp", str(threads), "--nr_thr_saving", str(threads)]
        if task == "total":
            cmd += ["--roi_subset"] + [name for _, name, _, source in LABELS if source == task]
        result.append((task, cmd))
    return result


def plan(cases, root, device="gpu:0", threads=2):
    root = validate_output(root, cases)
    results = []
    for case in cases:
        row = dict(case)
        try:
            row["geometry"] = inspect_image(case["image"])
            row["commands"] = [cmd for _, cmd in commands(case["image"], root / "cases" / case["case_id"] / "ATTEMPT", device, threads)]
            row["status"] = "ready"
        except Exception as error:
            row.update(status="blocked", error=str(error))
        results.append(row)
    return {"mode": "plan_only", "output_root": str(root), "cases": results,
            "ready": sum(row["status"] == "ready" for row in results), "environment": environment(),
            "note": "预检只读文件头；真实 HU/间距依赖清单来源确认，不代表分割质量通过"}


@contextmanager
def run_lock(root):
    path = Path(root) / ".run.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": stamp()}, stream)
    except FileExistsError:
        raise ValueError("输出目录已有运行锁；先核对原任务，不要直接删除锁") from None
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def stopped(stop_file):
    return stop_file is not None and Path(stop_file).exists()


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def execute(cmd, log_path, stop_file=None):
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                if stopped(stop_file):
                    raise KeyboardInterrupt("用户请求停止")
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f"模型退出码 {process.returncode}，详见 {Path(log_path).name}")
        finally:
            stop_child(process)


def reusable(case_dir, identity):
    try:
        record = json.loads((case_dir / "latest.json").read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("identity") != identity:
            return None
        attempt = (case_dir / record["attempt"]).resolve()
        if not attempt.is_relative_to(case_dir.resolve()) or not record.get("artifacts"):
            return None
        expected_files = {str(Path(task) / f"{name}.nii.gz") for _, name, _, task in LABELS}
        expected_files.update({"aaoca_11_labels.nii.gz", "qc.json", "labels.ctbl"})
        if set(record["artifacts"]) != expected_files:
            return None
        for relative, expected in record["artifacts"].items():
            artifact = (attempt / relative).resolve()
            if not artifact.is_relative_to(attempt) or sha256(artifact) != expected:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_batch(cases, root, device="gpu:0", threads=2, stop_file=None, backend=None):
    root = validate_output(root, cases)
    validate_device(device)
    if not 1 <= threads <= 64:
        raise ValueError("threads 必须在 1–64 之间")
    root.mkdir(parents=True, exist_ok=True)
    backend = backend or require_backend()
    code_hashes = {p.name: sha256(p) for p in sorted((CODE_ROOT / "aaoca_seg").glob("*.py"))}
    state = {"started_at": stamp(), "status": "running", "total": len(cases), "finished": 0,
             "completed": 0, "reused": 0, "failed": 0, "cases": [], "review_required": True}
    with run_lock(root):
        try:
            for case in cases:
                if stopped(stop_file):
                    raise KeyboardInterrupt("用户请求停止")
                state.update(current_case=case["case_id"], phase="input_check")
                write_json(root / "status.json", state)
                print(f"[{case['case_id']}] 检查输入", flush=True)
                case_dir = root / "cases" / case["case_id"]
                attempt = None
                record = None
                try:
                    geometry = inspect_image(case["image"], full=True)
                    identity = {"source": str(Path(case["image"]).resolve()), "sha256": sha256(case["image"]),
                                "device": device, "threads": threads, "backend": backend,
                                "source_code_sha256": code_hashes}
                    old = reusable(case_dir, identity)
                    if old:
                        state["reused"] += 1
                        state["cases"].append({"case_id": case["case_id"], "status": "reused", "attempt": old["attempt"]})
                        print(f"[{case['case_id']}] 哈希核验通过，复用已完成结果", flush=True)
                    else:
                        attempt = case_dir / ("attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
                        attempt.mkdir(parents=True)
                        record = {"case_id": case["case_id"], "identity": identity, "geometry": geometry,
                                  "attempt": attempt.name, "started_at": stamp(), "status": "running"}
                        write_json(attempt / "record.json", record)
                        for task, cmd in commands(case["image"], attempt, device, threads):
                            if stopped(stop_file):
                                raise KeyboardInterrupt("用户请求停止")
                            state["phase"] = task
                            state["model_log"] = str((attempt / f"{task}.log").relative_to(root))
                            write_json(root / "status.json", state)
                            print(f"[{case['case_id']}] {task}", flush=True)
                            write_json(attempt / f"{task}.command.json", cmd)
                            execute(cmd, attempt / f"{task}.log", stop_file)
                        state["phase"] = "merge_and_qc"
                        write_json(root / "status.json", state)
                        files, qc = combine(case["image"], attempt)
                        if sha256(case["image"]) != identity["sha256"]:
                            raise ValueError("运行期间输入内容发生变化，结果不能标记完成")
                        record.update(status="completed", ended_at=stamp(), review_required=True,
                                      empty_masks=qc["empty_masks"],
                                      artifacts={str(p.relative_to(attempt)): sha256(p) for p in files})
                        write_json(attempt / "record.json", record)
                        write_json(case_dir / "latest.json", record)
                        state["completed"] += 1
                        state["cases"].append({"case_id": case["case_id"], "status": "completed", "attempt": attempt.name})
                        print(f"[{case['case_id']}] 粗分割完成，待人工复核", flush=True)
                except KeyboardInterrupt:
                    if attempt and record:
                        record.update(status="stopped", ended_at=stamp())
                        write_json(attempt / "record.json", record)
                    raise
                except Exception as error:
                    failure = {"case_id": case["case_id"], "status": "failed", "error": str(error), "ended_at": stamp()}
                    if attempt:
                        write_json(attempt / "failure.json", failure)
                        if record:
                            record.update(status="failed", ended_at=stamp(), error=str(error))
                            write_json(attempt / "record.json", record)
                    state["failed"] += 1
                    state["cases"].append(failure)
                    print(f"[{case['case_id']}] 失败: {error}", flush=True)
                state["finished"] += 1
                write_json(root / "status.json", state)
            state["status"] = "completed_with_errors" if state["failed"] else "completed"
        except KeyboardInterrupt:
            state["status"] = "stopped"
            print("已停止；完整完成的病例可核验后复用，其余病例下次使用新 attempt", flush=True)
        finally:
            state.update(ended_at=stamp(), phase="idle", current_case=None)
            write_json(root / "status.json", state)
    return state
