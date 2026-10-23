"""Persistent controller shared by reconnecting server consoles."""
import csv
import io
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from .operations import (BUILDS, apply_license, check_device, hardware, install,
                         probe, python_path, runtime_env, stage_license)
from .schema import FIELDS, TASKS, read_manifest
from .service_state import (CODE_ROOT, alive, detached, file_lock, private_dir,
                            managed_release,
                            process_identity, read_json, redact, tail, timestamp,
                            write_json, write_private)


def validate_paths(config, workspace):
    output = Path(config["output_root"]).expanduser().resolve()
    if output == CODE_ROOT or output.is_relative_to(CODE_ROOT) or CODE_ROOT.is_relative_to(output):
        raise ValueError("结果目录必须与代码分开")
    if output == workspace or workspace.is_relative_to(output):
        raise ValueError("结果目录不能包含整个运行工作区")
    if any(output.is_relative_to(workspace / name) for name in ("venv", "private", "weights", "jobs", "manifests")):
        raise ValueError("结果目录不能使用环境、许可、模型缓存或任务记录目录")
    cases = read_manifest(config["manifest"]) if config["manifest"] else []
    for row in cases:
        image = Path(row["image"]).resolve()
        if output.is_relative_to(image.parent) or image.is_relative_to(output):
            raise ValueError("结果目录必须与输入影像分开")
        if image.is_relative_to(workspace / "venv") or image.is_relative_to(workspace / "jobs"):
            raise ValueError("影像不能位于运行环境或任务目录中")
    return cases


class ManagedController:
    def __init__(self, workspace):
        self.workspace = Path(workspace).expanduser().resolve()
        if (self.workspace.is_relative_to(CODE_ROOT)
                or (CODE_ROOT.is_relative_to(self.workspace) and not managed_release(self.workspace))):
            raise ValueError("工作区必须在源码目录之外")
        private_dir(self.workspace)
        self.config_path = self.workspace / "settings.json"
        if not self.config_path.exists():
            write_json(self.config_path, {"manifest": "", "output_root": str(self.workspace / "results"),
                       "device": "cpu", "threads": 2, "torch_build": "auto", "gpu_confirmed": False, "gpu_uuid": None})
        self.children = []

    @property
    def config(self):
        return read_json(self.config_path, {})

    def job(self):
        # Reap our own exited children, while restored controllers use OS identity.
        self.children[:] = [p for p in self.children if p.poll() is None]
        current = read_json(self.workspace / "current.json", {})
        job_id = current.get("id", "")
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            return None, {}
        path = self.workspace / "jobs" / job_id
        state = read_json(path / "state.json", {})
        owner = read_json(path / "owner.json", {})
        state = dict(state, active=alive(owner) or alive(read_json(path / "child.json")))
        if not state["active"] and state.get("status") in {"queued", "running"}:
            state["status"] = "interrupted"
        return path, state

    def active(self):
        return self.job()[1].get("active", False)

    def idle(self):
        if self.active():
            raise ValueError("已有后台操作，完成或停止后才能修改环境与设置")
        if (Path(self.config["output_root"]) / ".run.lock").exists():
            raise ValueError("输出目录仍有分割运行锁；请先核查旧任务，面板不会删除锁")

    def configure(self, payload):
        allowed = {"manifest", "output_root", "device", "threads", "torch_build", "gpu_confirmed"}
        if not isinstance(payload, dict) or set(payload) != allowed:
            raise ValueError("设置字段不完整或包含未知字段")
        with file_lock(self.workspace / "control.lock"):
            self.idle()
            config = dict(payload)
            for key in ("manifest", "output_root"):
                if not isinstance(config[key], str) or not config[key].strip():
                    if key == "manifest" and config[key] == "":
                        continue
                    raise ValueError("请填写服务器上的路径")
                config[key] = str(Path(config[key]).expanduser().resolve())
            if not isinstance(config["device"], str) or not re.fullmatch(r"cpu|gpu:[0-9]+", config["device"]):
                raise ValueError("请选择 CPU 或可见 GPU 编号")
            if type(config["threads"]) is not int or not 1 <= config["threads"] <= 64:
                raise ValueError("CPU 线程应为 1–64")
            if config["torch_build"] not in BUILDS or type(config["gpu_confirmed"]) is not bool:
                raise ValueError("安装方案或 GPU 确认字段无效")
            config["gpu_uuid"] = None
            if config["device"] != "cpu":
                info = read_json(self.workspace / "environment.json", {})
                index = int(config["device"].split(":")[1])
                config["gpu_uuid"] = next((d["uuid"] for d in info.get("devices", []) if d["index"] == index), None)
                check_device(config, info)
            cases = validate_paths(config, self.workspace)
            write_json(self.config_path, config)
        return {"saved": True, "count": len(cases), "note": "设置已保存"}

    def import_manifest(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"csv"} or not isinstance(payload["csv"], str):
            raise ValueError("请上传 CSV 清单")
        with file_lock(self.workspace / "control.lock"):
            self.idle()
            path = self.workspace / "manifests" / (uuid.uuid4().hex + ".csv")
            # Relative paths from a local upload are ambiguous on the server.
            rows = list(csv.DictReader(io.StringIO(payload["csv"].lstrip("\ufeff"))))
            if not rows or any(not Path(r.get("image", "")).is_absolute() for r in rows):
                raise ValueError("上传的清单必须包含服务器绝对影像路径")
            write_private(path, payload["csv"])
            config = dict(self.config, manifest=str(path))
            try:
                cases = validate_paths(config, self.workspace)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            write_json(self.config_path, config)
        return {"count": len(cases), "note": "清单已导入；请先预检服务器文件"}

    def selection(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"case_ids"}:
            raise ValueError("请求只能包含 case_ids")
        ids = payload["case_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError("请选择至少一个体积")
        return read_manifest(self.config["manifest"], ids)

    def submit(self, action, payload=None):
        if action not in {"install", "probe", "license", "plan", "run"}:
            raise ValueError("未知后台操作")
        with file_lock(self.workspace / "control.lock"):
            self.idle()
            config = self.config
            cases = self.selection(payload) if action in {"plan", "run"} else []
            if action in {"plan", "run"}:
                validate_paths(config, self.workspace)
            if action != "install" and not python_path(self.workspace).is_file():
                raise ValueError("请先安装独立环境")
            if action == "license":
                if payload and payload.get("license"):
                    stage_license(self.workspace, payload["license"])
                if not (self.workspace / "private" / "license.input").is_file():
                    raise ValueError("请先填写许可码，或导入含许可的私有配置")
            if action == "run":
                check_device(config, read_json(self.workspace / "environment.json", {}))
            job_id = uuid.uuid4().hex
            job = private_dir(self.workspace / "jobs" / job_id)
            if cases:
                stream = io.StringIO(newline="")
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(dict(r, hu_confirmed="true", spacing_confirmed="true") for r in cases)
                write_private(job / "manifest.csv", stream.getvalue())
            # Freeze the actual inference code so later git updates cannot alter a running batch.
            snapshot = private_dir(job / "code" / "aaoca_seg")
            for source in (CODE_ROOT / "aaoca_seg").glob("*.py"):
                shutil.copy2(source, snapshot / source.name)
            for name in ("run.py", "requirements.txt", "requirements-inference.txt"):
                shutil.copy2(CODE_ROOT / name, snapshot.parent / name)
            write_json(job / "request.json", {"action": action, "config": config, "count": len(cases)})
            write_json(job / "state.json", {"id": job_id, "action": action, "status": "queued", "started_at": timestamp()})
            command = [sys.executable, "-u", "-m", "aaoca_seg.worker", str(self.workspace), job_id]
            try:
                child = detached(command, job / "worker.log", runtime_env(self.workspace))
                self.children.append(child)
                write_json(job / "owner.json", process_identity(child.pid))
                write_json(self.workspace / "current.json", {"id": job_id})
            except Exception:
                write_json(job / "state.json", {"id": job_id, "action": action, "status": "failed", "error": "后台进程启动失败"})
                raise
        return {"submitted": True, "count": len(cases), "note": "已提交后台任务；断开 SSH 或关闭网页后继续运行"}

    def preflight(self, payload):
        return self.submit("plan", payload)

    def start(self, payload):
        return self.submit("run", payload)

    def stop(self):
        with file_lock(self.workspace / "control.lock"):
            job, state = self.job()
            if not state.get("active") or state.get("action") != "run":
                raise ValueError("没有可停止的粗分割任务；安装和检查请等待结束")
            write_private(job / "stop", "stop\n")
        return {"stop_requested": True, "note": "已请求停止分割；已完成结果保留"}

    def setup(self, action, payload):
        if action == "configure":
            return self.configure(payload)
        if action == "manifest":
            return self.import_manifest(payload)
        if action == "license" and isinstance(payload, dict) and set(payload) <= {"license"}:
            return self.submit(action, payload)
        if action in {"install", "probe"} and payload == {}:
            return self.submit(action)
        raise ValueError("操作或参数无效")

    def status(self):
        config = self.config
        job, operation = self.job()
        root = Path(config["output_root"])
        progress = read_json(root / "status.json", {})
        cases, error = [], None
        if config["manifest"]:
            try:
                cases = read_manifest(config["manifest"])
            except (ValueError, OSError) as exc:
                error = str(exc)
        current = {r["case_id"]: r for r in progress.get("cases", [])}
        rows = []
        for case in cases:
            record = read_json(root / "cases" / case["case_id"] / "latest.json", {})
            row = {"case_id": case["case_id"], "status": record.get("status", "pending"), "empty_masks": record.get("empty_masks", [])}
            row.update(current.get(case["case_id"], {}))
            if operation.get("active") and operation.get("action") == "run" and progress.get("current_case") == case["case_id"]:
                row["status"] = "running"
            rows.append(row)
        if progress.get("status") == "running" and not operation.get("active"):
            progress = dict(progress, status="interrupted_or_stale")
        info = read_json(self.workspace / "environment.json", {"versions": {}, "devices": [], "license_configured": False})
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置；需遵守服务器分配")
        info["license_configured"] = bool(read_json(self.workspace / "private" / "totalseg" / "config.json", {}).get("license_number"))
        return {"managed": True, "workspace": str(self.workspace), "config": config,
                "cases": rows, "manifest_error": error, "manifest": config["manifest"], "output_root": str(root),
                "device": config["device"], "threads": config["threads"], "environment": info,
                "operation": operation, "active": operation.get("active", False),
                "locked": (root / ".run.lock").exists(), "stop_requested": bool(job and (job / "stop").exists()),
                "license_pending": (self.workspace / "private" / "license.input").exists(),
                "progress": progress, "preflight": read_json(job / "plan.json") if job else None}

    def logs(self):
        job, state = self.job()
        text = tail(job / "worker.log") if job else "尚未提交后台操作。"
        if job:
            text += "\n" + tail(job / "operation.log")
        if state.get("action") == "run":
            root = Path(read_json(job / "request.json", {})["config"]["output_root"])
            progress = read_json(root / "status.json", {})
            if progress.get("model_log"):
                path = (root / progress["model_log"]).resolve()
                if path.is_relative_to(root.resolve() / "cases") and path.name in {f"{t}.log" for t in TASKS}:
                    text += f"\n\n{path.stem} 模型日志\n" + tail(path)
        return {"text": redact(text)}

    def host(self):
        return hardware(self.workspace)
