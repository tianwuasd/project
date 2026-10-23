"""Loopback-only panel. No shell execution, arbitrary path API or foreign PID stop."""
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import uuid

from .pipeline import CODE_ROOT, environment, plan, require_backend, validate_device, validate_output
from .schema import FIELDS, TASKS, read_manifest


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def tail(path, size=24000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def gpu_info():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
        return result.stdout.strip() if result.returncode == 0 else "GPU 查询不可用"
    except (OSError, subprocess.TimeoutExpired):
        return "未检测到 nvidia-smi；可使用 CPU，服务器 GPU 状态待核查"


class Controller:
    def __init__(self, manifest, root, device="gpu:0", threads=2):
        self.manifest = Path(manifest).resolve()
        self.root = validate_output(root, read_manifest(self.manifest))
        self.device = validate_device(device)
        if not 1 <= threads <= 64:
            raise ValueError("threads 必须在 1–64 之间")
        self.threads = threads
        self.process = None
        self.log_path = None
        self.stop_file = None
        self.mutex = threading.Lock()

    def active(self):
        return self.process is not None and self.process.poll() is None

    def selection(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"case_ids"}:
            raise ValueError("请求只能包含 case_ids")
        ids = payload["case_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            raise ValueError("请选择至少一个病例")
        return read_manifest(self.manifest, ids)

    def preflight(self, payload):
        return plan(self.selection(payload), self.root, self.device, self.threads)

    def start(self, payload):
        cases = self.selection(payload)
        with self.mutex:
            if self.active() or (self.root / ".run.lock").exists():
                raise ValueError("此输出目录已有任务或遗留锁，请先核查")
            require_backend()
            check = plan(cases, self.root, self.device, self.threads)
            if check["ready"] != len(cases):
                raise ValueError("部分病例预检未通过，请先查看预检结果")
            private = self.root / "panel"
            private.mkdir(parents=True, exist_ok=True)
            job = uuid.uuid4().hex
            frozen_manifest = private / f"{job}.csv"
            with frozen_manifest.open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(dict(row, spacing_confirmed="true", hu_confirmed="true") for row in cases)
            self.log_path = private / f"{job}.log"
            self.stop_file = private / f"{job}.stop"
            command = [sys.executable, "-u", str(CODE_ROOT / "run.py"), "run", "--manifest", str(frozen_manifest),
                       "--output-root", str(self.root), "--device", self.device, "--threads", str(self.threads),
                       "--stop-file", str(self.stop_file)]
            with self.log_path.open("wb") as log:
                self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                cwd=str(CODE_ROOT), start_new_session=os.name != "nt")
            return {"submitted": True, "count": len(cases), "note": "已提交；完成状态以任务记录为准"}

    def stop(self):
        with self.mutex:
            if not self.active():
                raise ValueError("没有本面板正在运行的任务；不会停止其他进程")
            self.stop_file.touch(exist_ok=True)
            return {"stop_requested": True, "note": "正在停止本任务；输入校验或合并阶段会在下一安全检查点响应"}

    def status(self):
        cases = read_manifest(self.manifest)
        progress = read_json(self.root / "status.json", {})
        current = {r["case_id"]: r for r in progress.get("cases", [])}
        rows = []
        for case in cases:
            record = read_json(self.root / "cases" / case["case_id"] / "latest.json", {})
            row = {"case_id": case["case_id"], "status": record.get("status", "pending"),
                   "attempt": record.get("attempt"), "empty_masks": record.get("empty_masks", [])}
            row.update(current.get(case["case_id"], {}))
            if self.active() and progress.get("current_case") == case["case_id"]:
                row["status"] = "running"
            rows.append(row)
        # A running file is historical if no worker lock exists. Never call it completed.
        if progress.get("status") == "running" and not self.active() and not (self.root / ".run.lock").exists():
            progress = dict(progress, status="interrupted_or_stale")
        return {"cases": rows, "progress": progress, "active": self.active(),
                "locked": (self.root / ".run.lock").exists(),
                "stop_requested": bool(self.stop_file and self.stop_file.exists()),
                "exit_code": self.process.poll() if self.process else None,
                "manifest": str(self.manifest), "output_root": str(self.root),
                "device": self.device, "threads": self.threads, "environment": environment()}

    def logs(self):
        text = tail(self.log_path) if self.log_path else "本面板尚未提交任务。"
        progress = read_json(self.root / "status.json", {})
        if progress.get("model_log"):
            path = (self.root / progress["model_log"]).resolve()
            if path.is_relative_to(self.root / "cases") and path.name in {f"{task}.log" for task in TASKS}:
                text += f"\n\n--- {path.stem} 模型日志 ---\n" + tail(path)
        return {"text": text}


def make_server(controller, port=8787, token=None):
    token = token or secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Tokens, private paths and case IDs must not enter access logs.

        def send(self, value, code=200, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not secrets.compare_digest(supplied.encode("utf-8"), ("Bearer " + token).encode("utf-8")):
                self.send({"error": "请填写服务器启动时显示的访问令牌"}, 401)
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in static:
                filename, mime = static[path]
                self.send((CODE_ROOT / "web" / filename).read_bytes(), content_type=mime)
                return
            if not self.authorized():
                return
            try:
                if path == "/api/status":
                    self.send(controller.status())
                elif path == "/api/logs":
                    self.send(controller.logs())
                elif path == "/api/gpu":
                    self.send({"text": gpu_info()})
                else:
                    self.send({"error": "接口不存在"}, 404)
            except Exception as error:
                self.send({"error": str(error)}, 400)

        def do_POST(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("请求长度无效")
                payload = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/plan":
                    self.send(controller.preflight(payload))
                elif path == "/api/start":
                    self.send(controller.start(payload))
                elif path == "/api/stop" and payload == {}:
                    self.send(controller.stop())
                else:
                    self.send({"error": "接口或参数无效"}, 404)
            except Exception as error:
                self.send({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server, token


def serve(manifest, root, device="gpu:0", threads=2, port=8787):
    if not 1 <= port <= 65535:
        raise ValueError("端口必须在 1–65535 之间")
    controller = Controller(manifest, root, device, threads)
    server, token = make_server(controller, port)
    print(f"AAOCA 独立粗分割面板：http://127.0.0.1:{port}", flush=True)
    print(f"访问令牌（仅在本机/SSH 终端使用）：{token}", flush=True)
    print("请经 SSH 隧道访问；关闭浏览器不停止任务。退出面板前先在页面停止正在运行的任务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if controller.active():
            controller.stop()
            print("已请求停止本面板任务；退出后请检查输出目录的状态和锁。", flush=True)
    finally:
        server.server_close()
