"""Dependency-free launcher; only the private worker venv loads imaging packages."""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import time
import urllib.request

from .managed import ManagedController
from .operations import stage_license
from .panel import make_server
from .service_state import (CODE_ROOT, alive, detached, file_lock, private_dir,
                            process_identity, read_json, tail, write_json, write_private)


def describe(workspace):
    info = read_json(workspace / "service.json", {})
    running = alive(info.get("owner"))
    print("面板正在后台运行。" if running else "面板未运行；再次执行启动命令即可恢复查看。")
    if running:
        print(f"浏览器地址：http://127.0.0.1:{info['port']}（经 SSH / VS Code 端口转发）")
        print("访问令牌：" + (workspace / "private" / "panel.token").read_text(encoding="utf-8"))
    print(f"私有工作区：{workspace}")
    return running


def serve(workspace, port):
    with file_lock(workspace / "service.lock"):
        token_path = workspace / "private" / "panel.token"
        if not token_path.exists():
            write_private(token_path, secrets.token_urlsafe(32))
        controller = ManagedController(workspace)
        server, _ = make_server(controller, port, token_path.read_text(encoding="utf-8"))
        write_json(workspace / "service.json", {"owner": process_identity(), "port": server.server_port})
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass  # HTTP server shutdown never cancels an independent worker.
        finally:
            server.server_close()
            write_json(workspace / "service.json", {"owner": None, "port": server.server_port})


def import_profile(controller, path):
    path = Path(path).resolve()
    payload = read_json(path)
    if not isinstance(payload, dict) or set(payload) - {"manifest", "output_root", "device", "threads", "torch_build", "gpu_confirmed", "license"}:
        raise ValueError("私有配置格式无效")
    config = {k: v for k, v in controller.config.items() if k != "gpu_uuid"}
    config.update({k: v for k, v in payload.items() if k != "license"})
    for key in ("manifest", "output_root"):
        if config[key] and not Path(config[key]).is_absolute():
            config[key] = str(path.parent / config[key])
    controller.configure(config)
    if payload.get("license"):
        with file_lock(controller.workspace / "control.lock"):
            controller.idle()
            stage_license(controller.workspace, payload["license"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="AAOCA 一体化服务器面板：默认后台运行，SSH 断线后继续")
    parser.add_argument("--workspace", required=True, help="源码之外的私有运行目录")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--profile", help="可选私有 JSON 配置（不提交 Git）")
    parser.add_argument("--status", action="store_true", help="显示已有面板地址与访问令牌")
    parser.add_argument("--stop-panel", action="store_true", help="仅关闭面板，后台任务继续")
    parser.add_argument("--foreground", action="store_true", help="前台调试；生产运行请使用默认后台模式")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    try:
        if sys.version_info < (3, 11):
            raise ValueError("请用 Python 3.11 或更高版本启动")
        if not 1 <= args.port <= 65535:
            raise ValueError("端口应为 1–65535")
        if args.status:
            return 0 if describe(workspace) else 1
        if args.stop_panel:
            info = read_json(workspace / "service.json", {})
            if not alive(info.get("owner")):
                raise ValueError("面板未运行")
            token = (workspace / "private" / "panel.token").read_text(encoding="utf-8")
            request = urllib.request.Request(f"http://127.0.0.1:{info['port']}/api/panel-stop", data=b"{}",
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10).close()
            print("已请求关闭面板；后台安装与分割任务继续运行。")
            return 0
        controller = ManagedController(workspace)
        if args.serve:
            serve(workspace, args.port)
            return 0
        with file_lock(workspace / "launch.lock"):
            info = read_json(workspace / "service.json", {})
            if alive(info.get("owner")):
                describe(workspace)
                if args.profile:
                    print("已有面板运行，未重新导入配置；请在现有页面修改设置。")
                return 0
            if args.profile:
                import_profile(controller, args.profile)
            if args.foreground:
                print("前台调试模式；可用 --status 查询令牌。", flush=True)
                serve(workspace, args.port)
                return 0
            child = detached([sys.executable, "-u", str(CODE_ROOT / "server.py"), "--workspace", str(workspace),
                              "--port", str(args.port), "--serve"], workspace / "service.log")
            for _ in range(100):
                info = read_json(workspace / "service.json", {})
                if alive(info.get("owner")):
                    describe(workspace)
                    return 0
                if child.poll() is not None:
                    raise ValueError("启动失败：" + tail(workspace / "service.log", 1800))
                time.sleep(0.1)
            raise ValueError("面板启动较慢，请用 --status 查看，或检查 service.log")
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
