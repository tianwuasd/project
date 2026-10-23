"""Shared CLI for the Linux server and local validation."""
import argparse
import json
import signal
import sys

from .pipeline import environment, plan, require_backend, run_batch
from .schema import export_inventory, read_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="AAOCA 11 类粗分割与独立服务器面板")
    sub = parser.add_subparsers(dest="action", required=True)
    inventory = sub.add_parser("inventory", help="从研究总清单导出 AAOCA CSV，不扫描影像目录")
    inventory.add_argument("--inventory", required=True)
    inventory.add_argument("--manifest", required=True)
    inventory.add_argument("--data-root", help="服务器迁移后的研究根目录，内部含 AAOCA/")
    sub.add_parser("doctor", help="只读环境检查，不安装依赖、不验证许可或下载权重")
    for name in ("plan", "run", "panel"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output-root", required=True)
        command.add_argument("--device", default="gpu:0")
        command.add_argument("--threads", type=int, default=2)
        if name != "panel":
            command.add_argument("--case-id", action="append", dest="case_ids")
            command.add_argument("--limit", type=int)
        if name == "run":
            command.add_argument("--stop-file", help="本次任务的私有停止请求文件")
        if name == "panel":
            command.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    try:
        if args.action == "inventory":
            count = export_inventory(args.inventory, args.manifest, args.data_root)
            print(f"已导出 {count} 个 AAOCA 体积；清单含私人路径，请勿上传 GitHub")
            return 0
        if args.action == "doctor":
            print(json.dumps(environment(), ensure_ascii=False, indent=2))
            return 0
        if args.action == "panel":
            from .panel import serve
            serve(args.manifest, args.output_root, args.device, args.threads, args.port)
            return 0
        cases = read_manifest(args.manifest, args.case_ids, args.limit)
        if args.action == "plan":
            result = plan(cases, args.output_root, args.device, args.threads)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] == len(cases) else 2
        # Graceful termination preserves state and terminates only this task's child.
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        result = run_batch(cases, args.output_root, args.device, args.threads,
                           args.stop_file, require_backend())
        return 0 if result["status"] == "completed" else (130 if result["status"] == "stopped" else 2)
    except (ValueError, OSError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2
