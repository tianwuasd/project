"""Upgrade an old panel with the new updater without following the old launcher."""

import argparse
import os
from pathlib import Path
import subprocess

from . import defaults, network, updater
from .state import Registry, dispatch_lock, read


def upgrade(runtime, checkout, emit=print):
    runtime, checkout = (
        Path(runtime).expanduser().resolve(),
        Path(checkout).expanduser().resolve(),
    )
    old_root = updater.managed_release(runtime)
    if old_root is None:
        old_root = checkout / "AAOCA"
        if not (old_root / "start_aaoca_server.py").is_file():
            raise ValueError("未找到 AAOCA 旧入口；--checkout 应为原 project 根目录")
    previous = read(runtime / "release.json", {})
    if previous.get("checkout") and Path(previous["checkout"]).resolve() != checkout:
        raise ValueError("指定 checkout 与原发布记录不同；请使用原项目目录")
    # The new updater must back up the code the old worker actually used,
    # not this temporary bootstrap package. All existing locks still apply.
    original_root = updater.CODE_ROOT
    original_env = os.environ.get("AAOCA_PREPROCESS_CHECKOUT")
    try:
        updater.CODE_ROOT = old_root
        os.environ["AAOCA_PREPROCESS_CHECKOUT"] = str(checkout)
        return updater.update(Registry(runtime), emit)
    finally:
        updater.CODE_ROOT = original_root
        if original_env is None:
            os.environ.pop("AAOCA_PREPROCESS_CHECKOUT", None)
        else:
            os.environ["AAOCA_PREPROCESS_CHECKOUT"] = original_env


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="一次性升级旧 AAOCA 预处理面板；不改共享 checkout 或独立分割"
    )
    parser.add_argument("--checkout", required=True, help="原 project Git 根目录")
    parser.add_argument("--runtime", default=defaults.runtime_default())
    parser.add_argument("--github-proxy", help="本机 HTTP 代理；none 清除")
    parser.add_argument("--github-transport", choices=network.TRANSPORTS)
    parser.add_argument("--github-ssh-key", help="密钥路径，不传入私钥内容")
    args = parser.parse_args(argv)
    runtime = Path(args.runtime).expanduser().resolve()
    try:
        values = dict(
            proxy="" if args.github_proxy == "none" else args.github_proxy,
            git_transport=args.github_transport,
            ssh_key=args.github_ssh_key,
        )
        if any(value is not None for value in values.values()):
            network.validate(
                {
                    **network.load(runtime),
                    **{k: v for k, v in values.items() if v is not None},
                }
            )
            with dispatch_lock(runtime):
                network.save(runtime, **values)
        print(upgrade(runtime, args.checkout)["message"])
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print("升级未完成：" + str(exc))
        return 1
