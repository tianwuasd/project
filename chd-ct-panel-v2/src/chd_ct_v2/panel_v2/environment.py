"""Direct-first installation; one configured-proxy retry and real API probes."""

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from ..server import network

PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def install_source(root, base):
    from uuid import uuid4

    destination = Path(base) / "build" / uuid4().hex
    # setuptools writes egg-info and build files into its source tree.
    shutil.copytree(root, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def verify_versions(target, log):
    from ..recovery import read_json, write_json

    report = read_json(Path(log).parent / "dependency-probe.json")
    current = {name: record.get("version") for name, record in report["packages"].items()}
    current["python"] = report["python"]
    current["installed_distributions"] = report.get("installed_distributions", {})
    receipt = Path(target) / ".ct-v2-dependency-versions.json"
    if receipt.exists() and read_json(receipt) != current:
        raise ValueError("该冻结代码的依赖版本已改变，拒绝静默续跑；保留原环境记录并另建环境核验")
    if not receipt.exists():
        write_json(receipt, current)


def routes(env):
    direct = {k: v for k, v in env.items() if k not in PROXIES}
    direct.update(
        PIP_DEFAULT_TIMEOUT="30",
        PIP_RETRIES="1",
        CONDA_REMOTE_CONNECT_TIMEOUT_SECS="15",
        CONDA_REMOTE_READ_TIMEOUT_SECS="60",
        CONDA_REMOTE_MAX_RETRIES="1",
    )
    yield "直连", direct
    proxy = network.load(env["CHD_V2_RUNTIME"])["proxy"]
    if proxy:
        yield (
            "本V2配置代理（仅重试一次）",
            dict(direct, **{k: proxy for k in PROXIES}, NO_PROXY="", no_proxy=""),
        )


def checked(command, env, log):
    from ..server.bootstrap import run_logged

    for label, child in routes(env):
        print("环境安装线路：" + label, flush=True)
        code = run_logged(command, child, log)
        if not code:
            return
    raise RuntimeError(f"环境安装失败，保留现场：{log}")


def download(url, target, env):
    error = None
    for label, child in routes(env):
        print("环境下载线路：" + label, flush=True)
        proxies = (
            {"https": child["HTTPS_PROXY"], "http": child["HTTP_PROXY"]} if "HTTPS_PROXY" in child else {}
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        try:
            with opener.open(url, timeout=60) as response, target.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            return
        except OSError as problem:
            error = problem
    raise RuntimeError("环境下载失败；partial文件保留") from error


def probe(python, env, log):
    try:
        result = subprocess.run(
            [str(python), "-B", "-m", "chd_ct_v2.quickstart.environment", "--probe"],
            env=dict(env, CUDA_VISIBLE_DEVICES=""),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        report = (
            json.loads(result.stdout)
            if not result.returncode
            else {"ok": False, "errors": [result.stderr[-8000:]]}
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report = {"ok": False, "errors": [str(error)]}
    from ..recovery import write_json

    write_json(Path(log).parent / "dependency-probe.json", report)
    return report.get("ok", False)
