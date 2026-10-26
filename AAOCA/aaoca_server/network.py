"""Non-secret, runtime-local GitHub routes; never mutate the parent environment."""

import json
import os
import shlex
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

HTTPS_UPSTREAM = "https://github.com/tianwuasd/project.git"
SSH_UPSTREAM = "ssh://git@ssh.github.com:443/tianwuasd/project.git"
TRANSPORTS = ("github-https", "github-ssh")
DEFAULT = dict(schema_version=1, proxy="", git_transport="github-https", ssh_key="")
_active = ContextVar("aaoca_github_network", default=None)


def validate(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT):
        raise ValueError("网络配置字段无效；仅保存代理地址、Git 线路和密钥路径")
    result = {**DEFAULT, **value}
    if result["schema_version"] != 1 or result["git_transport"] not in TRANSPORTS:
        raise ValueError("网络配置版本或 Git 线路无效")
    proxy, key = result["proxy"], result["ssh_key"]
    if not isinstance(proxy, str) or not isinstance(key, str):
        raise ValueError("代理地址和密钥路径必须是文本")
    if any(ord(c) < 32 for c in proxy + key) or "%" in key:
        raise ValueError("网络配置包含不支持的控制字符或路径占位符")
    if proxy:
        try:
            url = urlsplit(proxy)
            valid = (
                url.scheme == "http"
                and url.hostname in {"127.0.0.1", "localhost", "::1"}
                and url.port is not None
                and 1 <= url.port <= 65535
                and url.username is None
                and url.password is None
                and not url.path
                and not url.query
                and not url.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                "代理请填写本机 HTTP 地址，如 http://127.0.0.1:17891；不含账号或订阅"
            )
    if result["git_transport"] == "github-ssh" and not key:
        raise ValueError("官方 SSH 线路需要指定部署密钥路径（不读取密钥内容）")
    return result


def load(runtime):
    path = Path(runtime) / "network.json"
    if not path.exists():
        return dict(DEFAULT)
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        raise ValueError(
            "无法读取 runtime/network.json；请先在原 runtime 备份并改名该文件，再重新配置网络"
        ) from error


def save(runtime, **values):
    settings = validate(
        {**load(runtime), **{k: v for k, v in values.items() if v is not None}}
    )
    directory = Path(runtime)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "network.json"
    temporary = directory / (".network-" + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            if os.name == "posix":
                os.chmod(temporary, 0o600)
            json.dump(settings, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return settings


@contextmanager
def use(settings):
    token = _active.set(validate(settings))
    try:
        yield
    finally:
        _active.reset(token)


def current():
    return dict(_active.get() or DEFAULT)


def child_env(*, ssh=False):
    env = dict(os.environ)
    settings = current()
    if settings["proxy"]:
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            env[name] = settings["proxy"]
        env["NO_PROXY"] = env["no_proxy"] = ""
    if ssh and settings["git_transport"] == "github-ssh":
        key = Path(settings["ssh_key"]).expanduser().resolve()
        if not key.is_file():
            raise ValueError("SSH 密钥路径不存在；请检查网络设置，不要粘贴私钥内容")
        command = [
            "ssh",
            "-F",
            os.devnull,
            "-i",
            str(key),
            "-p",
            "443",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
        ]
        if settings["proxy"]:
            nc = shutil.which("nc")
            if not nc:
                raise ValueError(
                    "SSH 经 HTTP 代理需要 OpenBSD nc（支持 -X connect -x）；当前未找到 nc"
                )
            parsed = urlsplit(settings["proxy"])
            host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
            tunnel = shlex.join(
                [nc, "-X", "connect", "-x", f"{host}:{parsed.port}", "%h", "%p"]
            )
            command += ["-o", "ProxyCommand=" + tunnel]
        env["GIT_SSH_COMMAND"] = shlex.join(command)
        env["GIT_SSH_VARIANT"] = "ssh"
    return env


def routes(settings):
    settings = validate(settings)
    return (
        (SSH_UPSTREAM,)
        if settings["git_transport"] == "github-ssh"
        else (HTTPS_UPSTREAM,)
    )


def show(settings, emit=print):
    emit("GitHub 网络设置（仅用于 AAOCA 预处理代码更新）")
    emit("代理：" + (settings["proxy"] or "未显式设置，沿用进程网络环境"))
    emit("官方线路：" + settings["git_transport"])
    if settings["git_transport"] == "github-ssh":
        emit("SSH 密钥路径：" + settings["ssh_key"] + "；目标 ssh.github.com:443")
    emit("私有仓库不使用公共代码镜像。HTTPS 沿用已有凭据助手；不在面板保存密码或令牌。")
    emit("保存不代表连接成功；不修改全局代理、Python 安装镜像或后台任务。")
