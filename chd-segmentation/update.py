import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from time import monotonic
import tomllib
from uuid import uuid4
from urllib.parse import urlsplit


REPOSITORY = "tianwuasd/project"
SUBDIRECTORY = "chd-segmentation"
OFFICIAL_URLS = {"https://github.com/tianwuasd/project.git", "https://github.com/tianwuasd/project",
                 "git@github.com:tianwuasd/project.git", "ssh://git@ssh.github.com:443/tianwuasd/project.git"}
ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "pyproject.toml", "run.py", "smoke.py", "setup_env.py",
              "update.py", "launcher.py"}
SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt"}
PROTOCOL = 1
DEFAULT_NETWORK = {"mirrors": ["https://ghfast.top", "https://gh-proxy.com"], "probe_timeout_seconds": 10}


def public_environment():
    env = dict(os.environ)
    for name in list(env):
        if name in {"GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_DIR", "GIT_WORK_TREE", "GIT_SSL_NO_VERIFY"} or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            env.pop(name)
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
                "GIT_NO_LAZY_FETCH": "1"})
    return env


def probe_source(url, timeout):
    started = monotonic()
    env = {**public_environment(), "GIT_TRACE_PACKET": "1"}
    command = ["git", "-c", "protocol.version=2", "-c", "credential.helper=", "-c", "http.extraHeader=",
               "-c", "http.sslVerify=true", "-c", "http.followRedirects=false", "ls-remote", "--exit-code", url, "refs/heads/main"]
    # 用户要求允许逐项探测镜像；探测错误保留在结果中，下载与内容校验错误直接抛出。
    try:
        result = subprocess.run(command, cwd=Path.cwd().anchor, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"url": url, "ok": False, "seconds": monotonic() - started, "error": "probe_timeout"}
    rows = result.stdout.decode("utf-8", "replace").splitlines()
    valid = result.returncode == 0 and len(rows) == 1 and re.fullmatch(r"[0-9a-f]{40}\s+refs/heads/main", rows[0])
    filtered = bool(re.search(rb"fetch=[^\r\n]*\bfilter\b", result.stderr))
    return {"url": url, "ok": bool(valid and filtered), "sha": rows[0].split()[0] if valid else None,
            "seconds": monotonic() - started, "partial_fetch": filtered,
            "error": None if valid and filtered else "partial_fetch_not_supported" if valid else "git_probe_failed"}


def mirror_url(prefix):
    parsed = urlsplit(prefix)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("镜像前缀必须是无认证参数的 HTTPS 地址")
    return prefix.rstrip("/") + "/https://github.com/" + REPOSITORY + ".git"


def network_settings(path):
    value = read_json(path) if Path(path).exists() else dict(DEFAULT_NETWORK)
    if set(value) != {"mirrors", "probe_timeout_seconds"} or not isinstance(value["mirrors"], list):
        raise ValueError("镜像配置字段无效")
    if not 1 <= len(value["mirrors"]) <= 4 or not 1 <= value["probe_timeout_seconds"] <= 30:
        raise ValueError("镜像数量或探测超时超出范围")
    for prefix in value["mirrors"]:
        mirror_url(prefix)
    return value


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def journal(runtime, phase, **fields):
    path = Path(runtime) / "updates" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= 1024 ** 2:
        os.replace(path, path.with_suffix(".jsonl.1"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, **fields},
                                ensure_ascii=False) + "\n")


@contextmanager
def file_lock(path, shared=False):
    path = Path(path).resolve()
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
        yield


def deployment_lock(runtime, shared=False):
    return file_lock(Path(runtime) / ".code.lock", shared)


def allowed(path):
    parts = PurePosixPath(path).parts
    return (bool(parts) and not path.startswith("/") and ".." not in parts and "\\" not in path
            and ((len(parts) == 1 and path in ROOT_FILES)
                 or (parts[0] in {"src", "configs", "docs", "tests"} and PurePosixPath(path).suffix in SUFFIXES)))


def require_head(expected, actual):
    if expected != actual:
        raise ValueError("GitHub main 已变化，固定目标与远端不一致；保留当前生效版本")


class GitCache:
    def __init__(self, path, expected_urls=None, timeout=900):
        self.path = Path(path).resolve(strict=True)
        self.urls = OFFICIAL_URLS if expected_urls is None else set(expected_urls)
        self.deadline = monotonic() + timeout
        self.transport = "origin"
        self.network_info = None
        self.public_options = []

    def run(self, *args, data=None, allow_absent=False, network=False):
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("代码同步超过总时间限制")
        env = public_environment() if self.network_info is not None else {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
        offline = [] if network else ["-c", "protocol.allow=never"]
        result = subprocess.run(["git", "-C", str(self.path), *offline, "-c", "http.followRedirects=false",
                                 *self.public_options,
                                 "-c", "remote.origin.promisor=true", "-c", "remote.origin.partialCloneFilter=blob:none",
                                 *args], input=data, stdout=subprocess.PIPE, env=env, timeout=remaining, check=False)
        if not (allow_absent and result.returncode == 1):
            result.check_returncode()
        return result.stdout

    def choose_source(self, candidates, *, expected=None, timeout=10, official=None):
        with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as pool:
            results = list(pool.map(lambda url: probe_source(url, timeout), candidates))
        official_result = next((row for row in results if row["url"] == official and row.get("sha")), None)
        reference = expected if expected is not None else official_result["sha"] if official_result else None
        good = [row for row in results if row["ok"] and (reference is None or row["sha"] == reference)]
        if not good:
            raise ConnectionError("所有候选均不可用、提交不一致或不支持按需下载：" + json.dumps(results, ensure_ascii=False))
        if reference is None and len({row["sha"] for row in good}) != 1:
            raise ValueError("候选镜像的 main 提交不一致，停止自动选择")
        mirrors = [row for row in good if row["url"] != official]
        selected = min(mirrors or good, key=lambda row: row["seconds"])
        # 清空本地仓库中可能存在的 HTTP 认证头、cookie 与凭据助手。
        keys = self.run("config", "--includes", "--name-only", "--get-regexp",
                        r"^(http\..*(extraheader|cookiefile|savecookies)|credential\..*helper)$", allow_absent=True).decode().splitlines()
        self.public_options = ["-c", "credential.helper=", "-c", "http.extraHeader=", "-c", "http.cookieFile=",
                               "-c", "http.saveCookies=false", "-c", "http.sslVerify=true"]
        for key in keys:
            self.public_options.extend(["-c", key + "="])
        self.transport = selected["url"]
        self.network_info = {"source": self.transport, "probes": results,
                             "reference": "pinned" if expected is not None else "github" if official_result else "mirror_agreement"}
        print("本次下载来源：" + self.transport, flush=True)
        return selected["sha"]

    def validate_origin(self):
        if self.run("remote", "get-url", "origin").decode().strip() not in self.urls:
            raise ValueError("Git 缓存 origin 不属于已确认的代码来源")
        if self.run("config", "--get-regexp", r"^url\..*\.insteadof$", allow_absent=True):
            raise ValueError("存在 Git URL 重写，拒绝通过替代地址传递认证")

    def remote_head(self):
        rows = self.run("ls-remote", "--exit-code", self.transport, "refs/heads/main", network=True).decode().splitlines()
        if len(rows) != 1:
            raise ValueError("不能唯一确定远端 main")
        return rows[0].split()[0]

    def inventory(self, commit):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("需要完整的 GitHub 提交 SHA")
        records = self.run("ls-tree", "-r", "-z", commit + ":" + SUBDIRECTORY).split(b"\0")
        result = {}
        for record in filter(None, records):
            header, raw_path = record.split(b"\t", 1)
            mode, kind, blob = header.decode().split()
            path = raw_path.decode("utf-8")
            if mode != "100644" or kind != "blob" or not allowed(path):
                raise ValueError("发布范围含不允许的路径或文件类型：" + path)
            result[path] = blob
        if not {"run.py", "setup_env.py", "pyproject.toml", "launcher.py", "update.py"} <= result.keys() or len(result) > 200:
            raise ValueError("发布内容缺少必要入口或超出本项目文件预算")
        return result

    def obtain(self, commit):
        self.validate_origin()
        require_head(commit, self.remote_head())
        print("正在获取本项目提交与目录信息", flush=True)
        self.run("fetch", "--no-tags", "--filter=blob:none", self.transport, "main", network=True)
        require_head(commit, self.run("rev-parse", "FETCH_HEAD").decode().strip())
        inventory = self.inventory(commit)
        local = set(self.run("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines())
        missing = sorted(set(inventory.values()) - local)
        print("本项目文件：%d，需获取文件对象：%d" % (len(inventory), len(missing)), flush=True)
        for start in range(0, len(missing), 64):
            data = ("\n".join(missing[start:start + 64]) + "\n").encode()
            self.run("-c", "fetch.negotiationAlgorithm=noop", "fetch", self.transport, "--no-tags",
                     "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "--stdin", data=data, network=True)
        require_head(commit, self.remote_head())
        return inventory


def inspect_release(code, pointer):
    code = Path(code).resolve()
    commit = pointer["sha"]
    if pointer["protocol"] != PROTOCOL or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("代码指针协议或提交无效")
    release = code / "code_versions" / commit
    manifest = release / "release.manifest.json"
    if (release.is_symlink() or not release.resolve().is_relative_to(code)
            or manifest.is_symlink() or sha256(manifest) != pointer["manifest_sha256"]):
        raise ValueError("代码版本路径或清单摘要不符")
    record = read_json(manifest)
    if record["sha"] != commit or record["repository"] != REPOSITORY or record["subdirectory"] != SUBDIRECTORY:
        raise ValueError("代码来源身份不符")
    if not all(allowed(name) for name in record["files"]):
        raise ValueError("已安装版本清单包含非法路径")
    observed = set()
    for path in release.rglob("*"):
        if path.is_symlink():
            raise ValueError("代码版本包含符号链接")
        if path.is_file() and path != manifest:
            name = path.relative_to(release).as_posix()
            if name not in record["files"] or sha256(path) != record["files"][name]["sha256"]:
                raise ValueError("已安装代码被修改：" + name)
            observed.add(name)
    if observed != set(record["files"]):
        raise ValueError("代码文件缺失")
    return release, record


def install(cache, code, runtime, commit, *, activate=False, python=None):
    started = monotonic()
    code, runtime = Path(code).resolve(), Path(runtime).resolve()
    if code == runtime or code.is_relative_to(runtime) or runtime.is_relative_to(code) or cache.path.is_relative_to(code):
        raise ValueError("代码、过程目录与 Git 缓存必须相互独立")
    if (code / "code_versions").is_symlink() or (runtime / "updates").is_symlink():
        raise ValueError("代码版本或更新暂存目录不能经符号链接指向其他位置")
    code.mkdir(parents=True, exist_ok=True)
    if any(code.iterdir()) and not (code / "deployment.json").is_file():
        raise FileExistsError("目标包含未登记文件，请使用独立的新项目代码目录")
    if (code / "deployment.json").exists():
        registration = read_json(code / "deployment.json")
        if registration != {"protocol": PROTOCOL, "repository": REPOSITORY, "runtime": str(runtime)}:
            raise ValueError("目标代码目录已登记给其他运行区或协议")
    if activate and python is None:
        raise ValueError("激活前需要指定本项目实际环境的 --python 解释器")
    inventory = cache.inventory(commit)
    with deployment_lock(runtime), ExitStack() as locks:
        journal(runtime, "install_started", sha=commit)
        # 原有入口的训练锁也参与更新准入；受管入口同时持有共享代码锁。
        for path in sorted((runtime / "runs").rglob(".run.lock")):
            locks.enter_context(file_lock(path))
        pointer_path = code / "current.json"
        installed_path = code / "installed.json"
        installed = read_json(installed_path) if installed_path.exists() else None
        installed_release = inspect_release(code, installed)[0] if installed else None
        if pointer_path.exists():
            previous = read_json(pointer_path)
            inspect_release(code, previous)
            cache.run("merge-base", "--is-ancestor", previous["sha"], commit)
        versions = code / "code_versions"
        versions.mkdir(exist_ok=True)
        release = versions / commit
        stage = runtime / "updates" / ("stage-" + uuid4().hex)
        candidate = versions / (".candidate-" + uuid4().hex)
        stage.mkdir(parents=True)
        write_json(stage / "owner.json", {"commit": commit, "pid": os.getpid()})
        write_json(code / "deployment.json", {"protocol": PROTOCOL, "repository": REPOSITORY, "runtime": str(runtime)})
        try:
            if release.exists():
                manifest = release / "release.manifest.json"
                pointer = {"protocol": PROTOCOL, "sha": commit, "manifest_sha256": sha256(manifest)}
                _, record = inspect_release(code, pointer)
                if {name: row["git_blob"] for name, row in record["files"].items()} != inventory:
                    raise ValueError("已有版本与目标 Git 提交不一致")
            else:
                archive = cache.run("archive", "--format=tar", commit + ":" + SUBDIRECTORY)
                if len(archive) > 16 * 1024 ** 2:
                    raise ValueError("代码包超过本项目容量预算")
                files, payload = {}, stage / "code"
                payload.mkdir()
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
                    for member in bundle.getmembers():
                        if member.isdir():
                            continue
                        name = member.name
                        if not member.isfile() or name not in inventory or name in files or not allowed(name):
                            raise ValueError("代码包包含越界、重复或未知文件")
                        with bundle.extractfile(member) as stream:
                            data = stream.read()
                        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                        if blob != inventory[name]:
                            raise ValueError("代码文件与 Git 对象摘要不一致：" + name)
                        path = payload / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                        if path.suffix == ".py":
                            compile(data, str(path), "exec")
                        files[name] = {"git_blob": blob, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                if set(files) != set(inventory):
                    raise ValueError("代码包不完整")
                with (payload / "pyproject.toml").open("rb") as stream:
                    project = tomllib.load(stream)["project"]
                record = {"protocol": PROTOCOL, "repository": REPOSITORY, "subdirectory": SUBDIRECTORY,
                          "sha": commit, "version": project["version"], "files": files}
                write_json(payload / "release.manifest.json", record)
                shutil.copytree(payload, candidate)
                candidate.rename(release)
                pointer = {"protocol": PROTOCOL, "sha": commit,
                           "manifest_sha256": sha256(release / "release.manifest.json")}
                inspect_release(code, pointer)
            for source, destination in (("update.py", "update.py"), ("launcher.py", "run.py")):
                target = code / destination
                if target.exists() and target.read_bytes() != (release / source).read_bytes():
                    if source != "update.py" or installed_release is None or target.read_bytes() != (installed_release / source).read_bytes():
                        raise ValueError("根入口存在未知修改或协议变化；当前生效版本保持不变")
                    pending = target.with_suffix(".py.pending")
                    pending.write_bytes((release / source).read_bytes())
                    os.replace(pending, target)
                if not target.exists():
                    target.write_bytes((release / source).read_bytes())
            pointer.update({"repository": REPOSITORY, "branch": "main", "subdirectory": SUBDIRECTORY,
                            "runtime": str(runtime), "code": str(code), "version": record["version"]})
            write_json(installed_path, pointer)
            journal(runtime, "code_installed", sha=commit, files=len(inventory))
            if activate:
                # 保留环境内解释器入口与 pyvenv.cfg 的关联。
                executable = Path(os.path.abspath(Path(python).expanduser()))
                if not executable.is_file():
                    raise FileNotFoundError("环境解释器不存在：" + str(executable))
                check = subprocess.run([str(executable), "-B", str(release / "setup_env.py"), "--check"],
                                       stdout=subprocess.PIPE, timeout=120, check=True,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
                environment = json.loads(check.stdout)
                if environment["state"] != "passed":
                    raise ValueError("环境检查未通过，当前代码指针保持不变")
                pointer["python"] = str(executable)
                write_json(pointer_path, pointer)
                journal(runtime, "activated", sha=commit, python=str(executable))
            active = read_json(pointer_path)["sha"] if pointer_path.exists() else None
            receipt = {**pointer, "state": "activated" if activate else "installed_current" if active == commit else "installed_not_activated",
                       "network": cache.network_info,
                       "active_sha": active,
                       "file_count": len(inventory), "code_bytes": sum(row["bytes"] for row in record["files"].values()),
                       "seconds": monotonic() - started, "checked_at": datetime.now(timezone.utc).isoformat()}
            write_json(runtime / "updates" / "latest.json", receipt)
            return receipt
        finally:
            if sys.exception() is not None:
                journal(runtime, "install_failed", sha=commit, error=str(sys.exception())[:2000])
            # 仅清理本次创建且位于指定父目录中的暂存；已安装代码与原始缓存均保留。
            for owned, parent in ((stage, runtime / "updates"), (candidate, versions)):
                if owned.exists():
                    if owned.is_symlink() or owned.resolve().parent != parent.resolve():
                        raise ValueError("更新暂存归属异常，停止清理")
                    shutil.rmtree(owned)


def main():
    parser = argparse.ArgumentParser(description="只同步先心分割代码，校验后安装独立版本")
    parser.add_argument("--git-cache", default="/data4/zhangruichen/code/git")
    parser.add_argument("--code", default="/data4/zhangruichen/code/chd-segmentation")
    parser.add_argument("--runtime", default="/data4/zhangruichen/course/chd-segmentation")
    parser.add_argument("--sha", help="完整目标提交；省略时在开始同步时固定 main")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--python")
    parser.add_argument("--mirror", action="append", help="auto、direct 或镜像 HTTPS 前缀；可重复指定")
    parser.add_argument("--network-config", help="镜像 JSON 配置；默认使用过程目录下的 update_network.json")
    args = parser.parse_args()
    cache = GitCache(args.git_cache)
    cache.validate_origin()
    settings_path = Path(args.network_config) if args.network_config else Path(args.runtime) / "update_network.json"
    settings = network_settings(settings_path)
    choice = args.mirror or ["auto"]
    if choice == ["direct"]:
        commit = cache.remote_head() if args.sha is None else args.sha
    else:
        prefixes = settings["mirrors"] if choice == ["auto"] else choice
        candidates = [mirror_url(prefix) for prefix in prefixes] + ["https://github.com/" + REPOSITORY + ".git"]
        commit = cache.choose_source(candidates, expected=args.sha, timeout=settings["probe_timeout_seconds"], official=candidates[-1])
        write_json(Path(args.runtime) / "updates" / "network-latest.json", cache.network_info)
    if not settings_path.exists():
        write_json(settings_path, settings)
    if args.check:
        require_head(commit, cache.remote_head())
        pointer = Path(args.code) / "current.json"
        current = read_json(pointer) if pointer.exists() else None
        if current:
            inspect_release(args.code, current)
        result = {"remote_sha": commit, "current": current}
    else:
        journal(args.runtime, "download_started", sha=commit)
        try:
            cache.obtain(commit)
        finally:
            if sys.exception() is not None:
                journal(args.runtime, "download_failed", sha=commit, error=str(sys.exception())[:2000])
        result = install(cache, args.code, args.runtime, commit, activate=args.activate, python=args.python)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
