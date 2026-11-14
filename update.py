import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4


REPOSITORY = "tianwuasd/project"
SUBDIRECTORY = "chd-segmentation"
OFFICIAL = "https://github.com/" + REPOSITORY + ".git"
DEFAULT_NETWORK = {"mirrors": ["https://ghfast.top", "https://gh-proxy.com"], "probe_timeout_seconds": 10}
ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "pyproject.toml", "run.py", "smoke.py", "setup_env.py", "update.py", "launcher.py"}


def public_environment():
    env = dict(os.environ)
    for name in list(env):
        if name in {"GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_SSL_NO_VERIFY"} or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            env.pop(name)
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"})
    return env


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


@contextmanager
def file_lock(path, shared=False):
    path = Path(path)
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


@contextmanager
def code_guard(source):
    code = Path(source).resolve().parent
    repo = code if (code / ".git").is_dir() else code.parent
    if not (repo / ".git").is_dir():
        raise ValueError("请从已同步的 Git 工作目录运行项目")
    with file_lock(repo / ".git" / "chd-code.lock", shared=True):
        yield repo


def mirror_url(prefix):
    parsed = urlsplit(prefix)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("镜像前缀必须是无认证参数的 HTTPS 地址")
    return prefix.rstrip("/") + "/" + OFFICIAL


def network_settings(path):
    value = read_json(path) if Path(path).exists() else dict(DEFAULT_NETWORK)
    if set(value) != {"mirrors", "probe_timeout_seconds"} or not isinstance(value["mirrors"], list):
        raise ValueError("镜像配置字段无效")
    if not 1 <= len(value["mirrors"]) <= 4 or not 1 <= value["probe_timeout_seconds"] <= 30:
        raise ValueError("镜像数量或超时超出范围")
    for prefix in value["mirrors"]:
        mirror_url(prefix)
    return value


def probe_source(url, timeout):
    started = monotonic()
    command = ["git", "-c", "protocol.version=2", "-c", "credential.helper=", "-c", "http.extraHeader=", "-c", "http.sslVerify=true", "-c", "http.followRedirects=false", "ls-remote", "--exit-code", url, "refs/heads/main"]
    # 用户允许探测多个镜像；只在探测阶段记录超时并检查其他来源。
    try:
        result = subprocess.run(command, cwd=Path.cwd().anchor, env={**public_environment(), "GIT_TRACE_PACKET": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"url": url, "ok": False, "seconds": monotonic() - started, "error": "probe_timeout"}
    rows = result.stdout.decode("utf-8", "replace").splitlines()
    valid = result.returncode == 0 and len(rows) == 1 and re.fullmatch(r"[0-9a-f]{40}\s+refs/heads/main", rows[0])
    filtered = bool(re.search(rb"fetch=[^\r\n]*\bfilter\b", result.stderr))
    return {"url": url, "ok": bool(valid and filtered), "sha": rows[0].split()[0] if valid else None, "seconds": monotonic() - started, "partial_fetch": filtered}


def allowed(path):
    parts = PurePosixPath(path).parts
    return (bool(parts) and not path.startswith("/") and ".." not in parts and "\\" not in path
            and ((len(parts) == 1 and path in ROOT_FILES) or (parts[0] in {"src", "configs", "docs", "tests"} and PurePosixPath(path).suffix in {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt"})))


def require_head(expected, actual):
    if expected != actual:
        raise ValueError("远端 main 已变化，停止本次同步")


def prepare_code_root(path):
    root = Path(path).resolve(strict=True)
    metadata = root / "git" / ".git"
    entry = root / ".git"
    if not metadata.is_dir():
        raise FileNotFoundError("已有 Git 元数据目录不存在：" + str(metadata))
    if entry.exists() or entry.is_symlink():
        if not entry.is_dir() or entry.resolve() != metadata.resolve():
            raise ValueError("code 目录已有其他 Git 元数据入口，停止迁移")
    else:
        entry.symlink_to(metadata, target_is_directory=True)


class GitRepository:
    def __init__(self, path, expected_urls=None, timeout=900):
        self.path = Path(path).resolve(strict=True)
        self.urls = {OFFICIAL, OFFICIAL.removesuffix(".git")} if expected_urls is None else set(expected_urls)
        self.deadline = monotonic() + timeout
        self.transport = "origin"
        self.network_info = None
        self.public_options = []

    def run(self, *args, data=None, absent=False, network=False, index=None):
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("代码同步超过总时间限制")
        command = ["git", "-C", str(self.path), "-c", "http.followRedirects=false", "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false", "-c", "gc.auto=0", *self.public_options]
        if not network:
            command += ["-c", "protocol.allow=never"]
        env = public_environment() if self.network_info else {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
        env.pop("GIT_INDEX_FILE", None)
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        result = subprocess.run(command + list(args), input=data, stdout=subprocess.PIPE, env=env, timeout=remaining, check=False)
        if not (absent and result.returncode == 1):
            result.check_returncode()
        return result.stdout

    def validate(self, force=False):
        if self.run("remote", "get-url", "origin").decode().strip() not in self.urls:
            raise ValueError("origin 不属于指定的项目仓库")
        if self.run("config", "--get-regexp", r"^url\..*\.insteadof$", absent=True):
            raise ValueError("存在 Git URL 重写，请先移除本项目的重写配置")
        if Path(self.run("rev-parse", "--show-toplevel").decode().strip()).resolve() != self.path or not (self.path / ".git").is_dir():
            raise ValueError("需要独立的普通 Git 工作目录")
        if self.run("symbolic-ref", "--short", "HEAD").decode().strip() != "main":
            raise ValueError("当前分支必须为 main")
        code = self.path / SUBDIRECTORY
        if code.is_symlink() or (code.exists() and not code.is_dir()):
            raise ValueError("本项目代码路径必须为普通目录")
        initial = not self.run("ls-files", "-z")
        if initial and not force:
            if code.exists() and any(code.iterdir()):
                raise ValueError("本项目未初始化索引，代码目录包含已有文件")
        elif not force and (self.run("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", SUBDIRECTORY)
                            or self.run("ls-files", "--others", "-z", "--", SUBDIRECTORY)):
            raise ValueError("chd-segmentation 存在本地修改或未跟踪文件，停止更新")
        if code.exists() and any(path.is_symlink() for path in code.rglob("*")):
            raise ValueError("本项目代码目录包含符号链接")
        return initial

    def choose_source(self, candidates, *, expected=None, timeout=10, official=None):
        with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as pool:
            rows = list(pool.map(lambda url: probe_source(url, timeout), candidates))
        reference_row = next((row for row in rows if row["url"] == official and row.get("sha")), None)
        reference = expected if expected is not None else reference_row["sha"] if reference_row else None
        good = [row for row in rows if row["ok"] and (reference is None or row["sha"] == reference)]
        if not good or (reference is None and len({row["sha"] for row in good}) != 1):
            raise ConnectionError("镜像不可用或提交不一致：" + json.dumps(rows, ensure_ascii=False))
        selected = min([row for row in good if row["url"] != official] or good, key=lambda row: row["seconds"])
        keys = self.run("config", "--includes", "--name-only", "--get-regexp", r"^(http\..*(extraheader|cookiefile|savecookies)|credential\..*helper)$", absent=True).decode().splitlines()
        self.public_options = ["-c", "credential.helper=", "-c", "http.extraHeader=", "-c", "http.cookieFile=", "-c", "http.saveCookies=false", "-c", "http.sslVerify=true"]
        for key in keys:
            self.public_options += ["-c", key + "="]
        self.transport = selected["url"]
        self.network_info = {"source": self.transport, "probes": rows, "reference": "pinned" if expected is not None else "github" if reference_row else "mirror_agreement"}
        return selected["sha"]

    def remote_head(self):
        rows = self.run("ls-remote", "--exit-code", self.transport, "refs/heads/main", network=True).decode().splitlines()
        if len(rows) != 1:
            raise ValueError("不能唯一确定远端 main")
        return rows[0].split()[0]

    def inventory(self, commit):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("需要完整的提交 SHA")
        files = {}
        for row in filter(None, self.run("ls-tree", "-r", "-z", commit + ":" + SUBDIRECTORY).split(b"\0")):
            header, raw = row.split(b"\t", 1)
            mode, kind, blob = header.decode().split()
            name = raw.decode("utf-8")
            if mode != "100644" or kind != "blob" or not allowed(name):
                raise ValueError("代码目录包含非法路径或文件类型：" + name)
            files[name] = blob
        if not {"run.py", "setup_env.py", "pyproject.toml", "update.py"} <= files.keys() or len(files) > 200:
            raise ValueError("代码目录缺少必要文件或超出文件预算")
        return files

    def fetch_files(self, files):
        local = set(self.run("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines())
        missing = sorted(set(files.values()) - local)
        for start in range(0, len(missing), 64):
            data = ("\n".join(missing[start:start + 64]) + "\n").encode()
            self.run("-c", "fetch.negotiationAlgorithm=noop", "fetch", self.transport, "--no-tags", "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "--stdin", data=data, network=True)

    def staged_overrides(self, head, commit, force=False):
        names = set(filter(None, self.run("diff", "--cached", "--no-renames", "--no-ext-diff", "--no-textconv", "--name-only", "-z", head, "--", ".", ":(exclude)" + SUBDIRECTORY).split(b"\0")))
        if not names:
            return b""
        entries = {}
        for row in filter(None, self.run("ls-files", "--stage", "-z").split(b"\0")):
            header, name = row.split(b"\t", 1)
            if name in names:
                mode, blob, stage = header.split()
                if stage != b"0":
                    raise ValueError("其他项目的 Git 索引存在未解决冲突")
                entries[name] = (mode, blob, row)
        trees = []
        for revision in (head, commit):
            tree = {}
            for row in filter(None, self.run("ls-tree", "-r", "-z", revision).split(b"\0")):
                header, name = row.split(b"\t", 1)
                if name in names:
                    mode, _, blob = header.split()
                    tree[name] = (mode, blob)
            trees.append(tree)
        overrides = []
        for name in sorted(names):
            entry = entries.get(name)
            staged = entry[:2] if entry else None
            if not force and trees[0].get(name) != trees[1].get(name) and staged != trees[1].get(name):
                raise ValueError("其他项目的暂存内容与远端冲突，保留原索引：" + name.decode("utf-8"))
            overrides.append(entry[2] if entry else b"0 " + b"0" * 40 + b"\t" + name)
        return b"\0".join(overrides) + b"\0"

    def update_project(self, head, commit, files, initial, force=False, relocate_from=None):
        gitdir = self.path / ".git"
        actual_index, index_lock = gitdir / "index", gitdir / "index.lock"
        candidate = gitdir / ("chd-index-" + uuid4().hex)
        old_index = actual_index.read_bytes() if actual_index.exists() else None
        old_files = self.inventory(head)
        code = self.path / SUBDIRECTORY
        for name, blob in old_files.items():
            path = code / name
            if not force and path.exists() and self.run("hash-object", "--no-filters", "--", str(path)).decode().strip() != blob:
                raise ValueError("本项目已有文件被修改：" + name)
        owned_lock = False
        try:
            # 从目标树取得索引元数据，保留其他项目的暂存内容，不读取其文件对象。
            overrides = b"" if initial else self.staged_overrides(head, commit, force=force)
            self.run("read-tree", "--no-sparse-checkout", "--index-output=" + str(candidate), commit)
            if overrides:
                self.run("update-index", "-z", "--index-info", data=overrides, index=candidate)
            names = list(filter(None, self.run("ls-files", "-z", index=candidate).split(b"\0")))
            outside = [name for name in names if not name.startswith((SUBDIRECTORY + "/").encode())
                       and not (self.path / name.decode("utf-8")).exists()]
            if outside:
                self.run("update-index", "--skip-worktree", "-z", "--stdin", data=b"\0".join(outside) + b"\0", index=candidate)
            project_names = [(SUBDIRECTORY + "/" + name).encode() for name in files]
            self.run("update-index", "--no-skip-worktree", "-z", "--stdin", data=b"\0".join(project_names) + b"\0", index=candidate)
            with index_lock.open("xb") as stream:
                owned_lock = True
                if (actual_index.read_bytes() if actual_index.exists() else None) != old_index:
                    raise ValueError("Git 索引被其他操作修改，停止同步")
                require_head(head, self.run("rev-parse", "HEAD").decode().strip())
                if relocate_from is not None:
                    if code.exists():
                        raise FileExistsError("目标项目目录已经存在，停止迁移：" + str(code))
                    relocate_from.rename(code)
                changed = [(SUBDIRECTORY + "/" + name).encode() for name, blob in files.items()
                           if old_files.get(name) != blob or not (code / name).is_file()
                           or (force and self.run("hash-object", "--no-filters", "--", str(code / name)).decode().strip() != blob)]
                directories = set()
                for name in old_files.keys() - files.keys():
                    path = code / name
                    if path.is_file():
                        path.unlink()
                    parent = path.parent
                    while parent != code:
                        directories.add(parent)
                        parent = parent.parent
                for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                if changed:
                    self.run("checkout-index", "--force", "-z", "--stdin", data=b"\0".join(changed) + b"\0", index=candidate)
                for name, blob in files.items():
                    path = code / name
                    if path.is_symlink() or self.run("hash-object", "--no-filters", "--", str(path)).decode().strip() != blob:
                        raise ValueError("检出代码与 Git 提交不一致：" + name)
                stream.write(candidate.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
                tracking = self.run("rev-parse", "refs/remotes/origin/main").decode().strip()
                updates = ("start\nupdate refs/heads/main " + commit + " " + head
                           + "\nupdate refs/remotes/origin/main " + commit + " " + tracking + "\nprepare\ncommit\n")
                self.run("update-ref", "--stdin", data=updates.encode())
            os.replace(index_lock, actual_index)
            owned_lock = False
        finally:
            if owned_lock and index_lock.exists():
                index_lock.unlink()
            for owned in (candidate, candidate.with_name(candidate.name + ".lock")):
                if owned.exists():
                    owned.unlink()


def legacy_files(repo, runtime):
    code = repo.path.parent / SUBDIRECTORY
    if not code.exists():
        return None, []
    if code.is_symlink() or code.resolve().parent != repo.path.parent:
        raise ValueError("旧代码目录路径异常")
    expected = {"protocol": 1, "repository": REPOSITORY, "runtime": str(runtime)}
    if read_json(code / "deployment.json") != expected:
        raise ValueError("旧目录不属于当前项目与过程目录")
    root_names = {"deployment.json", "installed.json", "current.json", "run.py", "update.py", "code_versions"}
    if set(path.name for path in code.iterdir()) - root_names or any(path.is_symlink() for path in code.rglob("*")):
        raise ValueError("旧目录包含未知文件或链接，保留全部内容")
    installed = read_json(code / "installed.json")
    for release in (code / "code_versions").iterdir():
        manifest = read_json(release / "release.manifest.json")
        if not re.fullmatch(r"[0-9a-f]{40}", release.name) or manifest["sha"] != release.name or manifest["repository"] != REPOSITORY or manifest["subdirectory"] != SUBDIRECTORY:
            raise ValueError("旧版本身份异常")
        observed = {path.relative_to(release).as_posix(): path for path in release.rglob("*") if path.is_file() and path.name != "release.manifest.json"}
        if set(observed) != set(manifest["files"]):
            raise ValueError("旧版本存在缺失或新增文件，保留旧目录")
        for name, path in observed.items():
            if not allowed(name) or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
                raise ValueError("旧代码存在本地修改，保留旧目录")
    for name in ("installed.json", "current.json"):
        if (code / name).exists():
            pointer = read_json(code / name)
            if not re.fullmatch(r"[0-9a-f]{40}", pointer["sha"]) or hashlib.sha256((code / "code_versions" / pointer["sha"] / "release.manifest.json").read_bytes()).hexdigest() != pointer["manifest_sha256"]:
                raise ValueError("旧指针与代码清单不一致")
    source = code / "code_versions" / installed["sha"]
    if (code / "update.py").read_bytes() != (source / "update.py").read_bytes() or (code / "run.py").read_bytes() != (source / "launcher.py").read_bytes():
        raise ValueError("旧更新入口存在本地修改，保留旧目录")
    return code, [path for path in code.rglob("*") if path.is_file()]


def synchronize(repo, runtime, commit, remove_legacy=False, force=False, relocate=False):
    started, runtime = monotonic(), Path(runtime).resolve()
    if runtime.is_relative_to(repo.path) or repo.path.is_relative_to(runtime):
        raise ValueError("代码与过程目录必须独立")
    with file_lock(runtime / ".code.lock"), file_lock(repo.path / ".git" / "chd-code.lock"), ExitStack() as locks:
        for path in sorted((runtime / "runs").rglob(".run.lock")):
            locks.enter_context(file_lock(path))
        relocate_from = None
        old_code = repo.path / "git" / SUBDIRECTORY
        if relocate and old_code.exists():
            if (old_code.is_symlink() or not old_code.is_dir()
                    or any(path.is_symlink() for path in old_code.rglob("*"))
                    or (repo.path / ".git").resolve() != (repo.path / "git" / ".git").resolve()):
                raise ValueError("旧项目目录或 Git 元数据路径异常，停止迁移")
            if (repo.path / SUBDIRECTORY).exists():
                raise FileExistsError("code 下已有同名项目目录，停止迁移")
            relocate_from = old_code
        initial = repo.validate(force=force)
        legacy, obsolete = legacy_files(repo, runtime) if remove_legacy else (None, [])
        head = repo.run("rev-parse", "HEAD").decode().strip()
        require_head(commit, repo.remote_head())
        mark = monotonic()
        repo.run("fetch", "--no-tags", "--filter=blob:none", repo.transport, "main", network=True)
        require_head(commit, repo.run("rev-parse", "FETCH_HEAD").decode().strip())
        if not force:
            repo.run("merge-base", "--is-ancestor", head, commit)
        files = repo.inventory(commit)
        repo.fetch_files(files)
        for name, blob in files.items():
            if name.endswith(".py"):
                compile(repo.run("cat-file", "blob", blob), name, "exec")
        downloaded = monotonic() - mark
        require_head(commit, repo.remote_head())
        mark = monotonic()
        repo.validate(force=force)
        repo.update_project(head, commit, files, initial, force=force, relocate_from=relocate_from)
        repo.run("config", "branch.main.remote", "origin")
        repo.run("config", "branch.main.merge", "refs/heads/main")
        require_head(commit, repo.run("rev-parse", "HEAD").decode().strip())
        require_head(commit, repo.run("rev-parse", "origin/main").decode().strip())
        repo.validate(force=force)
        size = 0
        for name, blob in files.items():
            path = repo.path / SUBDIRECTORY / name
            if path.is_symlink() or repo.run("hash-object", "--no-filters", "--", str(path)).decode().strip() != blob:
                raise ValueError("检出代码与 Git 提交不一致：" + name)
            size += path.stat().st_size
        checkout_seconds = monotonic() - mark
        require_head(commit, repo.remote_head())
        deleted_bytes = sum(path.stat().st_size for path in obsolete)
        if legacy is not None:
            # 已逐文件核验、持有任务锁，且路径严格限定为代码目录下的旧项目。
            legacy_files(repo, runtime)
            shutil.rmtree(legacy)
        result = {"state": "synchronized", "forced": force, "tracked_code_clean": True, "sha": commit, "head": commit, "origin_main": commit, "code": str(repo.path / SUBDIRECTORY), "repository": REPOSITORY, "network": repo.network_info, "file_count": len(files), "code_bytes": size, "download_seconds": downloaded, "checkout_verify_seconds": checkout_seconds, "seconds": monotonic() - started, "checked_at": datetime.now(timezone.utc).isoformat(), "legacy_cleanup": {"deleted_files": len(obsolete), "released_bytes": deleted_bytes}}
        result["relocated_from"] = str(relocate_from) if relocate_from is not None else None
        write_json(runtime / "updates" / "latest.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description="只检出先心分割目录，将本地 main 快进到远端 main")
    parser.add_argument("--repo", default="/data4/zhangruichen/code")
    parser.add_argument("--runtime", default="/data4/zhangruichen/course/chd-segmentation")
    parser.add_argument("--sha")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true", help="本项目代码以远端为准，修复未完整检出的索引，保留其他项目的文件和暂存记录")
    parser.add_argument("--relocate", action="store_true", help="将实体项目从 code/git/chd-segmentation 移到 code/chd-segmentation，复用 Git 元数据")
    parser.add_argument("--remove-legacy", action="store_true", help="核验并删除本项目旧版本安装目录")
    parser.add_argument("--mirror", action="append")
    parser.add_argument("--network-config")
    args = parser.parse_args()
    if args.relocate:
        if not args.force or args.remove_legacy:
            raise ValueError("目录迁移请使用 --force --relocate")
        prepare_code_root(args.repo)
    repo = GitRepository(args.repo)
    repo.validate(force=args.force)
    settings_path = Path(args.network_config) if args.network_config else Path(args.runtime) / "update_network.json"
    settings = network_settings(settings_path)
    choice = args.mirror or ["auto"]
    candidates = [OFFICIAL] if choice == ["direct"] else [mirror_url(prefix) for prefix in (settings["mirrors"] if choice == ["auto"] else choice)] + [OFFICIAL]
    commit = repo.choose_source(candidates, expected=args.sha, timeout=settings["probe_timeout_seconds"], official=OFFICIAL)
    if args.check:
        result = {"head": repo.run("rev-parse", "HEAD").decode().strip(), "remote_sha": commit, "code": str(repo.path / SUBDIRECTORY)}
    else:
        if not settings_path.exists():
            write_json(settings_path, settings)
        result = synchronize(repo, args.runtime, commit, remove_legacy=args.remove_legacy, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
