"""Publish immutable artifact snapshots using SSH Git authentication and Git LFS.

Only archive pointers and metadata enter an orphan branch. No gh, PAT, checkout
of main or global Git configuration is used. LFS HTTP authorization is obtained
by Git LFS through the configured SSH key.
"""

import hashlib
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

from . import network, upload

ATTRIBUTES = b"artifacts.tar.part* filter=lfs diff=lfs merge=lfs -text\n"
RESTORE = b"""# VISTA3D artifact snapshot

This branch contains approved logs, configuration, metrics and model weights.
Archive parts are Git LFS objects. Use Git and Git LFS with the same authorized
SSH key to clone this branch, then run `git lfs pull`. Verify manifest.json sizes
and SHA256, concatenate artifacts.tar.part* in numeric order, and extract the
resulting tar into a new empty directory. Do not merge this branch into main.
The manifest records source paths for provenance; it does not include raw CT.
"""


def require_ssh():
    if network.current()["git_transport"] != "github-ssh":
        raise ValueError("上传使用同一 SSH 密钥；请先在 c 网络设置选择 github-ssh 并填写密钥路径")
    network.child_env(ssh=True)  # Check the key path and CONNECT helper, not secret contents.


def git(root, *args, input_path=None, timeout=60, check=True, commit_date=None):
    """Bounded subprocesses in a private repo; neither inherited Git rewrites nor hooks."""
    require_ssh()
    if not shutil.which("git"):
        raise ValueError("未安装 Git；上传需要 Git 和 Git LFS，不需要 gh")
    env = network.child_env(ssh=True)
    for key in list(env):
        if key.startswith("GIT_") and key not in {"GIT_SSH_COMMAND", "GIT_SSH_VARIANT"}:
            env.pop(key)
        if key in {"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"}:
            env.pop(key)
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GCM_INTERACTIVE="never",
        GIT_LFS_SKIP_SMUDGE="1",
    )
    if commit_date:
        env.update(GIT_AUTHOR_DATE=commit_date, GIT_COMMITTER_DATE=commit_date)
    command = [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=",
        "-c",
        "core.autocrlf=false",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "core.hooksPath=" + str(Path(root) / ".git/no-hooks"),
        "-c",
        "lfs.locksverify=false",
        "-c",
        "lfs.ssh.retries=1",
        "-C",
        str(root),
        *args,
    ]
    if network.current()["proxy"]:
        command[1:1] = ["-c", "http.proxy=" + network.current()["proxy"]]
    process, cancelled = None, False
    handlers = {}
    source = Path(input_path).open("rb") if input_path else None

    def interrupt(sig, frame):
        nonlocal cancelled
        cancelled = True
        if process is not None:
            raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupt)
        process = subprocess.Popen(
            command,
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            start_new_session=os.name == "posix",
        )
        if cancelled:
            raise KeyboardInterrupt
        out, err = process.communicate(timeout=timeout)
        result = subprocess.CompletedProcess(command, process.returncode, out, err)
        if check and result.returncode:
            raise ValueError("Git/SSH/LFS 操作失败；检查密钥写权限、代理、Git LFS安装及额度，原包已保留。")
        return result
    except BaseException as error:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise ValueError("Git/SSH/LFS 操作超时；原包和上传进度已保留，可重试同一个包") from None
        raise
    finally:
        if source:
            source.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def remote_head(repo, ref):
    rows = git(repo, "ls-remote", network.SSH_UPSTREAM, ref).stdout.splitlines()
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("远端上传分支返回不唯一，停止上传")
    fields = rows[0].split()
    if len(fields) != 2 or fields[1] != ref or not re.fullmatch(r"[0-9a-f]{40}", fields[0]):
        raise ValueError("远端上传分支格式无效，停止上传")
    return fields[0]


def check_local_routes(repo):
    if (Path(repo) / ".lfsconfig").exists():
        raise ValueError("仓库含额外 LFS 配置文件，请先人工核对")
    settings = git(repo, "config", "--local", "--name-only", "--list").stdout.splitlines()
    for name in settings:
        key = name.lower()
        if (
            key.startswith(("url.", "include.", "includeif.", "lfs.customtransfer."))
            or key
            in {
                "lfs.url",
                "lfs.pushurl",
                "remote.origin.lfsurl",
                "remote.origin.lfspushurl",
                "lfs.standalonetransferagent",
                "remote.origin.pushurl",
            }
            or key.startswith("http.")
            and key.endswith((".extraheader", ".proxy", ".sslverify"))
        ):
            raise ValueError("仓库含额外网络地址/认证重写设置；请先人工核对，不发送凭据")
    resolved = git(repo, "ls-remote", "--get-url", network.SSH_UPSTREAM).stdout.strip()
    if resolved != network.SSH_UPSTREAM:
        raise ValueError("Git URL 重写改变了官方地址，停止上传")


def pointer(asset):
    return (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{asset['sha256']}\nsize {asset['size']}\n"
    ).encode("ascii")


def expected_files(package, assets):
    return {
        ".gitattributes": ATTRIBUTES,
        "README.md": RESTORE,
        "manifest.json": (package / "manifest.json").read_bytes(),
        **{a["name"]: pointer(a) for a in assets if a["name"] != "manifest.json"},
    }


def verify_tree(repo, expected):
    names = git(repo, "ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines()
    if set(names) != set(expected) or len(names) != len(expected):
        raise ValueError("本地上传提交包含不同文件；保留原包，停止上传")
    for name, content in expected.items():
        # Metadata is UTF-8; compare Git blob identities without newline conversion.
        oid = git(repo, "rev-parse", "HEAD:" + name).stdout.strip()
        raw_oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if oid != raw_oid:
            raise ValueError("本地上传提交内容与原包不一致：" + name)


def build_commit(package, manifest, assets):
    repo = upload._safe_path(package / "git-publication")
    marker = upload._safe_path(package / "git-publication.json")
    identity = {
        "package_id": manifest["package_id"],
        "manifest_sha256": upload._hash(package / "manifest.json"),
    }
    if marker.exists():
        if upload._json(marker) != identity:
            raise ValueError("上传工作目录身份与原包不一致")
    else:
        if repo.exists():
            raise ValueError("上传工作目录已存在但没有本包身份记录")
        upload._write_json(marker, identity)
    repo.mkdir(exist_ok=True)
    upload._safe_path(repo / ".git")
    if not (repo / ".git").exists():
        git(repo, "init", "-b", "snapshot", "--template=")
    git(repo, "lfs", "version")
    git(repo, "config", "--local", "remote.origin.url", network.SSH_UPSTREAM)
    check_local_routes(repo)
    expected = expected_files(package, assets)
    if git(repo, "rev-parse", "--verify", "HEAD", check=False).returncode == 0:
        verify_tree(repo, expected)
        return repo, git(repo, "rev-parse", "HEAD").stdout.strip()
    total = sum(a["size"] for a in assets)
    if shutil.disk_usage(repo).free < total + upload.BUFFER_BYTES:
        raise ValueError("Git LFS本地缓存空间不足；需额外约一份上传包的空间")
    for index, asset in enumerate(assets, 1):
        if asset["name"] == "manifest.json":
            continue
        print(f"准备LFS对象 {index}/{len(assets) - 1}：{asset['name']}", flush=True)
        # Stream from the immutable package; never hold a GiB part in memory.
        value = git(
            repo,
            "lfs",
            "clean",
            "--",
            asset["name"],
            input_path=package / asset["name"],
            timeout=upload.UPLOAD_TIMEOUT,
        ).stdout.encode("utf-8")
        if value != expected[asset["name"]]:
            raise ValueError("Git LFS对象摘要与上传清单不一致")
    for name, content in expected.items():
        path = upload._safe_path(repo / name)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("上传工作文件内容与原包不一致：" + name)
        path.write_bytes(content)
    git(repo, "add", "--", *sorted(expected))
    git(
        repo,
        "-c",
        "user.name=VISTA3D artifacts",
        "-c",
        "user.email=vista3d-artifacts@users.noreply.github.com",
        "commit",
        "-m",
        "VISTA3D artifact package " + manifest["package_id"],
        commit_date=manifest["created_at"],
    )
    verify_tree(repo, expected)
    return repo, git(repo, "rev-parse", "HEAD").stdout.strip()


def publish(package, manifest, assets):
    require_ssh()
    upload._finished(Path(manifest["job_directory"]), Path(manifest["runtime"]))
    expected = expected_files(package, assets)
    repo, commit = build_commit(package, manifest, assets)
    if expected != expected_files(package, assets):
        raise ValueError("上传清单发生改变，停止上传")
    verify_tree(repo, expected)
    branch = "vista3d-artifacts/" + manifest["package_id"]
    ref = "refs/heads/" + branch
    remote = remote_head(repo, ref)
    if remote is not None and remote != commit:
        raise ValueError("远端同名产物分支与本包提交不一致；不会覆盖或强推")
    if remote is None:
        print("正在通过同一SSH密钥认证上传LFS分卷……", flush=True)
        git(repo, "lfs", "push", "origin", commit, timeout=upload.UPLOAD_TIMEOUT)
        # Recheck immutable source/manifest before making the branch visible.
        upload._validate_package(package)
        if expected != expected_files(package, assets):
            raise ValueError("上传清单发生改变，停止上传")
        verify_tree(repo, expected)
        print("LFS上传完成，正在推送独立产物分支……", flush=True)
        git(repo, "push", network.SSH_UPSTREAM, f"{commit}:{ref}", timeout=upload.UPLOAD_TIMEOUT)
    if remote_head(repo, ref) != commit:
        raise ValueError("远端提交尚未确认；保留同一个包重试")
    url = f"https://github.com/{upload.REPOSITORY}/tree/{branch}"
    upload._write_json(
        package / "receipt.json",
        {
            "schema_version": 2,
            "transport": "git-ssh-lfs",
            "package_id": manifest["package_id"],
            "repository": upload.REPOSITORY,
            "branch": branch,
            "ref": ref,
            "commit": commit,
            "url": url,
            "verified_at": upload._now(),
            "assets": assets,
            "verification": "LFS command accepted objects; Git remote ref equals the verified pointer/manifest commit.",
        },
    )
    return url
