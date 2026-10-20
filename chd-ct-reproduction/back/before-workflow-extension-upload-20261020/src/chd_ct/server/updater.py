"""Bounded public Git mirror fallback; update only an idle, clean main checkout."""

import os
import re
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import inspection, jobs
from .launcher import ROOT, write_json

UPSTREAM = "https://github.com/tianwuasd/project.git"
MIRRORS = (
    "https://gh-proxy.org/" + UPSTREAM,
    "https://ghfast.top/" + UPSTREAM,
    "https://gitclone.com/github.com/tianwuasd/project.git",
    UPSTREAM,
)
# Recognize existing clones even when their origin is not a download route.
# Keep retired, known origins here when changing MIRRORS.
ORIGIN_URLS = (
    *MIRRORS,
    "https://gh-proxy.com/" + UPSTREAM,
    "git@github.com:tianwuasd/project.git",
)


def git(root, *arguments, timeout=20, check=True):
    """No shell, no credential forwarding; reap this command's helpers on cancellation."""
    command = [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=",
        "-c",
        "http.extraHeader=",
        "-c",
        "http.lowSpeedLimit=1",
        "-c",
        "http.lowSpeedTime=15",
        "-C",
        str(root),
        *arguments,
    ]
    # URL-specific configuration overrides generic resets in Git. Clear both,
    # including for a direct GitHub fallback, without changing saved settings.
    for url in {UPSTREAM, *MIRRORS}:
        if url.startswith("https://"):
            command[1:1] = ["-c", f"credential.{url}.helper=", "-c", f"http.{url}.extraHeader="]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    process, cancelled = None, False
    handlers = {}

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
            raise ValueError("Git 操作失败：" + (err or out).strip()[-1800:])
        return result
    except BaseException:
        # Ignore repeated interrupts only while reaping our own process group.
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
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def inspect_repository(project=ROOT):
    root = Path(git(project, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    subtree = Path(project).resolve().relative_to(root).as_posix()
    origin = git(root, "remote", "get-url", "origin").stdout.strip().rstrip("/")
    if origin.removesuffix(".git") not in {u.removesuffix(".git") for u in ORIGIN_URLS}:
        raise ValueError(
            "自动更新仅适用于 tianwuasd/project 的发布仓库；"
            "当前 origin 地址未识别，请运行 git remote get-url origin 核对。"
        )
    if subtree != "chd-ct-reproduction":
        raise ValueError("自动更新需要 project/chd-ct-reproduction 发布目录。")
    if git(root, "branch", "--show-current").stdout.strip() != "main":
        raise ValueError("自动更新只支持 main 分支；当前分支保持不变。")
    if git(root, "status", "--porcelain", "--untracked-files=normal").stdout.strip():
        raise ValueError("仓库有本地修改或未跟踪文件，请先保存处理后更新；不会覆盖或自动 stash。")
    for marker in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD"):
        path = Path(git(root, "rev-parse", "--git-path", marker).stdout.strip())
        if (path if path.is_absolute() else root / path).exists():
            raise ValueError("仓库有未完成的 Git 操作，请先处理后更新。")
    return dict(root=str(root), subtree=subtree, head=git(root, "rev-parse", "HEAD").stdout.strip())


@contextmanager
def idle_lock(runtime, target, timeout=60):
    """Retain the same inherited training lock through backup and checkout."""
    deadline = time.monotonic() + timeout
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(jobs.acquire_lock(runtime))
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise ValueError("任务尚未完全退出，本次没有更新代码；稍后在面板重试。") from None
                print("等待任务及其子进程退出……", flush=True)
                time.sleep(min(2, max(0, deadline - time.monotonic())))
        if inspection.latest(runtime) != target or inspection.job_active(target):
            raise ValueError("任务记录已改变或仍有存活进程，本次不更新。")
        yield


def ancestor(root, older, newer):
    result = git(root, "merge-base", "--is-ancestor", older, newer, check=False)
    if result.returncode not in (0, 1):
        raise ValueError("无法核验提交历史；不会覆盖本地分支。")
    return result.returncode == 0


def update(repo, runtime):
    root, before = Path(repo["root"]), repo["head"]
    if runtime.resolve() == root or root in runtime.resolve().parents:
        raise ValueError("更新备份目录必须位于代码仓库之外，请使用独立 runtime。")
    project = root / repo["subtree"]
    if inspect_repository(project) != repo:
        raise ValueError("更新前仓库状态已改变，请重新进入更新。")
    directory = runtime / "updates" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    report = dict(before=before, upstream=UPSTREAM, attempts=[], status="checking")
    report_file = directory / "update.json"
    refs = []
    try:
        expected = None
        print("检查 GitHub main 的版本（最多 12 秒）……", flush=True)
        try:
            output = git(root, "ls-remote", UPSTREAM, "refs/heads/main", timeout=12).stdout
            match = re.fullmatch(r"([0-9a-f]{40,64})\s+refs/heads/main\s*", output)
            if not match:
                raise ValueError("上游没有返回有效 main 提交")
            expected = match[1]
            report["upstream_head"] = expected
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["upstream_error"] = str(error)
            print("GitHub 版本查询不可用，将比较可访问镜像；无法保证镜像已同步最新版本。", flush=True)
        selected = before
        reachable = False
        divergent = False
        for index, url in enumerate(MIRRORS, 1):
            print(f"更新线路 {index}/{len(MIRRORS)}：{url}（最多 45 秒）", flush=True)
            ref = "refs/chd-update/" + uuid4().hex
            refs.append(ref)
            attempt = dict(url=url)
            report["attempts"].append(attempt)
            try:
                git(
                    root,
                    "fetch",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "--no-write-fetch-head",
                    url,
                    "refs/heads/main:" + ref,
                    timeout=45,
                )
                head = git(root, "rev-parse", ref + "^{commit}").stdout.strip()
                attempt["head"] = head
                if expected and head != expected:
                    raise ValueError("线路提交与本次 GitHub main 不一致，继续切换")
                if not ancestor(root, before, head):
                    raise ValueError("线路版本较旧或与本地分支分叉，继续切换")
                if not ancestor(root, selected, head):
                    if ancestor(root, head, selected):
                        attempt["status"] = "older_mirror"
                        continue
                    divergent = True
                    raise ValueError("镜像提交历史互相分叉，无法确定更新版本")
                selected, reachable = head, True
                attempt["status"] = "verified"
                if expected:
                    break
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                attempt.update(status="failed", error=str(error))
                print("此线路未通过：" + str(error), flush=True)
            finally:
                write_json(report_file, report)
        if divergent:
            raise ValueError("可访问镜像历史分叉，未更新代码；请待 GitHub 可用后重试。")
        if not reachable:
            raise ValueError("所有更新线路均未通过；代码未更新。请稍后重试，记录：" + str(report_file))
        report["selected"] = selected
        if selected == before:
            report["status"] = "current" if expected else "no_newer_mirror"
            print(
                "已是 GitHub 本次查询的版本。"
                if expected
                else "可访问镜像没有更新版本；GitHub 最新状态未能确认。"
            )
            return False
        if inspect_repository(project) != repo:
            raise ValueError("下载期间本地仓库发生变化，未应用更新，请重新检查。")
        archive = directory / "code-before-update.tar"
        git(
            root,
            "-c",
            "core.autocrlf=false",
            "archive",
            "--format=tar",
            "--output=" + str(archive),
            before,
            repo["subtree"],
            timeout=45,
        )
        report.update(status="applying", backup=str(archive))
        write_json(report_file, report)
        print("代码备份：" + str(archive), flush=True)
        git(root, "merge", "--ff-only", "--no-overwrite-ignore", "--no-edit", selected, timeout=45)
        after = git(root, "rev-parse", "HEAD").stdout.strip()
        if after != selected:
            raise ValueError("更新后提交核验失败，请查看更新记录。")
        report.update(status="updated", after=after)
        print(f"版本：{before[:12]} → {after[:12]}\n更新记录：{report_file}", flush=True)
        return True
    except BaseException as error:
        report.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        write_json(report_file, report)
        for ref in refs:
            git(root, "update-ref", "-d", ref, check=False)
