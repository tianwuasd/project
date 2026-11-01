"""Bounded official/private or public-mirror updates of an idle, clean checkout."""

import os
import re
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import inspection, jobs, network
from .launcher import ROOT, write_json

QUERY_TIMEOUT = 30
FETCH_TIMEOUT = 900
ARCHIVE_TIMEOUT = 180
APPLY_TIMEOUT = 120

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
    network.SSH_UPSTREAM,
)


def git(root, *arguments, timeout=20, check=True, progress_log=None, stage="Git 操作"):
    """Scope credentials to the selected route and reap helpers on cancellation."""
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
    settings = network.current()
    authenticated = settings["git_transport"] != "public-mirrors"
    if authenticated:
        # Auth is permitted only for the explicit official route. Local Git's
        # credential helpers remain usable; mirrors never enter this branch.
        command = [
            "git",
            "-c",
            "core.askPass=",
            "-c",
            "http.followRedirects=false",
            "-c",
            "http." + UPSTREAM + ".followRedirects=false",
            "-c",
            "http.lowSpeedLimit=1",
            "-c",
            "http.lowSpeedTime=15",
            "-C",
            str(root),
            *arguments,
        ]
    if settings["proxy"]:
        command[1:1] = ["-c", "http.proxy=" + settings["proxy"]]
        for url in {UPSTREAM, *network.routes(settings)}:
            if url.startswith("https://"):
                command[1:1] = ["-c", "http." + url + ".proxy=" + settings["proxy"]]
    env = network.child_env(ssh=network.SSH_UPSTREAM in arguments)
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
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
        if progress_log is None:
            out, err = process.communicate(timeout=timeout)
        else:
            from .update_progress import communicate

            out, err = communicate(process, timeout, Path(progress_log), stage)
        result = subprocess.CompletedProcess(command, process.returncode, out, err)
        if check and result.returncode:
            if progress_log is not None:
                from .update_progress import StageFailure

                raise StageFailure(stage, err, progress_log)
            if authenticated:
                raise ValueError("Git 操作失败；请检查官方线路、代理、主机指纹及仓库认证，不会回退公共镜像")
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
            if progress_log is None:
                process.communicate()
            else:
                process.wait()
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
    if subtree != "chd-ct-panel-v2":
        raise ValueError("自动更新需要 project/chd-ct-panel-v2 发布目录。")
    branch = git(root, "branch", "--show-current").stdout.strip()
    if branch not in {"main", "codex/ct-panel-v2"}:
        raise ValueError("自动更新只支持 main 或 codex/ct-panel-v2 分支；当前分支保持不变。")
    if git(root, "status", "--porcelain", "--untracked-files=normal").stdout.strip():
        raise ValueError("仓库有本地修改或未跟踪文件，请先保存处理后更新；不会覆盖或自动 stash。")
    for marker in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD"):
        path = Path(git(root, "rev-parse", "--git-path", marker).stdout.strip())
        if (path if path.is_absolute() else root / path).exists():
            raise ValueError("仓库有未完成的 Git 操作，请先处理后更新。")
    return dict(root=str(root), subtree=subtree, head=git(root, "rev-parse", "HEAD").stdout.strip(), branch=branch)


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


def error_detail(error, stage):
    from .update_progress import safe_text

    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout", f"{stage}总超时（上限 {error.timeout:g} 秒）"
    if isinstance(error, KeyboardInterrupt):
        return "cancelled", stage + "被用户取消"
    if isinstance(error, OSError):
        return "io", stage + "发生本地 I/O 错误：" + safe_text(error)
    return "validation" if stage == "版本校验" else "git", safe_text(error)


def update(repo, runtime):
    settings = network.load(runtime)
    network.show(settings)
    with network.use(settings):
        return _update(repo, runtime, settings)


def _update(repo, runtime, settings):
    root, before = Path(repo["root"]), repo["head"]
    branch = repo["branch"]
    branch_ref = "refs/heads/" + branch
    if runtime.resolve() == root or root in runtime.resolve().parents:
        raise ValueError("更新备份目录必须位于代码仓库之外，请使用独立 runtime。")
    project = root / repo["subtree"]
    if inspect_repository(project) != repo:
        raise ValueError("更新前仓库状态已改变，请重新进入更新。")
    directory = runtime / "updates" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    routes = network.routes(settings)
    official = routes[0] if settings["git_transport"] != "public-mirrors" else UPSTREAM
    started = time.monotonic()
    report = dict(
        before=before,
        upstream=official,
        attempts=[],
        status="checking",
        stage="仓库检查",
        limits=dict(query=QUERY_TIMEOUT, fetch=FETCH_TIMEOUT, archive=ARCHIVE_TIMEOUT, apply=APPLY_TIMEOUT),
        network={"proxy": settings["proxy"], "git_transport": settings["git_transport"]},
    )
    report_file = directory / "update.json"
    refs = []
    stage_started = started
    try:
        if settings["git_transport"] != "public-mirrors":
            resolved = git(root, "ls-remote", "--get-url", official).stdout.strip()
            if resolved != official:
                raise ValueError("Git URL 重写改变了官方地址；请检查 insteadOf 配置，不会发送认证或访问镜像")
        expected = None
        report["stage"] = "远端版本查询"
        query_started = time.monotonic()
        stage_started = query_started
        report["query_log"] = str(directory / "query.log")
        print(f"检查 GitHub {branch} 的版本（最多 {QUERY_TIMEOUT} 秒）……", flush=True)
        try:
            output = git(
                root,
                "ls-remote",
                official,
                branch_ref,
                timeout=QUERY_TIMEOUT,
                progress_log=report["query_log"],
                stage="远端版本查询",
            ).stdout
            match = re.fullmatch(r"([0-9a-f]{40,64})\s+" + re.escape(branch_ref) + r"\s*", output)
            if not match:
                raise ValueError(f"上游没有返回有效 {branch} 提交")
            expected = match[1]
            report["upstream_head"] = expected
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["upstream_error_kind"], report["upstream_error"] = error_detail(error, report["stage"])
            report["query_last_output"] = getattr(error, "output_tail", "")
            print("GitHub 版本查询未完成：" + report["upstream_error"], flush=True)
            print("将尝试配置的下载线路；最新状态仍需核验。", flush=True)
        finally:
            report["query_elapsed_seconds"] = round(time.monotonic() - query_started, 3)
            write_json(report_file, report)
        selected = before
        reachable = False
        for index, url in enumerate(routes, 1):
            print(
                f"更新线路 {index}/{len(routes)}：{url}（下载总上限 {FETCH_TIMEOUT} 秒，完成即继续）",
                flush=True,
            )
            ref = "refs/chd-update/" + uuid4().hex
            refs.append(ref)
            log = directory / f"fetch-{index}.log"
            attempt_started = time.monotonic()
            stage_started = attempt_started
            attempt = dict(url=url, stage="对象下载", log=str(log))
            report["attempts"].append(attempt)
            report["stage"] = "对象下载"
            write_json(report_file, report)
            print("下载日志：" + str(log), flush=True)
            try:
                result = git(
                    root,
                    "fetch",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "--no-write-fetch-head",
                    url,
                    branch_ref + ":" + ref,
                    "--progress",
                    timeout=FETCH_TIMEOUT,
                    progress_log=log,
                    stage="对象下载",
                )
                attempt["last_output"] = result.stderr[-4000:]
                attempt["download_elapsed_seconds"] = round(time.monotonic() - attempt_started, 3)
                attempt["stage"] = report["stage"] = "版本校验"
                stage_started = time.monotonic()
                head = git(root, "rev-parse", ref + "^{commit}").stdout.strip()
                attempt["head"] = head
                if expected and head != expected:
                    raise ValueError(f"线路提交与本次 GitHub {branch} 不一致，继续切换")
                if not ancestor(root, before, head):
                    raise ValueError("线路版本较旧或与本地分支分叉，继续切换")
                selected, reachable = head, True
                if settings["git_transport"] != "public-mirrors" and expected is None:
                    expected = head
                    report["upstream_head"] = head
                attempt["status"] = "verified"
                # Equal HEAD is valid too; later routes cannot replace this candidate.
                break
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                kind, detail = error_detail(error, attempt["stage"])
                attempt.update(status="failed", error=detail, error_kind=kind)
                if hasattr(error, "output_tail"):
                    attempt["last_output"] = error.output_tail
                print(f"此线路未通过：{detail}；下载日志：{log}", flush=True)
            except KeyboardInterrupt as error:
                attempt.update(status="cancelled", error="用户取消对象下载/校验", error_kind="cancelled")
                attempt["last_output"] = getattr(error, "output_tail", "")
                raise
            finally:
                attempt["elapsed_seconds"] = round(time.monotonic() - attempt_started, 3)
                if log.is_file() and "last_output" not in attempt:
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 4000))
                        attempt["last_output"] = stream.read().decode("utf-8", errors="replace")
                write_json(report_file, report)
        if not reachable:
            reason = report["attempts"][-1]["error"] if report["attempts"] else "无可用线路"
            raise ValueError(f"所有更新线路均未通过；最近失败：{reason}。代码未更新，记录：{report_file}")
        report["selected"] = selected
        if selected == before:
            report["stage"] = "完成"
            report["status"] = "current" if expected else "mirror_current"
            print(
                "已是 GitHub 本次查询的版本。"
                if expected
                else "当前线路与本地版本一致；GitHub 最新状态未能确认。"
            )
            return False
        if inspect_repository(project) != repo:
            raise ValueError("下载期间本地仓库发生变化，未应用更新，请重新检查。")
        changed = git(root, 'diff', '--name-only', before, selected).stdout.splitlines()
        if any(not n.startswith('chd-ct-panel-v2/') for n in changed):
            raise ValueError('更新包含V2之外的路径，保留原代码供人工检查')
        archive = directory / "code-before-update.tar"
        report["stage"] = "代码备份"
        archive_started = time.monotonic()
        stage_started = archive_started
        report["archive_log"] = str(directory / "archive.log")
        print(f"备份当前 CT 代码（最多 {ARCHIVE_TIMEOUT} 秒）……", flush=True)
        write_json(report_file, report)
        git(
            root,
            "-c",
            "core.autocrlf=false",
            "archive",
            "--format=tar",
            "--output=" + str(archive),
            before,
            repo["subtree"],
            timeout=ARCHIVE_TIMEOUT,
            progress_log=report["archive_log"],
            stage="代码备份",
        )
        report.update(
            status="applying",
            stage="应用更新",
            backup=str(archive),
            archive_elapsed_seconds=round(time.monotonic() - archive_started, 3),
        )
        write_json(report_file, report)
        print("代码备份：" + str(archive), flush=True)
        apply_started = time.monotonic()
        stage_started = apply_started
        report["apply_log"] = str(directory / "apply.log")
        write_json(report_file, report)
        if inspect_repository(project) != repo:
            raise ValueError("备份期间仓库分支、提交或文件发生改变，未应用更新。")
        git(
            root,
            "merge",
            "--ff-only",
            "--no-overwrite-ignore",
            "--no-edit",
            selected,
            timeout=APPLY_TIMEOUT,
            progress_log=report["apply_log"],
            stage="应用更新",
        )
        report["apply_elapsed_seconds"] = round(time.monotonic() - apply_started, 3)
        after = git(root, "rev-parse", "HEAD").stdout.strip()
        after_branch = git(root, "branch", "--show-current").stdout.strip()
        if after != selected or after_branch != branch:
            raise ValueError("更新后提交核验失败，请查看更新记录。")
        report.update(status="updated", stage="完成", after=after)
        print(f"版本：{before[:12]} → {after[:12]}\n更新记录：{report_file}", flush=True)
        return True
    except BaseException as error:
        kind, detail = error_detail(error, report["stage"])
        report.update(status="cancelled" if kind == "cancelled" else "failed", error=detail, error_kind=kind)
        report["stage_elapsed_seconds"] = round(time.monotonic() - stage_started, 3)
        report["last_output"] = getattr(error, "output_tail", "")
        print(f"更新停止于{report['stage']}：{detail}；记录：{report_file}", flush=True)
        raise
    finally:
        for ref in refs:
            try:
                result = git(root, "update-ref", "-d", ref, check=False)
                if result.returncode:
                    report.setdefault("cleanup_errors", []).append("临时引用未清理：" + ref)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                report.setdefault("cleanup_errors", []).append(error_detail(error, "临时引用清理")[1])
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(report_file, report)
