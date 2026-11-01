"""Preview and publish committed CT code with the configured SSH key."""

import re
from pathlib import Path, PurePosixPath
from uuid import uuid4

from . import inspection, jobs, network, ssh_upload, updater, upload


def allowed(name):
    path = PurePosixPath(name)
    if not name.startswith("chd-ct-panel-v2/"):
        return False
    relative = path.parts[1:]
    if not relative or any(p.startswith(".") for p in relative[:-1]):
        return False
    # The integrated engine carries its own data/output directory convention.
    # Inspect that nested scope as strictly as the top-level code project.
    if relative[0] == "foundation_engine":
        nested = relative[1:]
        if not nested:
            return False
        if nested[0] == "04_data":
            return nested == ("04_data", "manifests", "README.md")
        if nested[0] in {"07_experiments", "08_results", "09_reports", "model_inputs", "model_weights"}:
            return False
        if nested[0] in {"data", "results", "runtime", "cache", "uploads", "back"}:
            return False
    if relative[0] in {"back", "data", "results", "runtime", "cache", "uploads"}:
        return False
    if relative[:2] == ("docs", "superpowers"):
        return False
    synthetic_update_test = relative == ("foundation_engine", "tests", "test_private_update.py")
    if not synthetic_update_test and re.search(r"secret|token|credential|password|id_rsa|id_ed25519|private", path.name, re.I):
        return False
    return relative == ("start_server_v2.cmd",) or path.name in {".gitignore", ".gitattributes"} or path.suffix in {
        ".py",
        ".sh",
        ".bat",
        ".toml",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".cff",
    }


def prepare(project=updater.ROOT):
    ssh_upload.require_ssh()
    repo = updater.inspect_repository(project)
    root, head = Path(repo["root"]), repo["head"]
    ssh_upload.check_local_routes(root)
    ref = "refs/chd-code-preview/" + uuid4().hex
    try:
        ssh_upload.git(
            root,
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            network.SSH_UPSTREAM,
            "refs/heads/codex/ct-panel-v2:" + ref,
            timeout=60,
        )
        remote = ssh_upload.git(root, "rev-parse", ref).stdout.strip()
        if not updater.ancestor(root, remote, head):
            raise ValueError("远端 codex/ct-panel-v2 有新提交或历史分叉；请先更新并处理差异，不能强推")
        span = remote + ".." + head
        if ssh_upload.git(root, "rev-list", "--merges", span).stdout.strip():
            raise ValueError("待上传历史包含合并提交，请人工检查后通过 Git 上传")
        commits = ssh_upload.git(root, "rev-list", "--reverse", span).stdout.splitlines()
        files = set()
        for commit in commits:
            names = ssh_upload.git(
                root, "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit
            ).stdout.split("\0")
            files.update(name for name in names if name)
        if any(not allowed(name) for name in files):
            raise ValueError(
                "待上传提交包含 CT 代码范围之外的内容；请人工整理提交，面板不会自动提交或改写历史"
            )
        # Inspect every new blob, including files removed in a later commit.
        objects = ssh_upload.git(root, "rev-list", "--objects", span).stdout.splitlines()
        for row in objects:
            oid = row.split(" ", 1)[0]
            if not re.fullmatch(r"[0-9a-f]{40}", oid):
                raise ValueError("待上传对象无效")
            kind = ssh_upload.git(root, "cat-file", "-t", oid).stdout.strip()
            if kind == "blob" and int(ssh_upload.git(root, "cat-file", "-s", oid).stdout) > 2 * 1024**2:
                raise ValueError("代码范围包含超过 2 MiB 的文件；模型和日志请通过 v 上传")
        return dict(repo=repo, remote=remote, commits=commits, files=sorted(files))
    finally:
        ssh_upload.git(root, "update-ref", "-d", ref)


def publish(preview, runtime):
    repo = preview["repo"]
    root = Path(repo["root"])
    if Path(runtime).resolve().is_relative_to(root.resolve()):
        raise ValueError("上传记录目录必须使用代码仓库之外的独立 runtime")
    if updater.inspect_repository(root / repo["subtree"]) != repo:
        raise ValueError("预览后代码发生改变，请重新预览")
    ssh_upload.check_local_routes(root)
    ref, commit = "refs/heads/codex/ct-panel-v2", repo["head"]
    current = ssh_upload.remote_head(root, ref)
    if current not in {preview["remote"], commit}:
        raise ValueError("预览后远端 codex/ct-panel-v2 已改变，请重新更新和预览")
    if current != commit:
        ssh_upload.git(root, "push", network.SSH_UPSTREAM, commit + ":" + ref, timeout=120)
    if ssh_upload.remote_head(root, ref) != commit:
        raise ValueError("未确认远端提交，重新预览可核对是否已经上传")
    receipt = Path(runtime) / "code-uploads" / (commit + ".json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    upload._write_json(
        receipt,
        dict(commit=commit, ref=ref, files=preview["files"], uploaded_at=upload._now(), transport="git-ssh"),
    )
    return "https://github.com/tianwuasd/project/commit/" + commit


def run(args, profile, interactive):
    from .launcher import personal_path
    from .prompts import ask

    runtime = personal_path(args.runtime or profile["runtime"])
    settings = network.load(runtime)
    network.show(settings)
    with network.use(settings), jobs.acquire_lock(runtime):
        if inspection.job_active(inspection.latest(runtime)):
            raise ValueError("请等待当前任务结束后上传代码")
        preview = prepare()
        print("上传目标：tianwuasd/project 的 codex/ct-panel-v2；仅上传已提交的 V2 代码。")
        print("提交：" + preview["repo"]["head"])
        print("待上传提交数：" + str(len(preview["commits"])))
        for name in preview["files"]:
            print("  " + name)
        if not preview["commits"]:
            print("代码已与远端一致。")
            return 0
        confirmed = args.yes_upload or interactive and ask("确认上传以上代码提交？[y/N]：").lower() == "y"
        if not confirmed:
            print("预览完成，尚未上传。确认后使用 --push-code --yes-upload，或在面板 w 中确认。")
            return 0
        print("代码上传完成：" + publish(preview, runtime))
        return 0
