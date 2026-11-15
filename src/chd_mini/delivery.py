from hashlib import sha256
import io
from pathlib import Path
import re
import subprocess
import zipfile

from .contracts import TOOL_ROOT, confined, file_sha, git


def public_paths(commit):
    paths = git(TOOL_ROOT, "ls-tree", "-r", "--name-only", commit).splitlines()
    roots = {".gitignore", ".gitattributes", "AGENTS.md", "README.md", "run.py", "update.py", "pyproject.toml"}
    for name in paths:
        if name not in roots and not name.startswith(("src/chd_mini/", "configs/", "docs/", "tests/")):
            raise ValueError("公开提交包含允许范围以外的文件：" + name)
        if ".local." in name or any(part in (".git", ".runs", ".venv", "data", "outputs") for part in Path(name).parts):
            raise ValueError("公开提交包含私有配置或运行文件：" + name)
    modes = git(TOOL_ROOT, "ls-tree", "-r", "--format=%(objectmode)", commit).splitlines()
    if any(mode not in ("100644", "100755") for mode in modes):
        raise ValueError("公开提交必须仅含普通代码文件")
    return paths


def code_archive(destination, dataset, method, date, commit="HEAD"):
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (dataset, method)) or not re.fullmatch(r"\d{8}", date):
        raise ValueError("交付命名需要数据集、方法和八位日期")
    commit = git(TOOL_ROOT, "rev-parse", "--verify", commit + "^{commit}")
    paths = public_paths(commit)
    payload = subprocess.check_output(["git", "-C", str(TOOL_ROOT), "archive", "--format=zip", commit])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if files != sorted(paths) or archive.comment.decode("ascii") != commit:
            raise ValueError("源码包与指定提交不一致")
    expected = sha256(payload).hexdigest()
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reused = None
    for candidate in root.rglob("*.zip"):
        path = confined(candidate, root)
        if path.stat().st_size == len(payload) and file_sha(path) == expected:
            reused = path
            break
    if reused is not None:
        path = reused
    else:
        path = confined(root / f"{dataset}-{method}-{date}_assistant3-code-{commit[:12]}.zip", root)
        if path.exists():
            raise FileExistsError("已有同名交付内容不同，拒绝覆盖")
        verified = False
        try:
            path.write_bytes(payload)
            if file_sha(path) != expected:
                raise ValueError("源码包写出后核验失败")
            verified = True
        finally:
            if not verified and path.exists():
                path.unlink()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": expected, "commit": commit,
            "source_files": len(paths), "reused": reused is not None, "scope": "public_code_only",
            "server_destination": "/data4/zhangruichen/code/chd-segmentation-assistant3",
            "uploaded_status": "not_uploaded", "github_branch": "codex/chd-segmentation-mini-workflow"}


def reference_file(path, server_destination, uploaded_status="unverified"):
    if uploaded_status not in ("uploaded", "not_uploaded", "unverified"):
        raise ValueError("上传状态必须有明确语义")
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("引用项必须是现有文件")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path),
            "server_destination": server_destination, "uploaded_status": uploaded_status,
            "copy_created": False, "action": "reuse_in_place"}
