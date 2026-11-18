import argparse
import ast
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib

sys.dont_write_bytecode = True


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def repository(root):
    root = Path(root).resolve(strict=True)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"]["name"] != "chd-mini-workflow":
        raise ValueError("更新器只允许助手3独立项目")
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("助手3必须使用自己的独立Git根目录")
    return root


def update(root, check=False, remote="origin", ref=None):
    root = repository(root)
    branch = git(root, "branch", "--show-current")
    if not branch:
        raise ValueError("更新需要具名独立分支")
    before = git(root, "rev-parse", "HEAD")
    result = {"tool_root": str(root), "commit": before, "branch": branch,
              "remotes": git(root, "remote").splitlines(),
              "production_projects_updated": False, "modified": bool(git(root, "status", "--porcelain"))}
    if check:
        return result
    from filelock import FileLock
    if result["modified"]:
        raise ValueError("本工具存在未提交修改，拒绝更新")
    if remote not in result["remotes"]:
        raise ValueError("尚未配置助手3自己的远端，不能使用生产项目远端代替")
    lock_root = root / ".runs"
    if lock_root.is_symlink() or (os.name == "nt" and lock_root.exists() and
                                  lock_root.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError("运行目录不能是链接")
    lock_root.mkdir(exist_ok=True)
    with FileLock(str(lock_root / ".tool.lock"), timeout=0):
        target_ref = ref or branch
        subprocess.run(["git", "check-ref-format", "--branch", target_ref], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(root), "fetch", "--no-tags", remote, target_ref], check=True)
        target = git(root, "rev-parse", "FETCH_HEAD")
        metadata = tomllib.loads(git(root, "show", target + ":pyproject.toml"))
        if metadata["project"]["name"] != "chd-mini-workflow":
            raise ValueError("远端提交不属于助手3独立工具")
        required = {"run.py", "update.py", "pyproject.toml", "src/chd_mini/cli.py"}
        paths = git(root, "ls-tree", "-r", "--name-only", target).splitlines()
        allowed_files = {".gitignore", ".gitattributes", "AGENTS.md", "README.md", "run.py", "update.py", "pyproject.toml"}
        if not required <= set(paths) or any(p not in allowed_files and not p.startswith(("src/chd_mini/", "configs/", "docs/", "tests/")) for p in paths):
            raise ValueError("远端提交含工具范围以外的文件或缺少必要入口")
        for entry in git(root, "ls-tree", "-r", "--format=%(objectmode)", target).splitlines():
            if entry not in ("100644", "100755"):
                raise ValueError("工具发布不允许链接或子模块")
        for path in paths:
            if path.endswith(".py"):
                ast.parse(git(root, "show", target + ":" + path), filename=path)
        if git(root, "rev-parse", "HEAD") != before or git(root, "status", "--porcelain"):
            raise ValueError("下载期间本工具内容已改变，拒绝覆盖")
        subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", before, target], check=True)
        subprocess.run(["git", "-C", str(root), "merge", "--ff-only", "--no-edit", target], check=True)
        if git(root, "rev-parse", "HEAD") != target or git(root, "status", "--porcelain"):
            raise ValueError("工具更新后的提交或内容不符")
        return {**result, "commit": target, "previous": before, "state": "updated"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只更新助手3独立工具代码")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--ref")
    args = parser.parse_args()
    print(json.dumps(update(Path(__file__).resolve().parent, args.check, args.remote, args.ref), ensure_ascii=False, indent=2))
