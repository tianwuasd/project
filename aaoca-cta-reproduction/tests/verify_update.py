import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from update import shared_updater


def main():
    parser = argparse.ArgumentParser(description="实际GitHub提交的独立代码同步与清理；不读取病例")
    parser.add_argument("--shared-updater", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                             check=True, capture_output=True).stdout.split(b"\0")
    source_files = [ROOT / name.decode("utf-8") for name in tracked if name]
    estimate_bytes = sum(p.stat().st_size for p in source_files) + 2 * 2**20
    if len(source_files) + 32 > 60 or estimate_bytes > 20 * 2**20:
        raise OSError("实际源码数量加Git元数据预估超过本轮测试预算")
    task = Path(args.work).resolve()
    if task.exists():
        raise FileExistsError("本轮独占同步目录已经存在")
    task.mkdir(parents=True)
    code, runtime = task / "code", task / "course"
    code.mkdir()
    start = perf_counter()
    success = False
    owned = []
    try:
        env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")

        def budget():
            files = [p for p in task.rglob("*") if p.is_file()]
            if len(files) > 60 or sum(p.stat().st_size for p in files) > 20 * 2**20:
                raise OSError("同步测试达到文件数或磁盘上限")

        def git(*command):
            result = subprocess.run(["git", "-C", str(code), "-c", "http.followRedirects=false", *command],
                                    env=env, check=True, capture_output=True, timeout=180)
            budget()
            return result.stdout

        git("init", "--initial-branch=" + args.branch, "--template=")
        git("remote", "add", "origin", "https://github.com/tianwuasd/project.git")
        git("fetch", "--depth=1", "--no-tags", "--filter=blob:none", "origin", args.base_sha)
        git("update-ref", "refs/heads/" + args.branch, args.base_sha)
        git("update-ref", "refs/remotes/origin/" + args.branch, args.base_sha)
        updater = shared_updater(args.shared_updater)
        updater.BRANCH = args.branch
        repo = updater.GitRepository(code)
        repo.validate(force=True)
        chosen = repo.choose_source([updater.OFFICIAL], expected=args.sha, timeout=30, official=updater.OFFICIAL)
        first = updater.synchronize(repo, runtime, chosen, force=True)
        budget()
        before = {p.relative_to(code): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in (code / updater.SUBDIRECTORY).rglob("*") if p.is_file()}
        second = updater.synchronize(repo, runtime, chosen, force=True)
        budget()
        after = {p.relative_to(code): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in (code / updater.SUBDIRECTORY).rglob("*") if p.is_file()}
        assert before == after and first["sha"] == second["sha"] == args.sha
        assert not (code / "chd-segmentation").exists()
        assert git("diff", "--name-only", "HEAD", "--", updater.SUBDIRECTORY).strip() == b""
        assert git("rev-parse", "HEAD").decode().strip() == args.sha
        entry = subprocess.run([sys.executable, "-X", "utf8", "-B",
                                str(code / updater.SUBDIRECTORY / "run.py"), "--code-info"],
                               check=True, capture_output=True, text=True, encoding="utf-8", timeout=60)
        loaded = json.loads(entry.stdout)
        assert loaded["sha"] == args.sha and not loaded["modified"]
        assert Path(loaded["code"]).resolve() == (code / updater.SUBDIRECTORY).resolve()
        success = True
    finally:
        owned = [p for p in task.rglob("*") if p.is_file()]
        if any(p.is_symlink() or not p.resolve().is_relative_to(task) for p in owned):
            raise RuntimeError("同步测试文件归属异常，保留目录")
        size = sum(p.stat().st_size for p in owned)
        print(json.dumps({"synchronization": "passed" if success else "failed", "seconds": perf_counter() - start,
                          "files": len(owned), "bytes": size, "training_updates": 0,
                          "scope": "Windows Git initial install and exact repeat using actual remote commits"},
                         ensure_ascii=False), flush=True)
        # 所有Git子进程均已结束，路径已核实为本轮新建独占目录。
        for path in owned:
            path.chmod(path.stat().st_mode | stat.S_IWRITE)
        shutil.rmtree(task)
        print(json.dumps({"state": "passed" if success else "failed", "cleanup": {"deleted_files": len(owned), "released_bytes": size,
                                      "task_directory_absent": not task.exists()}}), flush=True)
        if len(owned) > 60 or size > 20 * 2**20:
            raise OSError("同步测试超过本轮60文件或20MiB预算")


if __name__ == "__main__":
    main()
