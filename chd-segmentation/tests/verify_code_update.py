import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from time import perf_counter
import unittest


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chd_code_update", PROJECT / "update.py")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)
BASE = "78e7ef6ec8620779c2e06200400822f293287f90"


def git(directory, *args, data=None):
    return subprocess.run(["git", "-C", str(directory), "-c", "user.name=Codex", "-c", "user.email=codex@local", *args], input=data, stdout=subprocess.PIPE, timeout=60, check=True).stdout


def source_tree(origin):
    files = {}
    for name in filter(None, git(PROJECT, "ls-files", "-z").decode().split("\0")):
        path = PROJECT / name
        if path.is_file():
            files[name] = git(origin, "hash-object", "-w", "--stdin", data=path.read_bytes()).decode().strip()

    def subtree(prefix):
        entries = {}
        directories = set()
        for name, blob in files.items():
            if name.startswith(prefix):
                leaf, separator, _ = name[len(prefix):].partition("/")
                if separator:
                    directories.add(leaf)
                else:
                    entries[leaf] = "100644 blob " + blob
        for leaf in directories:
            entries[leaf] = "040000 tree " + subtree(prefix + leaf + "/")
        body = b"".join((value + "\t" + name).encode() + b"\0" for name, value in sorted(entries.items()))
        return git(origin, "mktree", "-z", data=body).decode().strip()

    return subtree("")


def fixture_commit(origin, tree, outside, unchanged, parent=None):
    other = git(origin, "mktree", data=("100644 blob " + outside + "\tREADME.md\n").encode()).decode().strip()
    root = git(origin, "mktree", data=("040000 tree " + tree + "\tchd-segmentation\n040000 tree " + other
               + "\tdataset1282_analysis\n100644 blob " + unchanged + "\tunrelated.md\n").encode()).decode().strip()
    args = ("commit-tree", root) + (() if parent is None else ("-p", parent))
    return git(origin, *args, data=b"Actual application source snapshot\n").decode().strip()


def run(root, report):
    root, report = Path(root).resolve(), Path(report).resolve()
    if root.exists() or report.is_relative_to(root) or root.is_relative_to(report):
        raise ValueError("测试目录必须独立且尚未使用")
    root.mkdir(parents=True)
    marker = root / "owner.json"
    updater.write_json(marker, {"scope": "code-sync", "pid": os.getpid()})
    started, checks, timings = perf_counter(), [], {}
    case = unittest.TestCase()
    result = {"state": "running", "source_versions": [BASE, "current working source"], "scope": "two actual code snapshots; no medical data or optimizer updates", "peak_rss_bytes": None, "peak_vram_bytes": None, "checks": checks, "timing": timings}
    try:
        mark = perf_counter()
        origin = root / "origin.git"
        subprocess.run(["git", "init", "--bare", str(origin)], check=True, stdout=subprocess.PIPE)
        git(origin, "config", "uploadpack.allowFilter", "true")
        git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
        git(origin, "fetch", "--no-tags", str(PROJECT), BASE)
        outside_data = git(PROJECT, "show", "48048b1d84592ba3080d83dfe8352185ca1a6a5f:README.md")
        outside = git(origin, "hash-object", "-w", "--stdin", data=outside_data).decode().strip()
        next_outside_data = git(PROJECT, "show", "80d96b1ee92665dc20237ce2afd47d3bb2956079:README.md")
        next_outside = git(origin, "hash-object", "-w", "--stdin", data=next_outside_data).decode().strip()
        first = fixture_commit(origin, git(origin, "rev-parse", BASE + "^{tree}").decode().strip(), outside, outside)
        target_tree = source_tree(origin)
        second = fixture_commit(origin, target_tree, next_outside, outside, first)
        git(origin, "update-ref", "refs/heads/main", first)
        repo_path, url = root / "code" / "git", origin.as_uri()
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", "--no-tags", "--branch", "main", url, str(repo_path)], check=True, stdout=subprocess.PIPE, timeout=60)
        repo = updater.GitRepository(repo_path, expected_urls={url})
        sibling = root / "code" / "dataset1282_analysis" / "README.md"
        untracked = repo_path / "local-project" / "README.md"
        for path in (sibling, untracked):
            path.parent.mkdir(parents=True)
            path.write_bytes(outside_data)
        untouched = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (sibling, untracked)}
        case.assertTrue(repo.validate())
        case.assertEqual(repo.choose_source([(root / "missing-mirror").as_uri(), url], expected=first, timeout=5), first)
        with case.assertRaises(ConnectionError):
            repo.choose_source([(root / "missing-mirror").as_uri()], timeout=5)
        with case.assertRaises(ValueError):
            updater.mirror_url("https://credential@mirror.example")
        repo.choose_source([url], expected=first, timeout=5)
        checks.append("actual filtered Git endpoint selection and unavailable or credential-bearing endpoint rejection")

        # 使用真实旧更新器产生旧安装，覆盖首次迁移与删除流程。
        old_path = root / "legacy_update.py"
        old_path.write_bytes(git(PROJECT, "show", BASE + ":update.py"))
        old_spec = importlib.util.spec_from_file_location("actual_old_update", old_path)
        old = importlib.util.module_from_spec(old_spec)
        old_spec.loader.exec_module(old)
        runtime, legacy = root / "course", root / "code" / "chd-segmentation"
        cache = old.GitCache(repo_path, expected_urls={url})
        cache.obtain(first)
        old.install(cache, legacy, runtime, first)
        retained = legacy / "keep.txt"
        retained.write_text("Existing user file", encoding="utf-8")
        with case.assertRaises(ValueError):
            updater.synchronize(repo, runtime, first, remove_legacy=True)
        case.assertTrue(retained.exists())
        retained.unlink()
        checks.append("unknown files in the actual old installation prevent deletion")
        installed = updater.synchronize(repo, runtime, first, remove_legacy=True)
        case.assertFalse(legacy.exists())
        case.assertGreater(installed["legacy_cleanup"]["deleted_files"], 0)
        case.assertFalse(repo.validate())
        case.assertFalse(repo.run("status", "--porcelain=v1", "--", "chd-segmentation"))
        case.assertEqual(repo.run("rev-parse", "HEAD").decode().strip(), first)
        checks.append("empty no-checkout index becomes a clean sparse working tree and verified old installation is removed")
        for path, snapshot in untouched.items():
            case.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)
        checks.append("sibling projects and untracked projects inside the Git directory are preserved during initial migration")
        case.assertNotIn(outside, set(repo.run("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines()))
        # 显式准备其他项目的真实文件及本地改动，验证后续同步不会写入它们。
        repo.run("fetch", repo.transport, "--no-tags", "--no-write-fetch-head", "--filter=blob:none", "--stdin", data=(outside + "\n").encode(), network=True)
        other_names = b"dataset1282_analysis/README.md\0unrelated.md\0"
        repo.run("update-index", "--no-skip-worktree", "-z", "--stdin", data=other_names)
        repo.run("checkout-index", "-z", "--stdin", data=other_names)
        tracked = repo_path / "dataset1282_analysis" / "README.md"
        staged = repo_path / "unrelated.md"
        tracked.write_bytes(tracked.read_bytes() + b"\nLocal working change\n")
        staged.write_bytes(staged.read_bytes() + b"\nLocal staged change\n")
        git(repo_path, "add", "--", "unrelated.md")
        staged_blob = repo.run("rev-parse", ":unrelated.md")
        for path in (tracked, staged):
            untouched[path] = (path.read_bytes(), path.stat().st_mtime_ns)
        timings["fixture_and_migration_seconds"] = perf_counter() - mark

        mark = perf_counter()
        before = repo.run("rev-parse", "HEAD")
        for lock in (runtime / ".code.lock", runtime / "runs" / "existing" / ".run.lock", repo_path / ".git" / "chd-code.lock"):
            with updater.file_lock(lock):
                with case.assertRaises(OSError):
                    updater.synchronize(repo, runtime, first)
        checks.append("active legacy entry, training task, and Git code lock prevent updates")
        index_lock = repo_path / ".git" / "index.lock"
        with index_lock.open("xb") as stream:
            stream.write(b"Existing external Git operation")
        with case.assertRaises(subprocess.CalledProcessError):
            updater.synchronize(repo, runtime, first)
        case.assertEqual(index_lock.read_bytes(), b"Existing external Git operation")
        case.assertEqual(repo.run("rev-parse", "HEAD"), before)
        index_lock.unlink()
        checks.append("an existing external Git index lock is preserved and prevents synchronization")
        readme = repo_path / "chd-segmentation" / "README.md"
        original = readme.read_bytes()
        readme.write_bytes(original + b"\nLocal edit\n")
        with case.assertRaises(ValueError):
            updater.synchronize(repo, runtime, first)
        readme.write_bytes(original)
        private = readme.parent / "local.yaml"
        private.write_text("Existing user configuration", encoding="utf-8")
        with case.assertRaises(ValueError):
            updater.synchronize(repo, runtime, first)
        private.unlink()
        case.assertEqual(repo.run("rev-parse", "HEAD"), before)
        checks.append("local modifications and untracked files are preserved without moving HEAD")
        git(origin, "update-ref", "refs/heads/main", second, first)
        with case.assertRaises(ValueError):
            updater.synchronize(repo, runtime, first)
        case.assertEqual(repo.run("rev-parse", "HEAD"), before)
        checks.append("remote movement rejects a stale target without changing the working tree")
        timings["protection_seconds"] = perf_counter() - mark

        mark = perf_counter()
        repo.choose_source([url], expected=second, timeout=5)
        git(repo_path, "add", "--", "dataset1282_analysis/README.md")
        staged_before = repo.run("ls-files", "--stage", "-z")
        with case.assertRaises(ValueError):
            updater.synchronize(repo, runtime, second)
        case.assertEqual(repo.run("rev-parse", "HEAD"), before)
        case.assertEqual(repo.run("ls-files", "--stage", "-z"), staged_before)
        for path, snapshot in untouched.items():
            case.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)
        # 仅恢复该测试夹具的暂存记录，保留工作文件，以继续检查普通同步。
        repo.run("update-index", "--cacheinfo", "100644," + outside + ",dataset1282_analysis/README.md")
        checks.append("a conflicting staged change in another project preserves HEAD, index entries, and all working files")
        updated = updater.synchronize(repo, runtime, second)
        case.assertEqual(updated["head"], second)
        case.assertEqual(repo.run("rev-parse", "HEAD").decode().strip(), second)
        case.assertEqual(repo.run("rev-parse", "origin/main").decode().strip(), second)
        case.assertEqual(repo.remote_head(), second)
        case.assertFalse(repo.run("status", "--porcelain=v1", "--", "chd-segmentation"))
        case.assertEqual({p.name for p in repo_path.iterdir()}, {".git", "chd-segmentation", "dataset1282_analysis", "local-project", "unrelated.md"})
        for path, snapshot in untouched.items():
            case.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)
        case.assertEqual(repo.run("rev-parse", ":unrelated.md"), staged_blob)
        case.assertIn(b"M dataset1282_analysis/README.md", repo.run("status", "--porcelain=v1", "--", "dataset1282_analysis"))
        checks.append("tracked working changes and staged changes of other projects retain identical content and modification times")
        case.assertFalse((readme.parent / "launcher.py").exists())
        case.assertFalse((readme.parent / "code_versions").exists())
        checks.append("fast-forward makes HEAD, origin/main and remote main identical with only the requested folder checked out")
        local_objects = set(repo.run("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines())
        case.assertNotIn(next_outside, local_objects)
        with case.assertRaises(subprocess.CalledProcessError):
            repo.run("cat-file", "blob", next_outside)
        checks.append("unrelated repository blob is absent and cannot be lazily fetched by offline commands")
        entry = readme.parent / "run.py"
        with updater.code_guard(entry):
            with case.assertRaises(OSError):
                updater.synchronize(repo, runtime, second)
        info = subprocess.run([sys.executable, "-B", str(entry), "--code-info"], check=True, stdout=subprocess.PIPE, timeout=60)
        case.assertEqual(json.loads(info.stdout)["sha"], second)
        case.assertFalse(json.loads(info.stdout)["modified"])
        case.assertEqual(Path(json.loads(info.stdout)["code"]), readme.parent)
        version = subprocess.run([sys.executable, "-B", str(entry), "--version"], check=True, stdout=subprocess.PIPE, timeout=60)
        case.assertEqual(version.stdout.decode().strip(), "0.1.0")
        checks.append("actual business entry reads the synchronized code path and SHA and holds the update guard")
        repeated = updater.synchronize(repo, runtime, second, remove_legacy=True)
        case.assertEqual(repeated["sha"], second)
        case.assertEqual(repeated["legacy_cleanup"]["deleted_files"], 0)
        for path, snapshot in untouched.items():
            case.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)
        case.assertFalse(list((repo_path / ".git").glob("chd-index-*")))
        case.assertFalse((repo_path / ".git" / "index.lock").exists())
        checks.append("repeated update reuses one fixed code directory without producing another code copy")
        timings["upgrade_and_entry_seconds"] = perf_counter() - mark
        result.update({"state": "passed", "fixture_commits": [first, second], "target_source_tree": target_tree, "check_count": len(checks), "preserved_files": [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(snapshot[0]).hexdigest(), "mtime_ns": snapshot[1]} for path, snapshot in untouched.items()]})
    finally:
        if sys.exception() is not None:
            result.update({"state": "failed", "error": str(sys.exception())})
        result["seconds"] = perf_counter() - started
        files = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size} for path in root.rglob("*") if path.is_file()]
        result.update({"files_before_cleanup": files, "file_count": len(files), "bytes_before_cleanup": sum(row["bytes"] for row in files), "technical_snapshots_per_second": 2 / result["seconds"] if result["state"] == "passed" else None})
        updater.write_json(report, result)
        if root != Path(os.environ["CHD_CODE_TEST_ROOT"]).resolve() or root.is_symlink() or updater.read_json(marker) != {"scope": "code-sync", "pid": os.getpid()}:
            raise ValueError("测试目录归属变化，停止清理")
        if os.name == "nt":
            for path in root.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IREAD | stat.S_IWRITE)
        shutil.rmtree(root)
        result["cleanup"] = {"deleted_files": len(files), "released_bytes": result["bytes_before_cleanup"], "root_exists": root.exists()}
        updater.write_json(report, result)
    print(json.dumps({key: value for key, value in result.items() if key != "files_before_cleanup"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(os.environ["CHD_CODE_TEST_ROOT"], os.environ["CHD_CODE_TEST_REPORT"])
