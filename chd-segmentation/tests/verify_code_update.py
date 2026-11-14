import importlib.util
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


def fixture_commit(origin, tree, outside, parent=None):
    root = git(origin, "mktree", data=("040000 tree " + tree + "\tchd-segmentation\n100644 blob " + outside + "\tunrelated.md\n").encode()).decode().strip()
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
        first = fixture_commit(origin, git(origin, "rev-parse", BASE + "^{tree}").decode().strip(), outside)
        target_tree = source_tree(origin)
        second = fixture_commit(origin, target_tree, outside, first)
        git(origin, "update-ref", "refs/heads/main", first)
        repo_path, url = root / "code" / "git", origin.as_uri()
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", "--no-tags", "--branch", "main", url, str(repo_path)], check=True, stdout=subprocess.PIPE, timeout=60)
        repo = updater.GitRepository(repo_path, expected_urls={url})
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
        case.assertFalse(repo.run("status", "--porcelain=v1"))
        case.assertEqual(repo.run("rev-parse", "HEAD").decode().strip(), first)
        checks.append("empty no-checkout index becomes a clean sparse working tree and verified old installation is removed")
        timings["fixture_and_migration_seconds"] = perf_counter() - mark

        mark = perf_counter()
        before = repo.run("rev-parse", "HEAD")
        for lock in (runtime / ".code.lock", runtime / "runs" / "existing" / ".run.lock", repo_path / ".git" / "chd-code.lock"):
            with updater.file_lock(lock):
                with case.assertRaises(OSError):
                    updater.synchronize(repo, runtime, first)
        checks.append("active legacy entry, training task, and Git code lock prevent updates")
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
        updated = updater.synchronize(repo, runtime, second)
        case.assertEqual(updated["head"], second)
        case.assertEqual(repo.run("rev-parse", "HEAD").decode().strip(), second)
        case.assertEqual(repo.run("rev-parse", "origin/main").decode().strip(), second)
        case.assertEqual(repo.remote_head(), second)
        case.assertFalse(repo.run("status", "--porcelain=v1"))
        case.assertEqual({p.name for p in repo_path.iterdir()}, {".git", "chd-segmentation"})
        case.assertFalse((readme.parent / "launcher.py").exists())
        case.assertFalse((readme.parent / "code_versions").exists())
        checks.append("fast-forward makes HEAD, origin/main and remote main identical with only the requested folder checked out")
        local_objects = set(repo.run("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines())
        case.assertNotIn(outside, local_objects)
        with case.assertRaises(subprocess.CalledProcessError):
            repo.run("cat-file", "blob", outside)
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
        checks.append("repeated update reuses one fixed code directory without producing another code copy")
        timings["upgrade_and_entry_seconds"] = perf_counter() - mark
        result.update({"state": "passed", "fixture_commits": [first, second], "target_source_tree": target_tree, "check_count": len(checks)})
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
