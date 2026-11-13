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
SOURCE_VERSIONS = ["48048b1d84592ba3080d83dfe8352185ca1a6a5f", "80d96b1ee92665dc20237ce2afd47d3bb2956079"]


def git(directory, *args, data=None):
    return subprocess.run(["git", "-C", str(directory), "-c", "user.name=Codex", "-c", "user.email=codex@local", *args],
                          input=data, stdout=subprocess.PIPE, timeout=60, check=True).stdout


def fixture_commit(origin, source, parent=None):
    entries = {}
    for row in filter(None, git(origin, "ls-tree", "-z", source).split(b"\0")):
        header, name = row.split(b"\t", 1)
        entries[name] = header
    for name in ("update.py", "launcher.py"):
        blob = git(origin, "hash-object", "-w", "--stdin", data=(PROJECT / name).read_bytes()).strip()
        entries[name.encode()] = b"100644 blob " + blob
    body = b"".join(header + b"\t" + name + b"\0" for name, header in sorted(entries.items()))
    subtree = git(origin, "mktree", "-z", data=body).strip()
    outside = git(origin, "hash-object", "-w", "--stdin", data=Path(__file__).read_bytes()).strip()
    tree = git(origin, "mktree", data=b"040000 tree " + subtree + b"\tchd-segmentation\n100644 blob " + outside + b"\tunrelated.md\n").strip()
    args = ("commit-tree", tree.decode()) + (() if parent is None else ("-p", parent))
    commit = git(origin, *args, data=("Actual source version " + source + "\n").encode()).decode().strip()
    return commit, outside.decode()


def run(root, report):
    root, report = Path(root).resolve(), Path(report).resolve()
    if root.exists() or root == report or report.is_relative_to(root) or root.is_relative_to(report):
        raise ValueError("需要未使用的独立验证目录与外部报告")
    root.mkdir(parents=True)
    marker = root / "owner.json"
    updater.write_json(marker, {"scope": "code-sync", "pid": os.getpid()})
    started, checks, timings = perf_counter(), [], {}
    case = unittest.TestCase()
    result = {"state": "running", "source_versions": SOURCE_VERSIONS, "scope": "two code snapshots; no medical data or optimizer updates",
              "peak_rss_bytes": None, "peak_vram_bytes": None, "checks": checks, "timing": timings}
    try:
        mark = perf_counter()
        origin = root / "origin.git"
        subprocess.run(["git", "init", "--bare", str(origin)], check=True, stdout=subprocess.PIPE)
        git(origin, "config", "uploadpack.allowFilter", "true")
        git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
        git(origin, "fetch", "--no-tags", str(PROJECT), *SOURCE_VERSIONS)
        first, _ = fixture_commit(origin, SOURCE_VERSIONS[0])
        second, outside = fixture_commit(origin, SOURCE_VERSIONS[1], first)
        git(origin, "update-ref", "refs/heads/main", first)
        cache_path, url = root / "git-cache", origin.as_uri()
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", "--no-tags", "--branch", "main", url, str(cache_path)],
                       check=True, stdout=subprocess.PIPE, timeout=60)
        cache = updater.GitCache(cache_path, expected_urls={url})
        code, runtime = root / "code", root / "course"
        with case.assertRaises(subprocess.CalledProcessError):
            updater.install(cache, code, runtime, first)
        case.assertFalse((code / "current.json").exists())
        checks.append("missing blobs cannot trigger network access during offline archive")
        cache.obtain(first)
        pointer = updater.install(cache, code, runtime, first)
        case.assertEqual(pointer["state"], "installed_not_activated")
        case.assertFalse((code / "current.json").exists())
        case.assertTrue((code / "code_versions" / first / "run.py").is_file())
        checks.append("first installation from a no-checkout Git cache")
        timings["first_install_seconds"] = perf_counter() - mark

        mark = perf_counter()
        active = updater.install(cache, code, runtime, first, activate=True, python=sys.executable)
        case.assertEqual(active["active_sha"], first)
        readback = subprocess.run([sys.executable, "-B", str(code / "run.py"), "--code-info"],
                                  check=True, stdout=subprocess.PIPE, timeout=60)
        case.assertEqual(json.loads(readback.stdout)["sha"], first)
        version = subprocess.run([sys.executable, "-B", str(code / "run.py"), "--version"],
                                 check=True, stdout=subprocess.PIPE, timeout=60)
        case.assertEqual(version.stdout.decode().strip(), "0.1.0")
        checks.append("real environment API check and actual managed entry")
        timings["activation_seconds"] = perf_counter() - mark

        current = (code / "current.json").read_bytes()
        inventory_before = sorted(path.relative_to(code).as_posix() for path in code.rglob("*") if path.is_file())
        reused = updater.install(cache, code, runtime, first)
        case.assertEqual(reused["state"], "installed_current")
        case.assertEqual(inventory_before, sorted(path.relative_to(code).as_posix() for path in code.rglob("*") if path.is_file()))
        case.assertEqual(current, (code / "current.json").read_bytes())
        checks.append("repeat installation reuses the existing release")

        with updater.deployment_lock(runtime, shared=True):
            with case.assertRaises(OSError):
                updater.install(cache, code, runtime, first)
        with updater.file_lock(runtime / "runs" / "existing" / ".run.lock"):
            with case.assertRaises(OSError):
                updater.install(cache, code, runtime, first)
        checks.append("active managed entry and existing training locks refuse updates")

        readme = code / "code_versions" / first / "README.md"
        original = readme.read_bytes()
        readme.write_bytes(original + b"\nLocal edit\n")
        with case.assertRaises(ValueError):
            updater.inspect_release(code, updater.read_json(code / "current.json"))
        with case.assertRaises(ValueError):
            updater.install(cache, code, runtime, first)
        case.assertEqual(current, (code / "current.json").read_bytes())
        readme.write_bytes(original)
        checks.append("local source edits are rejected while the pointer stays unchanged")

        with case.assertRaises(ValueError):
            cache.obtain(second)
        case.assertEqual(current, (code / "current.json").read_bytes())
        checks.append("remote branch mismatch refuses a changed target")

        mark = perf_counter()
        git(origin, "update-ref", "refs/heads/main", second, first)
        cache.obtain(second)
        local = set(cache.run("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().splitlines())
        case.assertNotIn(outside, local)
        staged = updater.install(cache, code, runtime, second)
        case.assertEqual(staged["state"], "installed_not_activated")
        case.assertEqual(current, (code / "current.json").read_bytes())
        with case.assertRaises(FileNotFoundError):
            updater.install(cache, code, runtime, second, activate=True, python=root / "missing-python")
        case.assertEqual(current, (code / "current.json").read_bytes())
        checks.append("a new version and missing environment keep the old active pointer")
        # 本地基底环境已核对缺少 MONAI，用实际依赖失败验证激活边界。
        with case.assertRaises(subprocess.CalledProcessError):
            updater.install(cache, code, runtime, second, activate=True, python=sys._base_executable)
        case.assertEqual(current, (code / "current.json").read_bytes())
        checks.append("a real missing-MONAI environment cannot change the active pointer")
        upgraded = updater.install(cache, code, runtime, second, activate=True, python=sys.executable)
        case.assertEqual(upgraded["active_sha"], second)
        case.assertTrue((code / "code_versions" / first).is_dir())
        readback = subprocess.run([sys.executable, "-B", str(code / "run.py"), "--code-info"],
                                  check=True, stdout=subprocess.PIPE, timeout=60)
        case.assertEqual(json.loads(readback.stdout)["sha"], second)
        checks.append("fast-forward activation and real entry select the new immutable version")
        checks.append("unrelated repository blobs were never downloaded")
        timings["upgrade_seconds"] = perf_counter() - mark

        foreign = root / "foreign-code"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("Existing user file", encoding="utf-8")
        with case.assertRaises(FileExistsError):
            updater.install(cache, foreign, runtime, second)
        case.assertEqual((foreign / "keep.txt").read_text(encoding="utf-8"), "Existing user file")
        checks.append("unknown target files are preserved")
        case.assertFalse(list((runtime / "updates").glob("stage-*")))
        case.assertFalse(list((code / "code_versions").glob(".candidate-*")))
        checks.append("update staging is cleaned after success and failure")
        result.update({"state": "passed", "fixture_commits": [first, second], "check_count": len(checks)})
    finally:
        if sys.exception() is not None:
            result.update({"state": "failed", "error": str(sys.exception())})
        result["seconds"] = perf_counter() - started
        files = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
                 for path in root.rglob("*") if path.is_file()]
        result.update({"files_before_cleanup": files, "file_count": len(files),
                       "bytes_before_cleanup": sum(row["bytes"] for row in files),
                       "technical_snapshots_per_second": 2 / result["seconds"] if result["state"] == "passed" else None})
        updater.write_json(report, result)
        if root.parent != Path(os.environ["CHD_CODE_TEST_ROOT"]).resolve().parent or root.is_symlink():
            raise ValueError("测试清理目录归属异常")
        if updater.read_json(marker) != {"scope": "code-sync", "pid": os.getpid()}:
            raise ValueError("测试目录归属记录变化")
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
