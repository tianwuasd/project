import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from aaoca_reproduction.records import write_json

TARGET = ROOT / ".deps" / "torchio0196"
TEMP = Path("D:/projects/back/tmp/aaoca-torchio0196")
REPORT = ROOT / "reports" / "torchio-installation-20261121.json"


def inventory(root):
    return [{"path": str(p), "bytes": p.stat().st_size} for p in root.rglob("*") if p.is_file()]


def clean(root):
    allowed = {TARGET.resolve(), TEMP.resolve()}
    if root.resolve() not in allowed or not root.exists():
        raise ValueError("清理路径不属于本轮安装目录")
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("安装目录含外部链接，停止清理")
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    root.rmdir()


def main():
    if TARGET.exists() or TEMP.exists() or REPORT.exists():
        raise FileExistsError("本批安装目录或报告已存在，拒绝覆盖")
    spec = importlib.util.spec_from_file_location("aaoca_existing_usage", "D:/projects/test/usage.py")
    usage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(usage)
    TEMP.mkdir(parents=True)
    (TEMP / ".gitignore").write_text("*\n", encoding="utf-8")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, TEMP=str(TEMP), TMP=str(TEMP), TMPDIR=str(TEMP),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       MPLCONFIGDIR=str(TEMP / "matplotlib"), XDG_CACHE_HOME=str(TEMP / "cache"))
    command = [sys.executable, "-B", "-m", "pip", "--isolated", "install", "--target", str(TARGET),
               "--no-deps", "--no-compile", "--no-cache-dir", "--only-binary", ":all:",
               "--disable-pip-version-check", "--index-url", "https://pypi.org/simple",
               "--proxy", "http://127.0.0.1:7897", "-r", str(ROOT / "configs/torchio0196-windows-overlay.txt")]
    started = monotonic()
    peak_files, peak_bytes = 0, 0
    success = False
    process = None
    import_result = None
    log = TEMP / "install.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                samples = [usage.live_usage(p) for p in (TARGET, TEMP)]
                count, size = sum(x["files"] for x in samples), sum(x["bytes"] for x in samples)
                peak_files, peak_bytes = max(peak_files, count), max(peak_bytes, size)
                if count > 2000 or size > 64 * 2**20 or monotonic() - started > 540:
                    process.kill()
                    process.wait()
                    raise RuntimeError("安装超出文件/磁盘/时间预算")
                sleep(0.1)
            if process.wait() != 0:
                raise RuntimeError("固定依赖安装失败，详见安装报告")
        check = "import sys,json,importlib.metadata as m;sys.path.insert(0,sys.argv[1]);import torchio,torch,numpy,SimpleITK;print(json.dumps({'torchio':torchio.__version__,'module':torchio.__file__,'torch':torch.__version__,'numpy':numpy.__version__,'packages':{n:m.version(n) for n in ['torchio','Deprecated','humanize','typer','click','shellingham','rich','wrapt']}}))"
        with (TEMP / "import.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-B", "-c", check, str(TARGET)],
                                       env=environment, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if monotonic() - started > 570:
                    process.kill()
                    process.wait()
                    raise RuntimeError("导入检查超出安装总时限")
                sleep(0.1)
            if process.wait() != 0:
                raise RuntimeError("真实TorchIO导入失败")
        import_result = json.loads((TEMP / "import.log").read_text(encoding="utf-8").splitlines()[-1])
        if import_result["torchio"] != "0.19.6" or not Path(import_result["module"]).is_relative_to(TARGET):
            raise AssertionError("实际导入版本或隔离路径不符")
        success = True
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        failure = sys.exc_info()[1]
        temporary = inventory(TEMP)
        installed = inventory(TARGET)
        total_files = len(temporary) + len(installed)
        total_bytes = sum(x["bytes"] for x in temporary + installed)
        if total_files > 2000 or total_bytes > 64 * 2**20:
            success = False
        record = {"state": "passed" if success else "failed", "seconds": monotonic() - started,
                  "python": sys.executable, "target": str(TARGET), "versions": import_result,
                  "download_wheel_bytes_from_metadata": 741579,
                  "observed_peak_files": max(peak_files, total_files),
                  "observed_peak_bytes": max(peak_bytes, total_bytes), "installed": installed,
                  "installed_files": len(installed), "installed_bytes": sum(x["bytes"] for x in installed),
                  "temporary": temporary, "error": str(failure) if failure else None,
                  "log_tail": log.read_text(encoding="utf-8")[-16000:] if log.exists() else ""}
        write_json(REPORT, record)
        clean(TEMP)
        if not success and TARGET.exists():
            clean(TARGET)
        record["cleanup"] = {"temporary_absent": not TEMP.exists(), "target_retained": success,
                             "deleted_files": len(temporary) + (0 if success else len(installed)),
                             "released_bytes": sum(x["bytes"] for x in temporary + ([] if success else installed))}
        record["seconds"] = monotonic() - started
        write_json(REPORT, record)
        if REPORT.stat().st_size > 2 * 2**20:
            raise OSError("安装报告超2MiB")
    print(json.dumps({k: record[k] for k in ("state", "seconds", "installed_files", "installed_bytes", "cleanup")}, ensure_ascii=False))
    if not success:
        raise RuntimeError("安装未通过预算或实际导入检查")


if __name__ == "__main__":
    main()
