import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    source = ROOT / "update.py"
    compile(source.read_text(encoding="utf-8"), str(source), "exec")
    spec = importlib.util.spec_from_file_location("aaoca_update_entry", source)
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    assert updater.SUBDIRECTORY == "aaoca-cta-reproduction"
    assert updater.BRANCH == "main"
    assert updater.allowed("update.py")
    assert updater.allowed("install_classification_dependencies.py")
    assert not updater.allowed("data/model.pt")
    assert not updater.allowed("course/inputs/project.server.json")
    assert not updater.allowed("../chd-segmentation/update.py")
    results = []
    for arguments, expected in ((["--help"], 0), ([], 2)):
        mark = perf_counter()
        result = subprocess.run([sys.executable, "-I", "-B", str(source), *arguments],
                                capture_output=True, timeout=30)
        assert result.returncode == expected, result.stderr.decode("utf-8", "replace")
        if arguments:
            assert b"--sha" in result.stdout and b"--force" in result.stdout
            assert b"--shared-updater" not in result.stdout
            assert b"--relocate" not in result.stdout and b"--remove-legacy" not in result.stdout
        results.append({"arguments": arguments, "returncode": result.returncode,
                        "seconds": perf_counter() - mark})
    print(json.dumps({"state": "passed", "checks": results, "seconds": perf_counter() - started,
                      "scope": "independent import, help, unpinned update rejection and path boundaries",
                      "files_created": 0, "artifact_bytes": 0, "network_requests": 0,
                      "cases_read": 0, "training_updates": 0, "gpu_operations": 0,
                      "memory_peak": "not_measured"}))


if __name__ == "__main__":
    main()
