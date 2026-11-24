import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aaoca_reproduction.audit import load_frozen
from aaoca_reproduction.records import load_config, read_json


def main():
    parser = argparse.ArgumentParser(description="复用现有Windows服务器仿真的只读路径接口")
    parser.add_argument("--simulator", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = perf_counter()
    simulator = Path(args.simulator).resolve(strict=True)
    sys.path.insert(0, str(simulator.parent))
    spec = importlib.util.spec_from_file_location("aaoca_existing_server_simulator", simulator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = read_json(simulator.parent / "profiles" / "zmic44.json")
    config = load_config(args.config)
    base = profile["paths"]["root"]
    mappings = {base + "/code/aaoca-cta-reproduction": str(ROOT),
                base + "/course/aaoca-cta-reproduction": str(config["project_root"]),
                base + "/data/AAOCA": str(config["aaoca_root"]),
                base + "/data/ImageCHD_dataset": str(config["imagechd_root"])}
    state = {"mappings": mappings}
    assert Path(module.resolve_path(base + "/code/aaoca-cta-reproduction/run.py", state)) == ROOT / "run.py"
    manifest = load_frozen(config)
    selected = [row for row in manifest["cases"] if row["case_id"] in config["smoke_cases"]]
    assert len(selected) == 2
    for row in selected:
        root = config[row["dataset"] + "_root"]
        prefix = "AAOCA" if row["dataset"] == "aaoca" else "ImageCHD_dataset"
        for kind in ("image", "label"):
            actual = Path(row[kind]).resolve(strict=True)
            server = base + "/data/" + prefix + "/" + actual.relative_to(root).as_posix()
            assert Path(module.resolve_path(server, state)) == actual
    memory = psutil.virtual_memory()
    print(json.dumps({"state": "passed", "scope": "existing Windows simulator path API and actual local host read",
                      "cases": 2, "paths": 5, "seconds": perf_counter() - started,
                      "host_total_bytes": memory.total, "host_available_bytes": memory.available,
                      "profile_live_verified": profile["live_verified"], "generated_files": 0,
                      "training_updates": 0, "cuda_nccl_tested": False,
                      "runtime_memory_guard_implemented": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
