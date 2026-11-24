import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aaoca_reproduction.audit import check_training, load_frozen
from aaoca_reproduction.inputs import bound_manifest, portable_manifest
from aaoca_reproduction.records import load_config, signature


def main():
    parser = argparse.ArgumentParser(description="既有两例真实清单的只读迁移校验，不生成测试文件")
    parser.add_argument("--config", default="project.local.json")
    args = parser.parse_args()
    start = perf_counter()
    config = load_config(args.config)
    stamp = signature(config["project_root"] / "private" / "manifest.json")
    manifest = load_frozen(config)
    selected = [row for row in manifest["cases"] if row["case_id"] in config["smoke_cases"]]
    assert len(selected) == 2 and {row["dataset"] for row in selected} == {"aaoca", "imagechd"}
    subset = dict(manifest, cases=selected)
    portable = portable_manifest(subset, config)
    assert all(not Path(row[key]).is_absolute() for row in portable["cases"] for key in ("image", "label"))
    # 用真实存在的子目录作为不同绑定根，修改相对引用后核对结果不变。
    relocated = deepcopy(portable)
    roots = dict(config)
    for row in relocated["cases"]:
        base = Path(config[row["dataset"] + "_root"]).parent
        roots[row["dataset"] + "_root"] = base
        for key in ("image", "label"):
            original = next(item for item in selected if item["case_id"] == row["case_id"])
            row[key] = Path(original[key]).resolve().relative_to(base).as_posix()
    rebound = bound_manifest(relocated, roots)
    assert rebound["cases"] == selected
    assert not check_training(config, manifest)["ready"]
    assert signature(config["project_root"] / "private" / "manifest.json") == stamp
    print(json.dumps({"state": "passed", "cases": 2, "roles_and_aliases_unchanged": True,
                      "different_root_binding": True, "training_updates": 0, "generated_files": 0,
                      "seconds": perf_counter() - start, "memory_peak": "未测量", "vram_peak": "未测量"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
