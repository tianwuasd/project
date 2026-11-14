import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd_segmentation.artifacts.store import file_digest, write_json
from chd_segmentation.config import DEFAULTS
from chd_segmentation.datasets.splits import apply_split


def run(args):
    started = perf_counter()
    original = json.loads(Path(args.chd_manifest).read_text(encoding="utf-8"))
    manifest = {"dataset": "chd1282", "cases": original}
    frozen = json.loads(Path(args.chd_cohort).read_text(encoding="utf-8"))
    before = deepcopy(manifest)
    config = deepcopy(DEFAULTS)
    config["split"]["path"] = args.chd_cohort
    selected = apply_split(manifest, config)
    assert manifest == before
    assert Counter(r["roles"][0] for r in selected["cases"]) == {"train": 598, "validation": 130}
    patients = {}
    for row in selected["cases"]:
        role = row["roles"][0]
        assert row["patient"] == frozen["cases"][row["case_id"]]["patient"]
        assert patients.setdefault(row["patient"], role) == role
    assert not {r["case_id"] for r in selected["cases"]} & {key for key, row in frozen["cases"].items() if row["role"] == "source_test"}
    image = json.loads(Path(args.image_manifest).read_text(encoding="utf-8"))
    config["split"]["path"] = args.image_splits
    image_selected = apply_split({"dataset": "imagechd", "cases": image["cases"]}, config)
    assert Counter(r["roles"][0] for r in image_selected["cases"]) == {"train": 66, "validation": 10, "test": 33}
    assert sum(r["cohort"] == "auxiliary" for r in image_selected["cases"]) == 13
    assert image_selected["split_source"]["unit"] == "case"
    # 主清单包含各折的旧角色映射；逐折核对读取结果。
    aggregate = Path(args.image_manifest).with_name("splits.json")
    config["split"]["path"] = str(aggregate)
    for fold in range(3):
        config["fold"] = fold
        rows = apply_split({"dataset": "imagechd", "cases": image["cases"]}, config)["cases"]
        by_id = {r["case_id"]: r for r in rows}
        for row in image["cases"]:
            role = row["roles"][str(fold)]
            assert by_id[row["case_id"]]["roles"] == [{"val": "validation", "aux_train": "train"}.get(role, role)]
    result = {"state": "passed", "scope": "existing split metadata only; no CT voxels or training",
        "chd1282": {"file": str(Path(args.chd_cohort).resolve()), "sha256": file_digest(args.chd_cohort),
                    "seed": frozen["seed"], "train": 598, "validation": 130, "sealed_source_test": 127,
                    "patient_roles_disjoint": True},
        "imagechd": {"file": str(Path(args.image_splits).resolve()), "sha256": file_digest(args.image_splits),
                     "train_main": 53, "train_auxiliary": 13, "validation": 10, "test": 33,
                     "all_three_legacy_folds_matched": True, "independence_unit": "case"},
        "seconds": perf_counter() - started, "peak_vram_bytes": None}
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读核验真实旧划分名单")
    for name in ("chd-manifest", "chd-cohort", "image-manifest", "image-splits", "report"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
