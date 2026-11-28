import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aaoca_reproduction.inputs import bound_manifest
from aaoca_reproduction.records import read_json, signature


parser = argparse.ArgumentParser(description="只读核对两例真实原件登记在迁移映射合同中的绑定")
parser.add_argument("--portable", required=True, type=Path)
parser.add_argument("--records", required=True, type=Path)
parser.add_argument("--raw-images", required=True, type=Path)
parser.add_argument("--raw-labels", required=True, type=Path)
args = parser.parse_args()
portable = read_json(args.portable)
sample = deepcopy(portable)
sample["cases"] = [row for row in sample["cases"] if row["dataset"] == "chd1282"][:2]
assert len(sample["cases"]) == 2
records = {}
for row in sample["cases"]:
    name = row["case_id"].removeprefix("chd1282_")
    source = read_json(args.records / (name + ".json"))
    assert source["case_id"] == name and len(source["images"]) == len(source["labels"]) == 1
    records[name] = {"image": str(args.raw_images / Path(source["images"][0]).name),
                     "label": str(args.raw_labels / Path(source["labels"][0]).name)}
paths = [Path(value).resolve(strict=True) for row in records.values() for value in row.values()]
before = [signature(path) for path in paths]
root = Path(os.path.commonpath(paths))
# 在内存中投影真实原始登记，未创建或冒充服务器source-inputs文件。
mapping = {"data_root": str(root), "cases": records}
result = bound_manifest(sample, {"chd1282_root": root}, chd1282_mapping=mapping)
legacy = bound_manifest(sample, {"chd1282_root": args.records.resolve(strict=True).parent})
for original, mapped, old in zip(sample["cases"], result["cases"], legacy["cases"], strict=True):
    assert mapped["case_id"] == old["case_id"] == original["case_id"]
    assert mapped["role"] == old["role"] == original["role"]
    assert all(mapped[key] == old[key] == original[key] for key in
        ("patient_key", "diagnosis", "patient_research_reference", "spatial_binding"))
    entry = records[mapped["source_path_binding"]["mapping_case_id"]]
    assert all(Path(mapped[key]) == Path(entry[key]).resolve(strict=True) for key in ("image", "label"))
assert before == [signature(path) for path in paths]
assert read_json(args.portable) == portable
print(json.dumps({"state": "passed", "cases": 2, "source": "actual_original_registry_in_memory_projection",
    "server_source_inputs_tested": False, "legacy_relative_binding": "passed", "roles_labels_geometry_preserved": True,
    "pixel_arrays_read": 0, "generated_files": 0, "source_unchanged": True}))
