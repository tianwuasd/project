import argparse
import json
import os
from pathlib import Path
import sys

import nibabel as nib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aaoca_reproduction.audit import load_frozen
from aaoca_reproduction.prepare import load_localization
from aaoca_reproduction.records import load_config, measure, signature, write_json


def main():
    parser = argparse.ArgumentParser(description="原获准一例AAOCA的真实标签读取，不读取CT体素或训练")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_frozen(config)
    selected = [row for row in manifest["cases"]
                if row["dataset"] == "aaoca" and row["case_id"] in config["smoke_cases"]]
    if len(selected) != 1:
        raise ValueError("需要原获准的一例AAOCA")
    case = selected[0]
    source = Path(case["label"])
    before, links = signature(source), source.stat().st_nlink
    task = config["project_root"] / "work" / "label-read-20261121"
    if task.exists():
        raise FileExistsError("本轮独占标签检查目录已存在")
    task.mkdir(parents=True)
    completed = False
    try:
        with measure() as measurement:
            lv, aorta, structures = load_localization(case, nib.load(case["image"]), task)
            assert lv.any() and aorta.any() and lv.shape == aorta.shape
            assert not tuple(task.iterdir())
        assert signature(source) == before and source.stat().st_nlink == links
        completed = True
    finally:
        error = sys.exc_info()[1]
        # 原读取函数已释放本轮硬链接；目录非空时拒绝删除。
        task.rmdir()
        record = {"state": "passed" if completed else "failed", "case_id": case["case_id"],
                  "scope": "one real label; CT header only; zero training updates",
                  "platform": os.name, "posix_executed": os.name != "nt",
                  "temporary_links": int(os.name == "nt"), "copied_label_bytes": 0,
                  "measurement": measurement, "structures": structures if completed else None,
                  "source_stat_and_links_unchanged": completed, "training_updates": 0,
                  "vram_peak": "未测量；未使用GPU", "task_directory_absent": not task.exists(),
                  "error": str(error) if error else None}
        write_json(config["project_root"] / "reports" / "label-read-20261121.json", record)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
