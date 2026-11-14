import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import file_digest, read_json
from chd_segmentation.config import load_config
from chd_segmentation.datasets.manifest import load_manifest
from chd_segmentation.datasets.splits import apply_split
from chd_segmentation.datasets.transfer import bind_inputs


def run(args):
    source = Path(args.bundle).resolve(strict=True)
    original = {p.name: file_digest(p) for p in source.iterdir() if p.is_file()}
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        packet = workspace.root / "inputs"
        packet.mkdir()
        for path in source.iterdir():
            if path.suffix in (".json", ".yaml"):
                shutil.copyfile(path, packet / path.name)
        roots = {"chd1282": args.chd_root, "imagechd": args.image_root}
        first = bind_inputs(packet, workspace.root, roots, args.weights)
        stamps = {p.name: p.stat().st_mtime_ns for p in packet.iterdir() if p.is_file()}
        repeated = bind_inputs(packet, workspace.root, roots, args.weights)
        assert not repeated["changed_files"]
        assert stamps == {p.name: p.stat().st_mtime_ns for p in packet.iterdir() if p.is_file()}
        record = {"scope": "existing audit metadata and file names/sizes only; no CT voxels", "binding": first,
                  "repeat_without_write": True, "experiments": []}
        workspace.results.append(record)
        descriptor = read_json(packet / "bundle.json")
        for experiment in descriptor["experiments"]:
            config = load_config(packet / experiment["filename"])
            assert Path(config["plan"]).is_relative_to(Path(config["work_root"]) / "plans")
            manifest = apply_split(load_manifest(config["manifest"], config["dataset"], config["scope"], allow_unassigned=True), config)
            record["experiments"].append({"dataset": config["dataset"], "fold": config["fold"],
                                          "roles": dict(Counter(r["roles"][0] for r in manifest["cases"]))})
        path = packet / descriptor["experiments"][0]["filename"]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["training"]["batch"] = 8
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        bind_inputs(packet, workspace.root, roots, args.weights)
        assert load_config(path)["training"]["batch"] == 8
        assert original == {p.name: file_digest(p) for p in source.iterdir() if p.is_file()}
        record["manual_batch_preserved"] = True
        record["delivery_unchanged"] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两数据集私有迁移包元数据绑定核验，不读取CT体素")
    for name in ("bundle", "chd-root", "image-root", "weights", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
