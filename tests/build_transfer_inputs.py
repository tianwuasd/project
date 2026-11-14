import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path, PureWindowsPath
import sys

import openpyxl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chd_segmentation.artifacts.store import digest, file_digest, read_json, write_json
from chd_segmentation.datasets.transfer import manifest_identity


def legacy_digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str).encode()).hexdigest()


def chd_manifest(args, cohort, policy):
    originals = read_json(args.chd_manifest)
    workbook = openpyxl.load_workbook(args.chd_excel, read_only=True, data_only=True)
    try:
        iterator = workbook["dicom2nii_info"].iter_rows(values_only=True)
        headers = list(next(iterator))
        by_source = defaultdict(list)
        for number, values in enumerate(iterator, 2):
            record = dict(zip(headers, values))
            if record.get("idx"):
                by_source[record["idx"]].append((number, record))
    finally:
        workbook.close()
    aliases = {r["case_id"]: r["excel_idx"] for r in policy.get("patient_aliases", [])}
    rows = []
    roles = {"train", "development", "source_test", "target_primary", "target_secondary"}
    for source in originals:
        name = source["case_id"]
        frozen = cohort["cases"][name]
        if frozen["role"] not in roles or not frozen.get("included", True):
            continue
        key = aliases.get(name, source["source_key"])
        linked = by_source.get(key, [])
        if len(linked) != 1:
            raise ValueError("病例缺少唯一的旧 Excel 关联：" + name)
        number, metadata = linked[0]
        spacing = [metadata[k] for k in ("PixelSpacingX", "PixelSpacingY", "calc_zSpacing")]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in spacing):
            raise ValueError("病例缺少可靠间距：" + name)
        if source["image"]["shape"] != [metadata[k] for k in ("Cols", "Rows", "valid_z")]:
            raise ValueError("Excel 与审计数组形状不同：" + name)
        expected = {PureWindowsPath(item["path"]).name: item for item in source["outputs"]}
        image_name, label_name = name + "_0000.nii.gz", name + ".nii.gz"
        files = {"image": image_name, "label": label_name}
        present = {int(k) for k, v in source["label"]["label_counts"].items() if v and int(k) > 0}
        absent = set(policy.get("known_absent_classes", {}).get(name, []))
        if not present <= set(range(1, 11)) or absent & present:
            raise ValueError("标签与旧部分监督政策不同：" + name)
        rows.append({"case_id": name, "source_key": source["source_key"], "roles": [],
            "patient": frozen["patient"], "spacing_xyz": list(map(float, spacing)),
            "geometry_policy": "shared_index_excel", "unknown_classes": sorted(set(range(1, 11)) - present - absent),
            "known_absent_classes": sorted(absent), "ignore_label_ids": [],
            "sha256": {kind: expected[filename]["sha256"] for kind, filename in files.items()},
            "source_bytes": {kind: expected[filename]["bytes"] for kind, filename in files.items()},
            "image": image_name, "label": label_name,
            "preferred_files": {kind: str(Path(source["collection"]) / ("images" if kind == "image" else "labels") / filename).replace("\\", "/")
                                for kind, filename in files.items()},
            "file_aliases": {kind: list(dict.fromkeys([filename, PureWindowsPath(source["images" if kind == "image" else "labels"][0]).name]))
                             for kind, filename in files.items()},
            "spacing_evidence": {"source": "author_excel", "sha256": cohort["inputs"]["excel"], "row": number}})
    return {"dataset": "chd1282", "scope": "formal", "data_root": ".", "binding_required": True, "cases": rows}


def image_manifest(args):
    audit = read_json(args.image_audit)
    rows = []
    for row in audit["cases"]:
        names = {kind: PureWindowsPath(row["pair"][kind]).name for kind in ("image", "label")}
        rows.append({"case_id": row["case_id"], "roles": [], "image": names["image"], "label": names["label"],
                     "sha256": row["source_hashes"], "spacing_xyz": row["spacing_xyz"],
                     "spacing_evidence": row["spacing_evidence"], "unknown_classes": row["annotation"]["unannotated_class_ids"],
                     "geometry_policy": "matching_affine_author_spacing", "ignore_label_ids": row["extra_label_ids"],
                     "preferred_files": names, "file_aliases": {k: [v] for k, v in names.items()}})
    return {"dataset": "imagechd", "scope": "formal", "data_root": ".", "binding_required": True, "cases": rows}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("迁移交付目录已存在，请保留原包并选择新目录")
    base, cohort, policy = [read_json(p) for p in (args.base_cohort, args.chd_cohort, args.chd_policy)]
    if legacy_digest(base) != cohort["parent_digest"] or legacy_digest(policy) != cohort["policy_digest"]:
        raise ValueError("旧冻结名单、派生名单与审核政策不能互相核验")
    if cohort["inputs"]["manifest"] != file_digest(args.chd_manifest) or cohort["inputs"]["excel"] != file_digest(args.chd_excel):
        raise ValueError("原始清单或 Excel 版本不匹配")
    manifests = {"chd1282": chd_manifest(args, cohort, policy), "imagechd": image_manifest(args)}
    output.mkdir(parents=True, exist_ok=False)
    immutable = {"chd1282_cohort.json": cohort, "chd1282_original_cohort.json": base,
                 "chd1282_policy.json": policy, "imagechd_splits.json": read_json(args.image_splits)}
    for name, value in immutable.items():
        write_json(output / name, value)
    descriptor = {"identity": "", "datasets": {}, "experiments": [],
                  "immutable_files": {name: file_digest(output / name) for name in immutable},
                  "weight_relative": "data/pretrained/STU-Net/small_ep4k.model"}
    source_root = Path(__file__).resolve().parents[1]
    for dataset, manifest in manifests.items():
        name = dataset + "_manifest.json"
        write_json(output / name, manifest)
        descriptor["datasets"][dataset] = {"manifest": name, "split": dataset + ("_cohort.json" if dataset == "chd1282" else "_splits.json"),
                                            "identity": manifest_identity(manifest)}
        for fold in range(1 if dataset == "chd1282" else 3):
            config = yaml.safe_load((source_root / "configs" / "experiments" / (dataset + "_stunet.yaml")).read_text(encoding="utf-8"))
            config["fold"] = fold
            config["split"] = {"path": descriptor["datasets"][dataset]["split"], "format": "auto", "id_field": "case_id", "unit": "auto"}
            run_name = f"{dataset}_stunet_fold{fold}"
            filename = run_name + ".yaml"
            config.update(manifest=name, pretrained="BIND_INPUTS_REQUIRED", work_root="BIND_INPUTS_REQUIRED", run_dir="BIND_INPUTS_REQUIRED", result_root="BIND_INPUTS_REQUIRED")
            descriptor["experiments"].append({"filename": filename, "run_name": run_name, "config": deepcopy(config)})
            (output / filename).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    descriptor["identity"] = digest({k: v for k, v in descriptor.items() if k != "identity"})
    write_json(output / "bundle.json", descriptor)
    print(json.dumps({"state": "created", "directory": str(output), "files": len(list(output.iterdir())),
                      "bytes": sum(p.stat().st_size for p in output.iterdir()),
                      "chd1282_included_roles": dict(Counter(cohort["cases"][r["case_id"]]["role"] for r in manifests["chd1282"]["cases"]))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从原审核元数据生成私有服务器迁移包，不读取病例体素")
    for name in ("base-cohort", "chd-cohort", "chd-policy", "chd-manifest", "chd-excel", "image-audit", "image-splits", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
