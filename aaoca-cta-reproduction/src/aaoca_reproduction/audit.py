import csv
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random

from .records import RELEASE, read_json, signature, write_json


TASKS = ("aaoca_present", "origin_left", "high_risk")


def load_frozen(config):
    manifest = read_json(config["project_root"] / "private" / "manifest.json")
    if manifest["schema"] != 1 or manifest["author_release"] != RELEASE:
        raise ValueError("冻结清单格式或作者发布版本不匹配")
    return manifest


def build_manifest(config):
    with config["aaoca_pairs"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["status"] != "completed" for row in rows):
        raise ValueError("需要已完成的AAOCA CT与定位标签登记")

    @lru_cache(None)
    def patient(folder):
        info = read_json(config["aaoca_root"] / folder / "编号信息.json")
        key = str(info["registered_patient_id"])
        aliases = set(json.loads(info["source_patient_row"]["aliases"])) | {key, str(folder)}
        return key, aliases

    patients = {}
    for row in rows:
        key, aliases = patient(row["folder_id"])
        patients.setdefault(key, {"aliases": set(), "volumes": []})
        patients[key]["aliases"].update(aliases)
        patients[key]["volumes"].append(row)
    keys = sorted(patients)
    hold_count = config["holdout_prediction"]
    held = keys[:hold_count]
    remaining = keys[hold_count:]
    random.Random(config["seed"]).shuffle(remaining)
    ntrain, nval = config["candidate_train"], config["candidate_validation"]
    if len(remaining) < ntrain + nval:
        raise ValueError("AAOCA患者数不足候选分组")
    groups = {"predict": held, "candidate_train": remaining[:ntrain],
              "candidate_validation": remaining[ntrain:ntrain + nval]}
    seen_aliases = {}
    cases = []
    for role, selected in groups.items():
        for key in selected:
            person = patients[key]
            for alias in person["aliases"]:
                if alias in seen_aliases and seen_aliases[alias] != key:
                    raise ValueError("候选患者存在未解决的编号别名冲突")
                seen_aliases[alias] = key
            row = min(person["volumes"], key=lambda x: x["original_volume_id"])
            image, label = Path(row["ct"]), Path(row["label"])
            cases.append({"case_id": "aaoca_" + key, "dataset": "aaoca", "patient_key": key,
                          "aliases": sorted(person["aliases"]), "role": role,
                          "image": str(image), "label": str(label),
                          "source_stat": {"image": signature(image), "label": signature(label)},
                          "spacing_xyz": None, "localization": "existing_totalsegmentator_2.18.0",
                          "diagnosis": dict.fromkeys(TASKS),
                          "excluded_sibling_volumes": [x["original_volume_id"] for x in person["volumes"]
                                                       if x is not row]})
    imagechd = read_json(config["imagechd_manifest"])
    if imagechd["dataset"] != "imagechd":
        raise ValueError("输入必须是ImageCHD，不能使用CHD1282替代")
    for row in imagechd["cases"]:
        image = config["imagechd_root"] / row["image"]
        label = config["imagechd_root"] / row["label"]
        evidence = row["spacing_evidence"]
        if evidence["source"] != "author_excel" or evidence["array_axis_mapping"] != "J,K,M -> original x,y,z":
            raise ValueError("ImageCHD缺少已登记的作者间距与数组轴证据")
        cases.append({"case_id": "imagechd_" + row["case_id"], "dataset": "imagechd",
                      "patient_key": "imagechd:" + row["case_id"], "aliases": [], "role": "predict",
                      "image": str(image), "label": str(label),
                      "source_stat": {"image": signature(image), "label": signature(label)},
                      "spacing_xyz": row["spacing_xyz"], "spacing_evidence": evidence,
                      "localization": "existing_imagechd_anatomical_annotation",
                      "diagnosis": dict.fromkeys(TASKS), "excluded_sibling_volumes": []})
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("病例编号重复")
    return {"schema": 1, "author_release": RELEASE, "seed": config["seed"],
            "source": {"aaoca": signature(config["aaoca_pairs"]),
                       "imagechd": signature(config["imagechd_manifest"])},
            "aaoca_available_volumes": len(rows), "aaoca_available_patients": len(patients),
            "groups": dict(Counter(row["role"] for row in cases)), "cases": cases,
            "status": "candidates_with_unknown_diagnosis", "patient_aliases_disjoint": True}


def save_audit(config):
    manifest = build_manifest(config)
    root = config["project_root"]
    target = root / "private" / "manifest.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("候选清单已冻结，输入或分组变化时须明确建立新任务")
    if not target.exists():
        write_json(target, manifest)
    template = root / "private" / "diagnosis-template.csv"
    if not template.exists():
        with template.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient_key", *TASKS, "reviewer", "reference", "review_status", "ct_matched"])
            writer.writeheader()
            writer.writerows({"patient_key": row["patient_key"]} for row in manifest["cases"]
                             if row["dataset"] == "aaoca")
    summary = {k: manifest[k] for k in ("aaoca_available_volumes", "aaoca_available_patients", "groups",
                                       "patient_aliases_disjoint", "author_release", "status")}
    summary.update(imagechd_cases=sum(row["dataset"] == "imagechd" for row in manifest["cases"]),
                   aaoca_holdout_patients=config["holdout_prediction"],
                   confirmed_diagnostic_labels=dict.fromkeys(TASKS, 0), confirmed_normal_controls=0,
                   training_updates=0, full_classification_prediction=False)
    write_json(root / "reports" / "audit.json", summary)
    return manifest, summary


def check_training(config, manifest):
    path = config["diagnoses"]
    if path is None:
        return {"ready": False, "reason": "没有医生确认的病例分类标签表",
                "confirmed_diagnostic_labels": dict.fromkeys(TASKS, 0), "training_updates": 0}
    from .labels import read_label_rows, reviewed_labels

    reviews = reviewed_labels(read_label_rows(path, config.get("diagnosis_sheet")), manifest)
    patients = {row["patient_key"]: row for row in manifest["cases"] if row["dataset"] == "aaoca"}
    counts = {task: {role: Counter() for role in ("candidate_train", "candidate_validation")} for task in TASKS}
    for key, review in reviews.items():
        for task in TASKS:
            value = review["labels"][task]
            if value is None:
                continue
            role = patients[key]["role"]
            if role in counts[task]:
                counts[task][role][value] += 1
    ready = {task: all(set(counts[task][role]) == {0, 1} for role in counts[task]) for task in TASKS}
    return {"ready": all(ready.values()), "ready_by_task": ready, "counts": counts,
            "reason": "各任务的候选训练与验证需要医生确认的两个类别；候选分组还需按标签复核"}
