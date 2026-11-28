from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import date, datetime
import json
from pathlib import Path, PurePosixPath
import random

from .audit import load_frozen, read_ct_registry
from .inputs import save_frozen
from .records import RELEASE, read_json, signature, write_json


ROLE_NAMES = ("candidate_train", "candidate_validation", "predict")


def demographic_audit(chd_excel, image_excel):
    from openpyxl import load_workbook

    def normalized_date(value):
        if value is None or str(value).strip().lower() in {"", "na", "none", "unknown"}:
            return None
        if isinstance(value, (date, datetime)):
            return value.strftime("%Y%m%d")
        text = str(int(value)) if isinstance(value, (int, float)) and float(value).is_integer() else str(value).strip()
        text = text.replace("-", "").replace("/", "")
        return datetime.strptime(text, "%Y%m%d").strftime("%Y%m%d")

    def keys(path, paired_rows=False):
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            values = iter(workbook.active.values)
            if paired_rows:
                rows = []
                for row in values:
                    if not any(x is not None for x in row):
                        continue
                    if [str(row[i]).strip() for i in (2, 4, 6)] != ["PatientBirthDate", "AcquisitionDate", "PatientSex"]:
                        raise ValueError("ImageCHD作者表逐行字段布局不符")
                    rows.append(dict(zip(("idx", "PatientBirthDate", "AcquisitionDate", "PatientSex"), (row[1], row[3], row[5], row[7]))))
            else:
                headers = next(values)
                rows = [dict(zip(headers, row)) for row in values if any(x is not None for x in row)]
        finally:
            workbook.close()
        result, missing = defaultdict(list), 0
        for row in rows:
            born, acquired = normalized_date(row["PatientBirthDate"]), normalized_date(row["AcquisitionDate"])
            sex = str(row["PatientSex"]).strip().upper()
            if born is None or acquired is None or sex not in {"M", "F"}:
                missing += 1
                continue
            result[(born, acquired, sex)].append(row["idx"])
        return result, len(rows), missing

    chd, n_chd, missing_chd = keys(chd_excel)
    image, n_image, missing_image = keys(image_excel, paired_rows=True)
    return {"method": "normalized_birth_date_acquisition_date_sex_metadata_candidate_match",
        "cross_source_candidate_keys": len(chd.keys() & image.keys()), "CHD_rows": n_chd, "ImageCHD_rows": n_image,
        "CHD_nonunique_keys": sum(len(n) > 1 for n in chd.values()), "ImageCHD_nonunique_keys": sum(len(n) > 1 for n in image.values()),
        "CHD_missing_keys": missing_chd, "ImageCHD_missing_keys": missing_image,
        "patient_independence_proven": False,
        "image_nonunique_row_groups": [sorted(int(x) for x in group) for group in image.values() if len(group) > 1],
        "candidate_pairs": [{"chd_source_key": str(source), "image_author_row": int(image_row)}
                            for key in sorted(chd.keys() & image.keys()) for source in chd[key] for image_row in image[key]],
        "source_files": [signature(chd_excel), signature(image_excel)]}


def source_audit(config):
    chd = read_json(config["chd1282_manifest"])
    image = read_json(config["imagechd_manifest"])
    if chd["dataset"] != "chd1282" or image["dataset"] != "imagechd":
        raise ValueError("三来源核查须使用各自实际数据集的登记")
    hashes = defaultdict(list)
    for dataset, rows in (("chd1282", chd["cases"]), ("imagechd", image["cases"])):
        for row in rows:
            hashes[row["sha256"]["image"]].append((dataset, row["case_id"]))
    duplicates = [group for group in hashes.values() if len({x[0] for x in group}) > 1]
    groups = Counter(str(row["patient"]) for row in chd["cases"])
    policy = config.get("chd1282_spatial_policy")
    report = {"schema": "aaoca-source-audit-v1", "chd_scans": len(chd["cases"]),
            "chd_registered_patients": len(groups), "chd_multiple_scan_patients": sum(v > 1 for v in groups.values()),
            "imagechd_registered_cases": len(image["cases"]), "exact_cross_source_file_groups": duplicates,
            "exact_cross_source_file_group_count": len(duplicates),
            "patient_level_cross_source_independence": "unverified_without_shared_patient_or_scan_mapping",
            "duplicate_check_scope": "existing_image_SHA256_and_within_CHD_registered_patient_IDs",
            "chd_spatial_policy": policy, "chd_direction_scope": "user_accepted_eight_case_anatomical_review_not_full_DICOM_validation",
            "chd_absolute_origin_required": False, "chd_relative_array_alignment_verified": False,
            "chd_intensity_calibration_verified": False,
            "source_manifest_stats": {"chd1282": signature(config["chd1282_manifest"]),
                                      "imagechd": signature(config["imagechd_manifest"])},
            "arrays_read": 0, "model_calls": 0}
    if config.get("demographic_excel"):
        report["demographic_audit"] = demographic_audit(config["demographic_excel"]["chd1282"], config["demographic_excel"]["imagechd"])
        chd_by_source = {row["source_key"]: row for row in chd["cases"]}
        image_by_row = {int(row["spacing_evidence"]["row"]): row for row in image["cases"]}
        report["demographic_candidate_links"] = [
            {"chd_case_id": chd_by_source[pair["chd_source_key"]]["case_id"],
             "chd_patient": str(chd_by_source[pair["chd_source_key"]]["patient"]),
             "image_case_id": image_by_row[pair["image_author_row"]]["case_id"]}
            for pair in report["demographic_audit"]["candidate_pairs"]
            if pair["chd_source_key"] in chd_by_source and pair["image_author_row"] in image_by_row]
        report["demographic_candidate_patient_count"] = len({row["chd_patient"] for row in report["demographic_candidate_links"]})
        report["imagechd_possible_repeat_groups"] = [
            [image_by_row[number]["case_id"] for number in group if number in image_by_row]
            for group in report["demographic_audit"]["image_nonunique_row_groups"]]
    return report


def _resolve_source(root, row, kind):
    names = [row["preferred_files"][kind], *row.get("file_aliases", {}).get(kind, [])]
    paths = [Path(root) / name for name in dict.fromkeys(names)]
    present = [path.resolve() for path in paths if path.is_file()]
    if not present:
        raise FileNotFoundError("当前数据根找不到已登记的" + kind + "：" + row["case_id"])
    return present[0]


def _portable(case, config):
    value = deepcopy(case)
    root = Path(config[case["dataset"] + "_root"]).resolve()
    value["expected_bytes"] = {name: case["source_stat"][name]["bytes"] if case["source_stat"][name] else None
                               for name in ("image", "label")}
    for name in ("image", "label"):
        value[name] = Path(case[name]).resolve().relative_to(root).as_posix()
    value.pop("source_stat")
    return value


def build_cohort(config, diagnoses, output, check=False):
    diagnoses = Path(diagnoses).resolve(strict=True)
    old = load_frozen(config)
    registry = read_ct_registry(config)
    aliases = {}
    for patient in registry:
        for alias in patient["aliases"]:
            if alias in aliases and aliases[alias] != patient["patient_key"]:
                raise ValueError("AAOCA登记别名冲突")
            aliases[alias] = patient["patient_key"]
    with diagnoses.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    accepted, excluded = {}, Counter()
    for row in rows:
        matched = {aliases[x] for x in (row["Admission Num"].strip(), row["Inpatient Num"].strip()) if x in aliases}
        if len(matched) > 1:
            raise ValueError("新CSV两项登记号指向不同患者")
        reason = ("unmatched_patient" if not matched else "uncertain_diagnosis" if row["aaoca_status"] != "confirmed" else
                  "comorbidity_exclusion" if row["comorbidity_exclusion"] != "not_flagged" else
                  "CT_document_status" if row["CT_status"] != "Performed" else None)
        if reason:
            excluded[reason] += 1
            continue
        key = next(iter(matched))
        if key in accepted and accepted[key]["aaoca_subtype"] != row["aaoca_subtype"]:
            raise ValueError("新CSV同一患者存在冲突侧别")
        accepted.setdefault(key, row)
    previous = {case["patient_key"]: case for case in old["cases"]}
    cases = []
    for patient in registry:
        key = patient["patient_key"]
        if key not in accepted:
            continue
        old_case = previous.get(key, {})
        volume = min(patient["volumes"], key=lambda v: (v["image"] != old_case.get("image"), v["original_volume_id"]))
        image, label = Path(volume["image"]).resolve(strict=True), Path(volume["label"]).resolve()
        row = accepted[key]
        cases.append({"case_id": "aaoca_" + key, "patient_key": key, "aliases": patient["aliases"], "dataset": "aaoca",
            "image": str(image), "label": str(label), "spacing_xyz": None, "localization": "existing_totalsegmentator_2.18.0",
            "source_stat": {"image": signature(image), "label": signature(label) if label.exists() else None},
            "origin_left": {"R-AAOCA": 0, "L-AAOCA": 1}.get(row["aaoca_subtype"]),
            "legacy_holdout": old_case.get("role") == "predict", "prior_role": old_case.get("role"),
            "patient_research_reference": {"patient_key": key, "source_type": "curated_patient_csv",
                "source_row_id": row["patient_id"], "reference": diagnoses.name + ":" + row["patient_id"]},
            "diagnosis": {"aaoca_present": 1, "origin_left": None, "high_risk": None}})
    audit = source_audit(config)
    raw_id_overlap = set(aliases) & {str(row["patient"]) for row in read_json(config["chd1282_manifest"])["cases"]}
    audit["aaoca_chd_raw_identifier_candidate_overlap"] = len(raw_id_overlap)
    if raw_id_overlap:
        raise ValueError("AAOCA与CHD登记号有潜在同人冲突，须先核实身份及标签")
    if not config.get("chd1282_spatial_policy"):
        raise ValueError("请先绑定CHD1282已有方向与同索引研究政策")
    chd_manifest = read_json(config["chd1282_manifest"])
    image_manifest = read_json(config["imagechd_manifest"])
    duplicate_chd = {case_id for group in audit["exact_cross_source_file_groups"] for dataset, case_id in group if dataset == "chd1282"}
    possible_shared_patients = {row["chd_patient"] for row in audit.get("demographic_candidate_links", [])}
    possible_shared_patients.update(str(row["patient"]) for row in chd_manifest["cases"] if row["case_id"] in duplicate_chd)
    selected = {}
    for row in sorted(chd_manifest["cases"], key=lambda r: (len(r["unknown_classes"]), r["case_id"])):
        missing_localization = (set(row["unknown_classes"]) | set(row["known_absent_classes"])) & {1, 5}
        if not missing_localization and row["case_id"] not in duplicate_chd and str(row["patient"]) not in possible_shared_patients:
            selected.setdefault(str(row["patient"]), row)
    audit["conservative_cross_source_excluded_chd_patients"] = len(possible_shared_patients)
    audit["chd_patients_without_usable_LV_AO_annotation"] = len({str(row["patient"]) for row in chd_manifest["cases"]}
        - possible_shared_patients - selected.keys())
    possible_image_repeats = {key for group in audit.get("imagechd_possible_repeat_groups", []) for key in sorted(group)[1:]}
    image_sources = [row for row in image_manifest["cases"] if row["case_id"] not in possible_image_repeats]
    audit["conservative_within_imagechd_excluded_cases"] = len(possible_image_repeats)
    for dataset, sources in (("chd1282", list(selected.values())), ("imagechd", image_sources)):
        root = config[dataset + "_root"]
        for row in sources:
            image, label = (_resolve_source(root, row, kind) for kind in ("image", "label"))
            patient = dataset + ":" + (str(row["patient"]) if dataset == "chd1282" else row["case_id"])
            case_id = dataset + "_" + row["case_id"]
            ref = {"source_type": "user_attestation", "confirmed_by": "user", "confirmed_on": "2026-11-28",
                "patient_key": patient, "aaoca_present": 0,
                "reference": "User explicitly confirmed this source cohort AAOCA-negative and authorized patient-level supervised splitting"}
            case = {"case_id": case_id, "patient_key": patient, "aliases": [], "dataset": dataset,
                "image": str(image), "label": str(label), "spacing_xyz": row["spacing_xyz"],
                "spacing_evidence": row["spacing_evidence"], "source_image_sha256": row["sha256"]["image"],
                "source_stat": {"image": signature(image), "label": signature(label)},
                "localization": "existing_11_class_anatomical_annotation", "patient_research_reference": ref,
                "legacy_holdout": previous.get(patient, {}).get("role") == "predict",
                "prior_role": previous.get(patient, {}).get("role"),
                "diagnosis": {"aaoca_present": 0, "origin_left": None, "high_risk": None}}
            if dataset == "chd1282":
                case["spatial_binding"] = deepcopy(config["chd1282_spatial_policy"])
                case["source_registered_patient"] = str(row["patient"])
            cases.append(case)
    rng = random.Random(20261128)
    by_source = {dataset: [row for row in cases if row["dataset"] == dataset] for dataset in ("aaoca", "chd1282", "imagechd")}
    for dataset, source in by_source.items():
        free = sorted((row for row in source if not row["legacy_holdout"]), key=lambda r: r["case_id"])
        rng.shuffle(free)
        if dataset == "aaoca":
            ordered = []
            for side, count in ((0, 8), (1, 2)):
                ordered.extend([row for row in free if row["origin_left"] == side][:count])
            if len(ordered) != 10:
                raise ValueError("明确侧别阳性不足以选择8右2左的微调训练组")
            free = ordered + [row for row in free if row not in ordered]
            amounts = (10, 20)
        else:
            amounts = (83, 166) if dataset == "chd1282" else (7, 14)
        if len(free) < sum(amounts) or len(source) <= sum(amounts):
            raise ValueError("真实可用患者不足已批准的微调/验证/测试分组：" + dataset)
        for row in source:
            row["role"] = "predict"
        for row in free[:amounts[0]]:
            row["role"] = "candidate_train"
        for row in free[amounts[0]:sum(amounts)]:
            row["role"] = "candidate_validation"
    counts = {role: dict(Counter(row["dataset"] for row in cases if row["role"] == role)) for role in ROLE_NAMES}
    portable = {"schema": 1, "portable_schema": 1, "author_release": RELEASE, "seed": 20261128,
        "diagnoses_file": diagnoses.name,
        "label_policy": "patient_research", "evaluation_scope": "exploratory_source_confounded_case_control",
        "source_audit": audit, "counts": counts, "source_csv_bytes": diagnoses.stat().st_size,
        "source_label_rows": len(rows), "positive_exclusions": dict(excluded),
        "cases": [_portable(case, config) for case in sorted(cases, key=lambda c: c["case_id"])]}
    summary = {"check_only": check, "cases": len(cases), "counts": counts,
        "positive": len(accepted), "positive_fraction": len(accepted) / len(cases), "source_audit": audit}
    if not check:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        save_frozen(output / "cohort.portable.json", portable)
        target = output / diagnoses.name
        if target.exists() and target.read_bytes() != diagnoses.read_bytes():
            raise FileExistsError("集中诊断CSV已有不同内容")
        if not target.exists():
            target.write_bytes(diagnoses.read_bytes())
        save_frozen(output / "source-audit.json", audit)
        for name in ("chd1282_manifest", "imagechd_manifest"):
            save_frozen(output / (name + ".json"), read_json(config[name]))
    return summary


def bind_cohort(config, inputs, output, check=False):
    from .inputs import bound_manifest
    inputs, output = Path(inputs).resolve(strict=True), Path(output).resolve()
    portable = read_json(inputs / "cohort.portable.json")
    name = portable["diagnoses_file"]
    if not isinstance(name, str) or PurePosixPath(name).name != name or "\\" in name or ":" in name or not name.lower().endswith(".csv"):
        raise ValueError("诊断文件名须为输入包内的单个CSV文件名")
    diagnoses = (inputs / name).resolve(strict=True)
    if diagnoses.stat().st_size != portable["source_csv_bytes"]:
        raise ValueError("集中CSV大小与清单登记不一致")
    config = deepcopy(config)
    mapping = None
    if config.get("chd1282_source_inputs"):
        path = Path(config["chd1282_source_inputs"]).resolve(strict=True)
        mapping = read_json(path)
        root = Path(mapping["data_root"]).resolve(strict=True)
        if config.get("chd1282_root") and Path(config["chd1282_root"]).resolve(strict=True) != root:
            raise ValueError("显式CHD根与原件映射根冲突")
        config["chd1282_root"] = root
    manifest = bound_manifest(portable, config, chd1282_mapping=mapping)
    if mapping is not None:
        manifest["source_bindings"] = {"chd1282": {"mapping_file": signature(path),
            "data_root": str(root), "mapped_cases": sum(row["dataset"] == "chd1282" for row in manifest["cases"])}}
    settings = deepcopy(config)
    settings.pop("demographic_excel", None)
    settings.update(project_root=output, state_root=str(output), course_root=output,
        diagnoses=str(diagnoses), diagnosis_sheet=None,
        label_policy="patient_research", active_tasks=["aaoca_present"], training_policy="local_adaptation",
        classification_recipe="canonical_ras_v1", prepared_input_policy="compatible", offset_ras_mm=[10, 0, 10],
        input_manifest=str(output / "manifest.json"))
    settings.update(chd1282_manifest=str(inputs / "chd1282_manifest.json"),
                    imagechd_manifest=str(inputs / "imagechd_manifest.json"),
                    chd1282_spatial_policy=manifest["source_audit"]["chd_spatial_policy"])
    settings["classification"] = {"task": "aaoca_present", "prediction_case_ids": [row["case_id"] for row in manifest["cases"] if row["role"] == "predict"]}
    if not check:
        save_frozen(output / "manifest.json", manifest)
        save_frozen(output / "project.dataset.json", json.loads(json.dumps(settings, default=str)))
    return {"check_only": check, "cases": len(manifest["cases"]), "config": str(output / "project.dataset.json"),
            "counts": portable["counts"], "pixel_arrays_read": 0}
