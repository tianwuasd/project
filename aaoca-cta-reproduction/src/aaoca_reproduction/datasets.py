from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter

from .audit import imagechd_cases, read_ct_registry
from .inputs import portable_manifest
from .labels import read_label_rows, reviewed_labels, validate_manifest
from .records import RELEASE, classification_for_task, signature


TASKS = ("aaoca_present", "origin_left")


def organize_two_tasks(config, old_manifest, workbook, output):
    started = perf_counter()
    output = Path(output).resolve()
    private = config["project_root"] / "private"
    course = Path(config["course_root"]).resolve()
    if output == private or not output.is_relative_to(private):
        raise ValueError("正式数据清单须存于本项目private下的独立数据集目录")
    if output.exists():
        raise FileExistsError("该数据集目录已存在，不覆盖正式清单")
    workbook = Path(workbook).resolve(strict=True)
    rows = read_label_rows(workbook, "data")
    registry = read_ct_registry(config)
    imagechd = imagechd_cases(config)
    reviews = reviewed_labels(rows, old_manifest, "document_explicit", registry=registry,
                               aaoca_root=config["aaoca_root"])
    root = config["aaoca_root"].resolve()
    old = {case["patient_key"]: case for case in validate_manifest(old_manifest) if case["dataset"] == "aaoca"}
    eligible = {task: {key for key, review in reviews.items()
                      if review["research_labels"][task] in (0, 1)
                      and review["ct_bindings"][task]["available_ct_matched"]} for task in TASKS}
    if not eligible["origin_left"].issubset(eligible["aaoca_present"]):
        raise ValueError("侧别病例须同时具备可对应CT的明确AAOCA阳性来源")
    randomizer = random.Random(config["seed"])
    roles = {}
    for target in (0, 1):
        patients = sorted(key for key in eligible["origin_left"]
                          if reviews[key]["research_labels"]["origin_left"] == target)
        if len(patients) < 3:
            raise ValueError("侧别每类至少需三位患者才能形成独立训练、验证和保留组")
        randomizer.shuffle(patients)
        forced = [key for key in patients if key in old and old[key]["role"] == "predict"]
        ntest = max(1, round(len(patients) * 0.15), len(forced))
        nval = max(1, round(len(patients) * 0.15))
        rest = [key for key in patients if key not in forced]
        held = forced + rest[:ntest - len(forced)]
        rest = [key for key in rest if key not in held]
        if len(rest) <= nval:
            raise ValueError("保留既有预测患者后，训练或验证类别不足")
        roles.update({key: "predict" for key in held})
        roles.update({key: "candidate_validation" for key in rest[:nval]})
        roles.update({key: "candidate_train" for key in rest[nval:]})
    for key in sorted(eligible["aaoca_present"] - eligible["origin_left"]):
        roles[key] = "predict" if key in old and old[key]["role"] == "predict" else "candidate_train"
    policy = "research_sources" if config.get("imagechd_negative_attestation") is not None else "document_explicit"
    if policy == "research_sources":
        negatives = sorted(imagechd, key=lambda case: case["case_id"])
        randomizer.shuffle(negatives)
        ntest = nval = max(1, round(len(negatives) * 0.15))
        if len(negatives) <= ntest + nval:
            raise ValueError("ImageCHD阴性患者不足独立训练、验证及保留组")
        for index, case in enumerate(negatives):
            case["role"] = "predict" if index < ntest else "candidate_validation" if index < ntest + nval else "candidate_train"
    selected, master = {}, []
    for patient in sorted(registry, key=lambda item: item["patient_key"]):
        key = patient["patient_key"]
        review = reviews.get(key)
        tasks = [task for task in TASKS if key in eligible[task]]
        relative = None
        if tasks:
            allowed = set.intersection(*(set(review["ct_bindings"][task]["matched_ct_images"]) for task in tasks))
            volumes = {Path(volume["image"]).resolve().relative_to(root).as_posix(): volume
                       for volume in patient["volumes"]}
            candidates = sorted(allowed.intersection(volumes),
                                key=lambda name: (volumes[name]["original_volume_id"], name))
            if not candidates:
                raise ValueError("两个任务没有共同的真实配对体积，须另行审核任务体积选择")
            previous = Path(old[key]["image"]).resolve().relative_to(root).as_posix() if key in old else None
            relative = previous if previous in candidates else candidates[0]
            volume = volumes[relative]
            selected[key] = {"case_id": "aaoca_" + key, "dataset": "aaoca", "patient_key": key,
                "aliases": patient["aliases"], "role": roles[key], "image": str(Path(volume["image"]).resolve()),
                "label": str(Path(volume["label"]).resolve()),
                "source_stat": {name: signature(volume["image"] if name == "image" else volume["label"])
                                for name in ("image", "label")},
                "spacing_xyz": None, "localization": "existing_totalsegmentator_2.18.0",
                "localization_pairing": {"catalog": str(config["aaoca_pairs"]),
                    "original_volume_id": volume["original_volume_id"], "catalog_status": "completed",
                    "geometry_verified_in_this_organization": False},
                "diagnosis": {task: review["research_labels"][task] for task in TASKS},
                "selected_original_volume_id": volume["original_volume_id"],
                "selection_rule": "prefer_previous_if_task_matched_else_sorted_original_volume_id",
                "excluded_sibling_volumes": [item["original_volume_id"] for item in patient["volumes"] if item != volume]}
        master.append({"patient_key": key, "dataset": "aaoca", "aliases": patient["aliases"],
                       "role": roles.get(key, "not_in_supervised_tasks"), "eligible_tasks": tasks,
                       "old_role": old[key]["role"] if key in old else None,
                       "preserve_as_holdout": key in old and old[key]["role"] == "predict",
                       "research_labels": {task: review["research_labels"][task] if review else None for task in TASKS},
                       "selected_image_relative": relative, "registered_volume_count": len(patient["volumes"]),
                       "eligibility": {task: "included" if task in tasks else "label_unknown_or_conflicting" if
                           review is None or review["research_labels"][task] is None else "ct_pairing_missing" for task in TASKS}})
    master.extend({"patient_key": case["patient_key"], "dataset": "imagechd", "aliases": case["aliases"],
                   "role": case["role"], "eligible_tasks": ["aaoca_present"] if policy == "research_sources" else [],
                   "old_role": "predict", "preserve_as_holdout": case["role"] == "predict",
                   "role_change_authority": config.get("imagechd_negative_attestation"),
                   "research_labels": {task: case["diagnosis"][task] for task in TASKS},
                   "eligibility": {task: "user_attested_negative" if task == "aaoca_present" and policy == "research_sources"
                                   else "not_applicable" if policy == "research_sources" else "diagnosis_unknown_prediction_only" for task in TASKS}}
                  for case in imagechd)
    payloads, counts, prediction_counts = {}, {}, {}
    for task in TASKS:
        cases = [deepcopy(selected[key]) for key in sorted(eligible[task])]
        manifest = {"schema": 1, "author_release": RELEASE, "seed": config["seed"],
                    "dataset_version": output.name, "task": task, "cases": cases,
                    "source": {"diagnoses": signature(workbook), "aaoca_pairs": signature(config["aaoca_pairs"])},
                    "status": "document_research_labels_with_task_matched_ct", "patient_aliases_disjoint": True}
        validate_manifest(manifest)
        checked = reviewed_labels(rows, manifest, "document_explicit", aaoca_root=root)
        if any(checked[case["patient_key"]]["labels"][task] != reviews[case["patient_key"]]["research_labels"][task]
               for case in cases):
            raise ValueError("新选体积与任务标签/配对不符")
        cases.extend(deepcopy(imagechd))
        validate_manifest(manifest)
        checked = reviewed_labels(rows, manifest, policy, aaoca_root=root)
        counts[task] = {role: dict(Counter(checked[case["patient_key"]]["labels"][task]
                                         for case in cases if case["role"] == role and case["patient_key"] in checked
                                         and checked[case["patient_key"]]["labels"][task] in (0, 1)))
                        for role in ("candidate_train", "candidate_validation", "predict")}
        prediction_counts[task] = dict(Counter(case["dataset"] for case in cases if case["role"] == "predict"))
        settings = {key: str(value) if isinstance(value, Path) else deepcopy(value)
                    for key, value in config.items() if key != "project_root"}
        settings.update(state_root=str(course / "datasets" / output.name / task),
                        input_manifest=task + "/private/manifest.json",
                        course_root=str(course / "datasets" / output.name / task),
                        data_root=str(config["data_root"] / "datasets" / output.name),
                        diagnoses=str(workbook), diagnosis_sheet="data", label_policy=policy,
                        classification_recipe="author_native_index", active_tasks=[task], smoke_cases=[],
                        dataset_version=output.name, classification=classification_for_task(config.get("classification", {}), task,
                            [case["case_id"] for case in cases if case["role"] == "predict"]))
        settings["prepared_inputs"] = {}
        for case in cases:
            prior = old.get(case["patient_key"])
            if prior and prior["image"] == case["image"] and prior["source_stat"] == case["source_stat"]:
                prepared = Path(config.get("prepared_inputs", {}).get(case["case_id"],
                                config["data_root"] / "aaoca-classification" / case["case_id"]))
                if (prepared / "classification.json").is_file():
                    settings["prepared_inputs"][case["case_id"]] = str(prepared.resolve())
        settings.update(candidate_train=sum(case["role"] == "candidate_train" for case in cases),
                        candidate_validation=sum(case["role"] == "candidate_validation" for case in cases),
                        holdout_prediction=sum(case["role"] == "predict" for case in cases))
        payloads[Path(task) / "private/manifest.json"] = manifest
        payloads[Path(task) / "private/manifest.portable.json"] = portable_manifest(manifest, config)
        payloads[Path("project." + task + ".json")] = settings
    for role in ("candidate_train", "candidate_validation", "predict"):
        if set(counts["origin_left"][role]) != {0, 1}:
            raise ValueError("侧别各组未同时具备两类")
    if any(roles.get(key) != "predict" for key in selected if key in old and old[key]["role"] == "predict"):
        raise ValueError("原预测保留患者进入新训练或验证组")
    revision = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    metadata = {"schema": 1, "dataset_version": output.name, "tasks": list(TASKS), "seed": config["seed"], "builder_code_sha": revision,
                "source_workbook": signature(workbook), "original_manifest_preserved": True,
                "counts": counts, "prediction_counts": prediction_counts, "label_policy": policy,
                "imagechd_cases": len(imagechd), "imagechd_negative_attestation": config.get("imagechd_negative_attestation"),
                "imagechd_prediction_cases": sum(case["role"] == "predict" for case in imagechd),
                "source_imagechd_manifest": signature(config["imagechd_manifest"]),
                "registered_patients": len(registry), "patients": master,
                "legacy_holdouts_preserved": sum(case["role"] == "predict" for case in old.values()),
                "eligible_legacy_holdouts": sum(key in old and old[key]["role"] == "predict" for key in selected),
                "readiness": {"aaoca_present": "supervision_partitioned_model_conditions_not_validated" if policy == "research_sources" else "blocked_missing_negative_controls",
                              "origin_left": "supervision_partitioned_model_conditions_not_validated"},
                "source_files_copied": 0, "arrays_read": 0, "training_updates": 0}
    metadata["training_ready"] = {task: False for task in TASKS}
    metadata["pending_requirements"] = {
        "aaoca_present": ([] if policy == "research_sources" else ["negative_controls"]) + ["direction_and_coverage_review", "training_parameters"],
        "origin_left": ["detection_parent_provenance", "direction_and_coverage_review", "training_parameters"]}
    payloads[Path("dataset.json")] = metadata
    texts = {name: json.dumps(value, ensure_ascii=False, indent=2) + "\n" for name, value in payloads.items()}
    texts[Path("README.md")] = ("# 前两个任务的数据集\n\n" +
        ("用户已明确确认全部ImageCHD为AAOCA阴性，并授权按患者划分训练、验证和独立预测。"
         "确认来源保存在各病例label_attestation；没有伪造医生复核字段。侧别与风险对这些阴性病例不适用。"
         if policy == "research_sources" else "全部ImageCHD纳入预测，诊断未知保持为空。") +
        "AAOCA使用明确文书及具体CT配对的部分病例参与监督。"
        "全部文件仅引用现有影像、定位标签及正式诊断表，不复制或预处理影像，不启动模型。\n\n"
        "" + json.dumps(counts, ensure_ascii=False, indent=2) + "\n\n"
        "以上为各任务适用监督分组。各任务预测候选来源数量：" + json.dumps(prediction_counts, ensure_ascii=False) + "。\n\n"
        "侧别按患者分层约70%/15%/15%，两个任务使用同一患者角色，原预测保留患者继续保留。"
        "此为本地研究划分，不声称复现作者90:10划分。数据来源与检测类别完全相关，性能须考虑来源差异。"
        "侧别只完成监督与分组；父检测权重来源、方向和覆盖及训练参数尚须单独核对。\n\n"
        "优先复用原清单中同一真实配对体积，否则按原体积标识排序选择；该规则不代替影像质控。"
        "定位标签直接引用已完成的同体积CT/标签登记，本轮未核对几何，不声称可直接预处理。"
        "所有期相按患者隔离；其余患者及未纳入原因保存在dataset.json。"
        "旧清单、旧运行及现有预处理不变。\n\n"
        "使用project.aaoca_present.json或project.origin_left.json执行check-training；"
        "其中ready只表示所选任务的监督条件，不表示已准许运行模型。"
        "配置保留已有classification公共参数并指定task及预测候选编号；跨任务不带入原策略、父权重或预测权重，旧运行输出目录另行指定。"
        "缺少权重、设备及训练参数时属于数据配置，不能直接train/predict。"
        "预处理和训练须另行取得具体产物与计算许可。\n\n"
        "任务配置通过input_manifest明确选择本批独立清单，运行目录另指向course。"
        "相对路径manifest.portable.json供将来服务器按现有数据根显式绑定；"
        "本地配置含本机路径，不声称直接可在服务器使用，不复制任何CT或Excel。"
        "这些是长期正式数据组织文件，保留至用户明确替换或删除。\n")
    size = sum(len(text.encode("utf-8")) for text in texts.values())
    if len(texts) > 10 or size > 5 * 2**20:
        raise OSError("正式清单超过10文件/5MiB范围，尚未写入")
    for name, text in texts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    return {"directory": str(output), "counts": counts, "prediction_counts": prediction_counts,
            "imagechd_cases": len(imagechd), "imagechd_prediction_cases": sum(case["role"] == "predict" for case in imagechd),
            "registered_patients": len(registry),
            "files": len(texts), "bytes": size, "seconds": perf_counter() - started,
            "training_updates": 0, "source_files_copied": 0, "arrays_read": 0}
