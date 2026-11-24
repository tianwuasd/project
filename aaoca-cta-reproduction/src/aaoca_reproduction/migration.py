from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

from .audit import check_training
from .inputs import bound_manifest, save_frozen
from .labels import imagechd_attestation, read_label_rows, reviewed_labels, validate_manifest
from .records import classification_for_task, read_json, signature


TASKS = ("aaoca_present", "origin_left")


def bind_dataset(config, directory, workbook, output, base_directory, check=False):
    started = perf_counter()
    directory = Path(directory).resolve(strict=True)
    workbook = Path(workbook).resolve(strict=True)
    output = Path(output).resolve()
    roots = [Path(config[key]).resolve() for key in ("project_root", "course_root")]
    if not any(output != root and output.is_relative_to(root) for root in roots):
        raise ValueError("绑定输出须在已有项目运行目录的独立子目录中")
    if any(output.is_relative_to(config[key]) for key in ("aaoca_root", "imagechd_root")):
        raise ValueError("绑定输出不能放原始影像目录")
    if any(Path(config["data_root"]).resolve().is_relative_to(config[key]) for key in ("aaoca_root", "imagechd_root")):
        raise ValueError("模型预处理data_root不能放原始影像目录")
    metadata = read_json(directory / "dataset.json")
    if metadata["schema"] != 1 or metadata["tasks"] != list(TASKS):
        raise ValueError("数据集须包含两个已组织任务")
    if signature(workbook)["bytes"] != metadata["source_workbook"]["bytes"]:
        raise ValueError("诊断工作簿大小与本批登记不同")
    rows = read_label_rows(workbook, "data")
    policy = metadata.get("label_policy", "document_explicit")
    payloads, results, roles = {}, {}, {}
    for task in TASKS:
        portable = read_json(directory / task / "private/manifest.portable.json")
        if portable["task"] != task or portable["dataset_version"] != metadata.get("dataset_version", directory.name):
            raise ValueError("任务或数据集版本与集中目录不符")
        manifest = bound_manifest(portable, config)
        cases = validate_manifest(manifest)
        imagechd = [case for case in cases if case["dataset"] == "imagechd"]
        if len(imagechd) != metadata["imagechd_cases"]:
            raise ValueError("ImageCHD数量与本批登记不符")
        for case in imagechd:
            if policy == "research_sources":
                if imagechd_attestation(case) is None or case["diagnosis"] != {"aaoca_present": 0, "origin_left": None, "high_risk": None}:
                    raise ValueError("ImageCHD须保留用户阴性确认及任务适用性")
            elif any(value is not None for value in case["diagnosis"].values()):
                raise ValueError("ImageCHD含未确认的诊断值")
        for case in cases:
            patient = case["dataset"], case["patient_key"]
            if patient in roles and roles[patient] != case["role"]:
                raise ValueError("两个任务存在患者跨组")
            roles[patient] = case["role"]
        reviews = reviewed_labels(rows, manifest, policy, aaoca_root=config["aaoca_root"])
        counts = {role: dict(Counter(str(reviews[case["patient_key"]]["labels"][task])
                                    for case in cases if case["role"] == role and case["patient_key"] in reviews
                                    and reviews[case["patient_key"]]["labels"][task] in (0, 1)))
                  for role in ("candidate_train", "candidate_validation", "predict")}
        if counts != metadata["counts"][task]:
            raise ValueError("绑定后的任务标签/当前CT配对与本批组织不符")
        settings = {key: str(value) if isinstance(value, Path) else deepcopy(value)
                    for key, value in config.items() if key != "project_root"}
        task_root = output / task
        settings.update(state_root=str(task_root), course_root=str(task_root),
                        input_manifest=str(task_root / "private/manifest.json"),
                        diagnoses=str(workbook), diagnosis_sheet="data", label_policy=policy,
                        classification_recipe="author_native_index", active_tasks=[task], smoke_cases=[],
                        dataset_version=portable["dataset_version"],
                        classification=classification_for_task(config.get("classification", {}), task,
                            [case["case_id"] for case in cases if case["role"] == "predict"]))
        settings["prepared_inputs"] = {case["case_id"]: str((Path(base_directory) / config["prepared_inputs"][case["case_id"]]).resolve())
                                       for case in cases if case["case_id"] in config.get("prepared_inputs", {})}
        settings.update(candidate_train=sum(case["role"] == "candidate_train" for case in cases),
                        candidate_validation=sum(case["role"] == "candidate_validation" for case in cases),
                        holdout_prediction=sum(case["role"] == "predict" for case in cases))
        results[task] = {"cases": len(cases), "imagechd_cases": len(imagechd),
                         "imagechd_prediction_cases": sum(case["role"] == "predict" for case in imagechd),
                         "supervision": check_training({**settings, "aaoca_root": config["aaoca_root"]}, manifest),
                         "classification_retained_fields": sorted(set(config.get("classification", {})) & set(settings["classification"]) - {"task", "prediction_case_ids"}),
                         "classification_removed_fields": sorted(set(config.get("classification", {})) - set(settings["classification"])),
                         "full_training_ready": False, "model_execution_validated": False}
        payloads[Path(task) / "private/manifest.json"] = manifest
        payloads[Path("project." + task + ".server.json")] = settings
    receipt = {"schema": 1, "dataset_version": metadata.get("dataset_version", directory.name),
               "source_dataset": signature(directory / "dataset.json"), "diagnoses": signature(workbook),
               "base_configuration_directory": str(Path(base_directory).resolve()),
               "dataset_roots": {key: str(config[key]) for key in ("aaoca_root", "imagechd_root", "data_root")},
               "tasks": {task: {key: result[key] for key in ("cases", "imagechd_prediction_cases", "full_training_ready",
                                                           "classification_retained_fields", "classification_removed_fields")}
                         for task, result in results.items()},
               "arrays_read": 0, "source_files_copied": 0, "training_updates": 0}
    payloads[Path("binding.json")] = receipt
    size = sum(len(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")) for value in payloads.values())
    if size > 5 * 2**20:
        raise OSError("绑定清单与配置超过5MiB，尚未写入")
    # 先核对全部已有目标，再决定是否写入，保护冻结输入和已有配置。
    for name, value in payloads.items():
        target = output / name
        if target.exists() and read_json(target) != value:
            raise FileExistsError("绑定目标已有不同内容，须明确选择独立输出目录")
    new_files = sum(not (output / name).exists() for name in payloads)
    if not check:
        for name, value in payloads.items():
            save_frozen(output / name, value)
    return {"check_only": check, "output": str(output), "tasks": results,
            "planned_files": len(payloads), "planned_bytes": size,
            "files_written": 0 if check else new_files, "seconds": perf_counter() - started,
            "arrays_read": 0, "training_updates": 0,
            "scope": "只读取路径、大小、stat及诊断表；不校验内容摘要、不处理像素、不安装或运行模型"}
