from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np

from .classification import LABEL_MAPS
from .data import guard_inputs
from .inputs import save_frozen
from .records import CLASSIFICATION_TARGET, RELEASE, admit_stage, event, read_json, signature, write_json


GROUPS = ("author", "finetune", "finetune_robust")


def author_members(config):
    root = Path(config["data_root"]) / "author-detection"
    return [{"task": "aaoca_present", "labels": LABEL_MAPS["aaoca_present"], "author_release": RELEASE,
        "format": "author_state_dict", "strategy": "author_pretrained", "experiment_name": "official_detection_deployment",
        "member_index": i, "path": str(root / f"member_{i + 1}.pt"), "memory_map": True,
        "source": f"official classification_models.zip/anomaly_detection/model_{i + 1}/best_val_model.pt"} for i in range(5)]


def prepare_author_weights(config, check=False):
    from .preprocessing_batch import batch_lock
    archive = Path(config["author_archive"]).resolve(strict=True)
    descriptors = author_members(config)
    names = [f"classification_models/anomaly_detection/model_{i + 1}/best_val_model.pt" for i in range(5)]
    with ZipFile(archive) as source:
        infos = [source.getinfo(name) for name in names]
        estimate = sum(info.file_size for info in infos)
        if check:
            return {"check_only": True, "members": 5, "estimated_new_bytes": estimate, "source": signature(archive),
                    "paths": [item["path"] for item in descriptors]}
        root = Path(descriptors[0]["path"]).parent
        root.mkdir(parents=True, exist_ok=True)
        with batch_lock(root / ".extract.lock"):
            for info, descriptor in zip(infos, descriptors, strict=True):
                path = Path(descriptor["path"])
                receipt = path.with_suffix(".json")
                owner = {"source": signature(archive), "entry": info.filename, "CRC": info.CRC, "bytes": info.file_size}
                if path.exists():
                    if not receipt.exists() or read_json(receipt) != {**owner, "output": signature(path)}:
                        raise ValueError("已有作者成员文件缺少对应提取记录，拒绝覆盖")
                    continue
                pending = path.with_suffix(".pt.pending")
                if pending.exists():
                    raise FileExistsError("作者成员提取有未完成文件，请先核对归属")
                if shutil.disk_usage(root).free < info.file_size:
                    raise OSError("作者成员提取磁盘空间不足")
                with source.open(info) as reader, pending.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=4 * 2**20)
                if pending.stat().st_size != info.file_size:
                    raise ValueError("作者成员提取长度不符")
                pending.replace(path)
                write_json(receipt, {**owner, "output": signature(path)})
    return {"members": 5, "source": str(archive), "output_root": str(Path(descriptors[0]["path"]).parent),
            "uncompressed_weight_bytes": estimate}


def generate_experiments(config, output, device="cuda:0", check=False):
    from .classification import validate_descriptor
    output = Path(output).resolve()
    members = author_members(config)
    for item in members:
        validate_descriptor(item, "aaoca_present")
    payloads = {}
    for group in GROUPS:
        value = deepcopy(config)
        value.pop("project_root", None)
        settings = value["classification"]
        settings.update(device=device, gpu_count=1, batch_size=1, epochs=30,
            learning_rate_recipe="release_default_lr_1e-4", member_index=0, seed=20261128,
            experiment_group=group, author_members=members,
            ensemble_mode="five_initializations", checkpoint_interval_updates=10,
            run_directory=str(Path(config["course_root"]) / "experiments" / group),
            augmentation="source_robust_v1" if group == "finetune_robust" else "author")
        if group != "author":
            settings.update(strategy="author_detector_finetune", pretrained_detection=members[0],
                fine_tuning={"warmup_epochs": 5, "head_lr": 0.0001, "backbone_lr": 0.00001,
                             "freeze_batchnorm": True, "selection_metric": "average_precision", "patience": 5})
        payloads[output / f"project.{group}.json"] = json.loads(json.dumps(value, default=str))
    if not check:
        for path, value in payloads.items():
            save_frozen(path, value)
    return {"check_only": check, "files": [str(path) for path in payloads], "members_per_group": 5,
            "training_groups": ["finetune", "finetune_robust"], "server_executed": False}


def finetune_ensemble(config, manifest, resume=False):
    from .training import train
    from .preprocessing_batch import batch_lock
    settings = config["classification"]
    if settings["experiment_group"] not in {"finetune", "finetune_robust"}:
        raise ValueError("仅允许微调实验调用五成员训练")
    members = settings["author_members"]
    if len(members) != 5 or {item["member_index"] for item in members} != set(range(5)):
        raise ValueError("微调必须明确提供5个不同的作者检测成员")
    root = Path(settings["run_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress = {"state": "running", "completed_members": [], "member_states": [], "current_member": None}
    with batch_lock(root / ".ensemble.lock"):
        for member in sorted(members, key=lambda x: x["member_index"]):
            local = deepcopy(config)
            item = local["classification"]
            index = member["member_index"]
            directory = root / f"member_{index + 1}"
            item.update(member_index=index, seed=settings["seed"] + index,
                        pretrained_detection=member, run_directory=str(directory))
            has_latest = (directory / "latest.pt").exists()
            if has_latest and not resume:
                raise ValueError("已有五成员运行须显式--resume")
            progress["current_member"] = index
            write_json(root / "progress.json", progress)
            event("classification_ensemble_member_started", member_index=index, directory=str(directory), resume=has_latest)
            result = train(local, manifest, resume=has_latest)
            if not result["state"]["complete"]:
                progress.update(state="stopped", stop_reason=result.get("stop_reason", "incomplete_member"))
                write_json(root / "progress.json", progress)
                return progress
            progress["completed_members"].append(index)
            progress["member_states"].append({key: result["state"][key] for key in
                ("member_index", "successful_updates", "epoch", "complete", "test_only")})
            write_json(root / "progress.json", progress)
            gc.collect()
        progress.update(state="completed", current_member=None)
        write_json(root / "progress.json", progress)
    return progress


def _local_members(config):
    settings = config["classification"]
    root = Path(settings["run_directory"])
    members = []
    for i in range(5):
        directory = root / f"member_{i + 1}"
        record = read_json(directory / "run.json")
        if record["settings"]["member_index"] != i or record["settings"]["strategy"] != "author_detector_finetune":
            raise ValueError("微调成员记录与五成员顺序不一致")
        members.append({"task": "aaoca_present", "labels": LABEL_MAPS["aaoca_present"], "author_release": RELEASE,
            "format": "local_complete_checkpoint", "strategy": "author_detector_finetune",
            "experiment_name": settings["experiment_group"], "member_index": i,
            "source": "best_validation_average_precision_then_loss", "path": str((directory / "best.pt").resolve(strict=True)),
            "test_only": config.get("run_purpose") == "smoke", "memory_map": True})
    return members


def validation_threshold(results):
    rows = [row for row in results if row["truth"] is not None]
    labels = np.asarray([row["truth"] for row in rows])
    scores = np.asarray([row["probability"] for row in rows])
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("阈值选择须有真实两类验证病例")
    candidates = np.unique(np.r_[0.0, 0.5, scores, 1.0])
    def rank(threshold):
        predicted = scores > threshold
        sensitivity = float(predicted[labels == 1].mean())
        specificity = float((~predicted[labels == 0]).mean())
        return sensitivity + specificity - 1, -abs(float(threshold) - 0.5)
    return float(max(candidates, key=rank))


def patient_confidence_intervals(results, threshold):
    from .prediction import metrics
    rows = [row for row in results if row["truth"] is not None]
    if len(rows) < 2 or {row["truth"] for row in rows} != {0, 1}:
        raise ValueError("正式患者级置信区间需要两类参考结果")
    rng = np.random.default_rng(20261128)
    fields = ("roc_auc", "average_precision", "sensitivity", "specificity", "ppv", "f1")
    samples = {key: [] for key in fields}
    for _ in range(1000):
        chosen = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        values = metrics([row["truth"] for row in chosen], [row["probability"] for row in chosen], threshold)
        for key in fields:
            if values[key] is not None:
                samples[key].append(values[key])
    return {"method": "patient_percentile_bootstrap", "replicates": 1000,
            "intervals_95": {key: np.quantile(value, [0.025, 0.975]).tolist() if value else None
                             for key, value in samples.items()}, "conditional_on_case_control_sampling": True}


def predict_ensemble(config, manifest, split="test", device="cpu", check=False):
    from .classification import validate_descriptor
    from .prediction import metrics, predict
    local = deepcopy(config)
    settings = local["classification"]
    group = settings["experiment_group"]
    if group not in GROUPS or split not in {"validation", "test"}:
        raise ValueError("未知五成员实验或数据范围")
    descriptors = settings["author_members"] if group == "author" else _local_members(local)
    for descriptor in descriptors:
        validate_descriptor(descriptor, "aaoca_present")
    role = "candidate_validation" if split == "validation" else "predict"
    selected = [row for row in manifest["cases"] if row["role"] == role]
    root = Path(settings["run_directory"])
    directory = root / ("validation-ensemble" if split == "validation" else "test-ensemble")
    calibration = root / "validation-ensemble" / "calibration.json"
    threshold = 0.5
    if split == "test" and config.get("run_purpose") != "smoke":
        data = read_json(calibration)
        if data["group"] != group or data["selection_split"] != "validation":
            raise ValueError("融合阈值缺少本实验的验证来源")
        if data["member_files"] != [signature(item["path"]) for item in descriptors]:
            raise ValueError("融合成员在验证选阈值之后发生变化，须重新进行验证")
        threshold = data["threshold"]
    settings.update(weights=descriptors, ensemble_mode="five_initializations", device=device, gpu_count=1,
                    evaluation_role=split, prediction_case_ids=[row["case_id"] for row in selected],
                    prediction_directory=str(directory), decision_threshold=threshold)
    if check:
        return {"check_only": True, "group": group, "split": split, "members": 5, "cases": len(selected),
                "device": device, "threshold": threshold, "output": str(directory / "predictions.json")}
    result = predict(local, manifest)
    if split == "validation" and not result["test_only"]:
        threshold = validation_threshold(result["results"])
        write_json(calibration, {"group": group, "selection_split": "validation", "threshold": threshold,
            "method": "maximum_Youden_J_tie_nearest_0.5", "validation_prediction_identity": result["identity"],
            "member_files": [signature(item["path"]) for item in descriptors]})
    lookup = {row["case_id"]: row for row in manifest["cases"]}
    by_source = {}
    for dataset in ("aaoca", "chd1282", "imagechd"):
        rows = [row for row in result["results"] if lookup[row["case_id"]]["dataset"] == dataset and row["truth"] is not None]
        by_source[dataset] = metrics([row["truth"] for row in rows], [row["probability"] for row in rows], result["decision_threshold"])
    result.update(source_metrics=by_source, source_label_confounding=True,
                  cross_source_patient_identity=manifest.get("source_audit", {}).get("patient_level_cross_source_independence", "unverified"))
    if split == "test" and not result["test_only"]:
        result["confidence_intervals"] = patient_confidence_intervals(result["results"], result["decision_threshold"])
    write_json(directory / "predictions.json", result)
    return {"group": group, "split": split, "output": str(directory / "predictions.json"),
            "metrics": result["metrics"], "source_metrics": by_source, "members": 5,
            "test_only": result["test_only"], "seconds": result["seconds"], "validation_selected_threshold": threshold}


def prepared_statistics(receipt, config):
    from .data import load_array
    from .preprocessing import array_statistics
    producer = receipt["producer_provenance"] if "producer_provenance" in receipt else receipt
    if producer is not None and "input_statistics" in producer:
        return producer["input_statistics"]
    # 旧格式及无生产回执的兼容输入直接统计现成数组，不重做预处理或改写旧文件。
    admit_stage(config, "existing_preprocessed_statistics", {"array_and_quantile_work": int(np.prod(CLASSIFICATION_TARGET)) * 24})
    return array_statistics(load_array(receipt, training=False))


@guard_inputs
def shortcut_audit(config, manifest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .data import prepared_receipt
    from .labels import read_label_rows, reviewed_labels
    from .exclusions import usable_cases
    from .prediction import metrics
    from .resources import memory_guard
    reviews = reviewed_labels(read_label_rows(config["diagnoses"]), manifest, "patient_research")
    rows, features = [], []
    cases, exclusions = usable_cases(manifest["cases"], config)
    with memory_guard(config):
        for case in cases:
            receipt = prepared_receipt(case, config)
            values = prepared_statistics(receipt, config)
            rows.append(case)
            features.append([values["mean"], values["std"], *values["q05_q50_q95"], values["zero_fraction"]])
    features = np.asarray(features)
    trained = np.asarray([row["role"] == "candidate_train" for row in rows])
    tested = np.asarray([row["role"] == "predict" for row in rows])
    truth = np.asarray([reviews[row["patient_key"]]["labels"]["aaoca_present"] for row in rows], dtype=int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=20261128))
    model.fit(features[trained], truth[trained])
    probabilities = model.predict_proba(features[tested])[:, 1]
    negative = np.asarray([row["dataset"] in {"chd1282", "imagechd"} for row in rows])
    source_labels = np.asarray([int(row["dataset"] == "imagechd") for row in rows])
    source_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=20261128))
    source_model.fit(features[trained & negative], source_labels[trained & negative])
    source_probabilities = source_model.predict_proba(features[tested & negative])[:, 1]
    result = {"fit_scope": "training_patients_only", "features": "ROI_mean_std_q05_q50_q95_zero_fraction",
        "localization_exclusions": exclusions,
        "diagnosis_from_intensity_statistics": metrics(truth[tested], probabilities),
        "negative_source_from_intensity_statistics": metrics(source_labels[tested & negative], source_probabilities),
        "diagnostic_validity_proven": False, "source_bias_eliminated": False}
    path = Path(config["course_root"]) / "shortcut-audit.json"
    if path.exists():
        raise FileExistsError("来源审查结果已存在，请保留原记录")
    write_json(path, result)
    return {"output": str(path), **result}
