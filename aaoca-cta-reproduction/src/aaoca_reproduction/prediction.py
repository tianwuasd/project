from pathlib import Path
import json
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .classification import LABEL_MAPS, build_model, coverage_reviews, load_weights, prediction_direction, validate_descriptor
from .data import guard_inputs, load_array, prepared_receipt
from .exclusions import usable_cases
from .labels import read_label_rows, reviewed_labels, validate_manifest
from .preprocessing import network_tensor
from .records import event, identity, signature, write_json
from .resources import admit_model, classification_device, configure_cpu, memory_guard, smoke_shape


def metrics(truth, probabilities, threshold=0.5):
    labels = np.asarray(truth, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = scores > threshold
    tp = int(np.count_nonzero((labels == 1) & predicted))
    tn = int(np.count_nonzero((labels == 0) & ~predicted))
    fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted))
    n = len(labels)
    both = set(labels.tolist()) == {0, 1}
    return {"unit": "one_scan_per_patient", "n": n, "positive": tp + fn, "negative": tn + fp,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "accuracy": (tp + tn) / n if n else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
            "average_precision": float(average_precision_score(labels, scores)) if both else None}


@guard_inputs
def predict(config, manifest):
    settings = config["classification"]
    task = settings["task"]
    if settings.get("gpu_count", 1) != 1:
        raise ValueError("预测逐病例单卡执行，请明确gpu_count=1")
    small = smoke_shape(config)
    threshold = settings.get("decision_threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("预测阈值须为0到1之间的有限值")
    evaluation_role = settings.get("evaluation_role", "test")
    if evaluation_role not in {"test", "validation"}:
        raise ValueError("预测范围须为明确的验证或独立测试")
    if any(d.get("test_only") for d in settings["weights"]) and not small:
        raise ValueError("小测模型仅允许在明确的小测配置中预测")
    if task not in LABEL_MAPS:
        raise ValueError("未知预测任务")
    validate_manifest(manifest)
    budget = settings.get("artifact_budget_bytes")
    if budget is not None and (not isinstance(budget, int) or budget <= 0):
        raise ValueError("显式产物预算须为正整数；省略时根据所选输入数估计")
    requested = settings["prediction_case_ids"]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("预测需显式列出不重复病例，禁止自动处理全部队列")
    role = "predict" if evaluation_role == "test" else "candidate_validation"
    candidates = {r["case_id"]: r for r in manifest["cases"]
                  if r["role"] == role or (small and r["role"] == "candidate_validation")}
    rows = [candidates[key] for key in requested]
    rows, exclusions = usable_cases(rows, config)
    if not rows:
        raise ValueError("所选预测病例均因定位分割缺失或为空而被排除")
    receipts = [prepared_receipt(row, config) for row in rows]
    review = prediction_direction(config, receipts)
    coverage = coverage_reviews(config, rows, receipts)
    descriptors = settings["weights"]
    count = {"five_initializations": 5, "single_model": 1, "partial_ensemble": len(descriptors)}[settings["ensemble_mode"]]
    if len(descriptors) != count or not 1 <= count <= 5:
        raise ValueError("实际成员数与显式集成模式不匹配")
    if settings["ensemble_mode"] == "partial_ensemble" and count not in (2, 3, 4):
        raise ValueError("部分集成需要2到4个成员")
    members = [int(item["member_index"]) for item in descriptors]
    if len(set(members)) != count or any(index not in range(5) for index in members):
        raise ValueError("初始化成员编号重复或越界")
    weight_paths = [validate_descriptor(item, task)[0] for item in descriptors]
    strategies = {(item["strategy"], item.get("experiment_name")) for item in descriptors}
    if len(strategies) != 1:
        raise ValueError("集成成员必须来自同一明确实验策略")
    if len(set(weight_paths)) != count:
        raise ValueError("同一权重文件不能充当多个初始化成员")
    label_policy = config.get("label_policy", "confirmed")
    truths = reviewed_labels(read_label_rows(config["diagnoses"], config.get("diagnosis_sheet")), manifest,
                             label_policy, aaoca_root=config["aaoca_root"]) if config.get("diagnoses") else {}
    directory = Path(settings["prediction_directory"]).resolve()
    course = Path(config["course_root"]).resolve()
    if directory == course or not directory.is_relative_to(course):
        raise ValueError("预测状态与日志须放course专用目录")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("预测目录已有内容，拒绝覆盖既有结果")
    configure_cpu(config)
    admission = admit_model(config, training=False)
    budget = budget if budget is not None else max(2 * 2**20, len(rows) * count * 16384)
    device = classification_device(config, settings.get("estimated_gpu_working_bytes", admission["plan"]["estimated_gpu_working_bytes"]))
    started = perf_counter()
    run_identity = identity({"task": task, "weights": [(d, signature(p)) for d, p in zip(descriptors, weight_paths)],
                             "inputs": [p["identity"] for p in receipts], "direction": review, "coverage": coverage})
    if "decision_threshold" in settings or "evaluation_role" in settings:
        run_identity = identity({"base": run_identity, "threshold": threshold, "evaluation_role": evaluation_role})
    directory.mkdir(parents=True, exist_ok=True)
    results = [{"case_id": row["case_id"], "patient_key": row["patient_key"], "task": task,
                "preprocessing_id": receipt["identity"], "members": [], "ensemble_mode": settings["ensemble_mode"]}
               for row, receipt in zip(rows, receipts)]
    with memory_guard(config) as memory:
        local_cohorts = set()
        local_seeds = set()
        for descriptor in descriptors:
            model = build_model(task)
            provenance = load_weights(model, descriptor, task)
            if "training_patient_keys" in provenance:
                forbidden = set(provenance["training_patient_keys"])
                if evaluation_role == "test" and not small:
                    forbidden.update(provenance["validation_patient_keys"])
                if forbidden.intersection(row["patient_key"] for row in rows):
                    raise ValueError("预测病例与该成员实际训练或模型选择患者重叠")
            if "cohort_identity" in provenance:
                local_cohorts.add(provenance["cohort_identity"])
                if provenance["seed"] in local_seeds or len(local_cohorts) > 1:
                    raise ValueError("本地初始化成员种子重复或训练/验证患者划分不一致")
                local_seeds.add(provenance["seed"])
            model = model.to(device).eval()
            with torch.no_grad():
                for index, receipt in enumerate(receipts):
                    array = load_array(receipt, training=False)
                    if small:
                        start = [(n - s) // 2 for n, s in zip(array.shape, small)]
                        array = array[tuple(slice(a, a + s) for a, s in zip(start, small))]
                        tensor = torch.tensor(array.copy(), dtype=torch.float32).permute(2, 0, 1)[None, None].to(device)
                    else:
                        tensor = network_tensor(array).to(device)
                    logit = model(tensor).reshape(-1)
                    if logit.numel() != 1 or not torch.isfinite(logit).all():
                        raise ValueError("实际模型未产生有限单logit")
                    probability = float(torch.sigmoid(logit).item())
                    results[index]["members"].append({"member_index": descriptor["member_index"],
                                                       "probability": probability, "provenance": provenance})
                    event("classification_prediction_member", case_id=rows[index]["case_id"],
                          task=task, member_index=descriptor["member_index"], probability=probability)
            del model
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        valid_truth, valid_probability = [], []
        confirmed_truth, confirmed_probability = [], []
        for row, result in zip(rows, results):
            probability = sum(x["probability"] for x in result["members"]) / count
            label = int(probability > threshold)
            reference = truths.get(row["patient_key"], {})
            labels = reference.get("labels", {})
            truth = labels.get(task)
            if task != "aaoca_present" and not reference.get("reference_positive_aaoca", False):
                truth = None
            result.update(probability=probability, predicted_label=label, truth=truth,
                          applicability="screening_detection" if task == "aaoca_present" else "conditional_on_reference_positive_AAOCA",
                          reference_provenance={key: reference.get(key) for key in
                              ("label_policy", "source_type", "reference", "certainty", "review_status", "ct_matched", "ct_match_status",
                               "selected_ct_image", "matched_ct_images", "ct_match_reference", "ct_binding_state")},
                          task_ct_binding=reference.get("ct_bindings", {}).get(task),
                          state="completed")
            if truth is not None:
                valid_truth.append(truth)
                valid_probability.append(probability)
            confirmed = reference.get("confirmed_labels", {})
            confirmed_value = confirmed.get(task)
            if task != "aaoca_present" and not reference.get("confirmed_positive_aaoca", False):
                confirmed_value = None
            if confirmed_value is not None:
                confirmed_truth.append(confirmed_value)
                confirmed_probability.append(probability)
        report = {"identity": run_identity, "task": task, "results": results, "metrics": metrics(valid_truth, valid_probability, threshold),
                  "decision_threshold": threshold, "evaluation_role": evaluation_role,
                  "test_only": bool(small) or config.get("run_purpose") == "smoke",
                  "localization_exclusions": exclusions,
                  "unknown_or_inapplicable": len(results) - len(valid_truth), "memory": memory,
                  "seconds": perf_counter() - started, "initialization_members": count,
                  "full_five_initializations": count == 5, "direction_review": review, "coverage_reviews": coverage,
                  "direction_status": review["direction_status"],
                  "evaluation_scope": "exploratory_predictions_with_" + label_policy + "_reference_only",
                  "label_policy": label_policy,
                  "confirmed_reference_metrics": metrics(confirmed_truth, confirmed_probability, threshold),
                  "paper_performance_validated": False, "clinical_diagnostic_validity": "not_established"}
        if len(json.dumps(report, ensure_ascii=False).encode("utf-8")) * 2 > budget:
            raise OSError("预测结果超过获准保存预算")
        write_json(directory / "predictions.json", report)
    return report
