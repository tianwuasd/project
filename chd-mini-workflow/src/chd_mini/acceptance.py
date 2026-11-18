from .attempts import STAGES


def summarize_run(invocations, ledger, versions, elapsed):
    counts = {}
    for stage in STAGES:
        rows = [r for r in ledger["attempts"] if r["stage"] == stage]
        events = [e for r in invocations if r.get('stage') == stage
                  for e in r.get('receipt', {}).get('optimizer_update_events', [])]
        counts[stage] = {"attempts": len(rows),
                         "ledger_successful": sum(r["outcome"] == "successful" for r in rows),
                         "successful_updates": None, "optimizer_records": events,
                         "update_count_note": "记录位置及台账状态不等于总真实优化器更新次数",
                         "outcomes": [r["outcome"] for r in rows]}
    completed = bool(invocations) and all(r["state"] == "completed" for r in invocations)
    coverage = {}
    for stage in STAGES:
        receipts = [r.get("receipt", {}) for r in invocations if r.get("stage") == stage]
        coverage[stage] = {"successful_update": bool(counts[stage]["successful_updates"]),
                           "restored": any(r.get("restored") is True for r in receipts),
                           "best": any(bool(r.get("best_path")) for r in receipts)}
    required = all(all(items.values()) for items in coverage.values())
    predictions = [r.get("receipt", {}) for r in invocations if r["kind"] == "predict"]
    prediction_coverage = {model: any(r.get("model") == model and r.get("checkpoint") == "best"
                                    and r.get("resumed_cases_skipped", 0) >= 1
                                    and r.get("source_geometry_restored") is True for r in predictions)
                           for model in ("stu", "q", "med")}
    passed = completed and required and all(prediction_coverage.values())
    return {"state": "TECHNICAL_COMPLETED" if passed else "INCOMPLETE",
            "versions": versions, "stages": invocations, "training": counts,
            "training_coverage": coverage, "prediction_coverage": prediction_coverage,
            "total_attempts": len(ledger["attempts"]), "wall_seconds": elapsed,
            "independent_train_validation_test": "未验证", "formal_q_groups": "未验证",
            "cuda_nccl": "未验证", "full128_capacity": "未验证",
            "measurement_note": "生产记录提供资源及计算计时；未提供的字段为未测量，嵌套计时不能相加"}
