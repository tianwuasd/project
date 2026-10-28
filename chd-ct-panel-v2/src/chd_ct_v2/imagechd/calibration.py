"""Fit postprocessing on each fold's validation patients, never its test patients."""

from pathlib import Path

from ..recovery import file_hash, read_json, write_json
from .common import load_case
from .config import STAGES
from .gpu_memory import gpu_budget
from .postprocessing import baseline_policy, choose_policy, keep_largest, selection_scores


def calibration_contract(directory, rows, config, output, mode):
    return {
        "mode": mode,
        "prepared_sha256": file_hash(Path(directory) / "dataset.json"),
        "weights": {s: file_hash(Path(output) / (s + ".pt")) for s in STAGES},
        "roi_sha256": file_hash(Path(output) / "roi.json"),
        "fold": config.get("fold"),
        "validation": {r["case_id"]: file_hash(Path(directory) / r["cache"]) for r in rows["val"]},
        "validation_patients": sorted({r["patient_id"] for r in rows["val"]}),
        "postprocessing": config.get("postprocessing", {"max_growth_steps": 2}),
    }


def calibrate(directory, rows, config, output, mode, resume=False):
    from .predict import predict_volume

    output = Path(output)
    path = output / "postprocessing.json"
    contract = calibration_contract(directory, rows, config, output, mode)
    fold = config.get("fold")
    if fold and (
        set(contract["validation"]) != set(fold["cases"]["val"])
        or set(contract["validation_patients"]) != set(fold["patients"]["val"])
    ):
        raise ValueError("后处理选择必须严格使用本折验证病例")
    if {r["patient_id"] for r in rows["train"]} & set(contract["validation_patients"]):
        raise ValueError("后处理验证患者与训练重叠")
    collection = read_json(output / "models.json")
    ref = collection.get("postprocessing")
    if resume and path.is_file() and ref:
        if ref.get("file") != path.name or file_hash(path) != ref.get("sha256"):
            raise ValueError("已完成后处理记录改变")
        saved = read_json(path)
        if saved.get("contract") != contract:
            raise ValueError("后处理验证来源已改变，拒绝复用")
        return {"file": path.name, "sha256": file_hash(path), "calibrated": saved["policy"]["calibrated"]}
    steps = contract["postprocessing"]["max_growth_steps"]
    records = []
    policy = baseline_policy(steps, "smoke_or_preflight_not_formal_validation")
    if mode == "train":
        with gpu_budget(config["device"], output / "calibration.gpu-memory.json", config) as guard:
            for row in rows["val"]:
                guard.check()
                image, _ = load_case(Path(directory), row, labels=False)
                _, details = predict_volume(
                    image,
                    output,
                    collection,
                    config["device"],
                    return_stages=True,
                    policy_override=baseline_policy(steps),
                )
                arrays = details.pop("stage_outputs")
                _, target = load_case(Path(directory), row)
                record = {
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "roi": details,
                    "scores": {},
                    "missing": {},
                }
                for name in ("vote", "limited"):
                    for label in range(8):
                        candidate = name if label == 0 else f"{name}_lcc_{label}"
                        prediction = arrays[name] if label == 0 else keep_largest(arrays[name], label)
                        score, missing = selection_scores(prediction, target)
                        record["scores"][candidate] = score.tolist()
                        record["missing"][candidate] = missing.tolist()
                records.append(record)
                print(
                    f"[calibration] validation {len(records)}/{len(rows['val'])}: {row['case_id']}",
                    flush=True,
                )
        policy = choose_policy(records, steps)
    write_json(
        path,
        {
            "contract": contract,
            "policy": policy,
            "cases": records,
            "test_cases_used": 0,
            "validation_reused_for_checkpoint_selection": True,
            "warning": "内部验证选择，不是独立测试成绩；需外层测试确认泛化",
        },
    )
    return {"file": path.name, "sha256": file_hash(path), "calibrated": policy["calibrated"]}


def load_policy(directory, collection):
    members = collection.get("_members")
    if members:
        if len(members) == 1:
            return load_policy(members[0]["directory"], members[0]["collection"])
        return baseline_policy(
            collection["config"].get("postprocessing", {}).get("max_growth_steps", 2),
            "external_ensemble_has_no_independent_calibration",
        )
    ref = collection.get("postprocessing", {})
    path = Path(directory) / "postprocessing.json"
    if ref.get("file") != path.name or not path.is_file() or file_hash(path) != ref.get("sha256"):
        raise ValueError("后处理验证记录缺失或已改变")
    saved = read_json(path)
    contract = saved.get("contract", {})
    expected = {s: collection["stages"][s]["sha256"] for s in STAGES}
    if (
        contract.get("weights") != expected
        or contract.get("fold") != collection["config"].get("fold")
        or contract.get("prepared_sha256") != collection["prepared_sha256"]
        or contract.get("roi_sha256") != collection.get("roi", {}).get("sha256")
        or contract.get("mode") != collection["mode"]
        or contract.get("postprocessing")
        != collection["config"].get("postprocessing", {"max_growth_steps": 2})
    ):
        raise ValueError("后处理记录与当前模型来源不一致")
    fold = collection["config"].get("fold")
    if fold and (
        set(contract.get("validation", {})) != set(fold["cases"]["val"])
        or set(contract.get("validation_patients", [])) != set(fold["patients"]["val"])
    ):
        raise ValueError("后处理使用了错误的验证集合")
    return saved["policy"]
