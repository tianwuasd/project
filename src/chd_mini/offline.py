from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path
from time import perf_counter

from .contracts import confined, file_sha, read_json


FORMAT = "chd-offline-npy-v1"
TRANSFORM = "stunet-ct-nnunet-bounded-v1"


def verify_selected_statistics(prepared):
    rows = prepared["manifest"]["cases"]
    if len(rows) != 4 or any(len(r["roles"]) != 1 for r in rows):
        raise ValueError("小样本描述必须精确包含四例单独角色")
    if Counter(r["roles"][0] for r in rows) != {"train": 2, "validation": 1, "test": 1}:
        raise ValueError("小样本描述的2/1/1角色不符")
    if any(not r.get("patient") for r in rows) or len({r["patient"] for r in rows}) != 4:
        raise ValueError("四例患者身份缺失或重叠")
    training = {r["case_id"]: r for r in rows if r["roles"] == ["train"]}
    fitted = prepared["plan"]["fit_ids"]
    sources = prepared["plan"]["generation"]["request"]["training_cases"]
    if sorted(fitted) != sorted(training) or sorted(r["case_id"] for r in sources) != sorted(training):
        raise ValueError("统计来源必须精确等于所选两例训练病例")
    for row in sources:
        actual = training[row["case_id"]]
        for key in ("sha256", "spacing_xyz", "geometry_policy", "unknown_classes"):
            if row[key] != actual[key]:
                raise ValueError("统计来源的病例事实不符：" + key)
        for key in ("ignore_label_ids", "known_absent_classes"):
            if row.get(key, []) != actual.get(key, []):
                raise ValueError("统计来源的标签规则不符：" + key)
    return sorted(training)


def events(config):
    path = Path(config["run_dir"]) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def data_snapshot(prepared):
    from chd_segmentation.preprocessing.offline_store import case_directory, inspect_case
    storage = prepared["storage"]
    result = {}
    for name, pointer in storage["cases"].items():
        directory = case_directory(storage["root"], pointer["key"])
        record = inspect_case(directory, pointer["key"], storage["recipe_identity"], full=True)
        if record is None:
            raise ValueError("离线数据缺失或核验失败：" + name)
        files = [*record["files"], "case.json", "owner.json"]
        result[name] = {kind: {"sha256": file_sha(directory / kind), "bytes": (directory / kind).stat().st_size,
                               "mtime_ns": (directory / kind).stat().st_mtime_ns} for kind in files}
    return result


def produce(config, source, output):
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.preprocessing.offline_store import validate_storage
    started = perf_counter()
    prepared = preprocess(config, source, output)
    verify_selected_statistics(prepared)
    validate_storage(prepared, full=True, config=config)
    receipts = [r for r in events(config) if r["event"] == "preprocess_completed" and r["output"] == str(output)]
    if not receipts:
        raise ValueError("缺少本次真实离线处理完成事件")
    return prepared, {"wall_seconds": perf_counter() - started, **receipts[-1]}


def run_checks(config, source, output, work):
    import numpy as np
    from chd_segmentation.artifacts.store import digest
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.preprocessing.offline import data_info
    from chd_segmentation.preprocessing.offline_store import case_directory, recipe, validate_storage

    first, cold = produce(config, source, output)
    if first["storage"]["format"] != FORMAT or recipe(first, config)["transform"] != TRANSFORM:
        raise ValueError("离线数据合同或固定转换版本不符")
    if cold["new"] != 4 or cold["reused"] != 0:
        raise ValueError("本次独占数据目录应首次处理四例")
    before = data_snapshot(first)
    variant = deepcopy(config)
    variant["training"]["epochs"] = 2 if config["training"]["epochs"] == 1 else 1
    variant["resources"]["gpu_count"] = 2
    variant["logging"]["heartbeat_seconds"] += 1
    reused, warm = produce(variant, source, Path(output).with_name("reused.json"))
    if warm["new"] != 0 or warm["reused"] != 4 or reused["identity"] != first["identity"] or data_snapshot(reused) != before:
        raise ValueError("相同数值合同未完整复用原有四例数据")
    changed_fold = deepcopy(first)
    changed_fold["fold"] += 1
    changed_source = deepcopy(config)
    changed_source["preprocessing"]["statistics_source"] = "Q-A-contract-check"
    identity = first["storage"]["recipe_identity"]
    if digest(recipe(changed_fold, config)) == identity or digest(recipe(first, changed_source)) == identity:
        raise ValueError("不同折或统计来源未分离数据身份")
    info = data_info(output)
    if info["active_references"] or info["cleanup_blocked"]:
        raise RuntimeError("本次独占数据仍被使用，拒绝执行缺失修补检查")
    target_name = next(r["case_id"] for r in first["manifest"]["cases"] if r["roles"] == ["validation"])
    pointer = first["storage"]["cases"][target_name]
    target = confined(case_directory(first["storage"]["root"], pointer["key"]) / "image.npy", work)
    if file_sha(target) != before[target_name]["image.npy"]["sha256"]:
        raise ValueError("待检查数组不属于本次已核验数据")
    offset = len(events(config))
    target.unlink()
    repaired, repair = produce(config, source, Path(output).with_name("repaired.json"))
    started_cases = [r["case_id"] for r in events(config)[offset:] if r["event"] == "preprocess_case_started"]
    after = data_snapshot(repaired)
    if repair["new"] != 1 or repair["reused"] != 3 or started_cases != [target_name] or repaired["identity"] != first["identity"]:
        raise ValueError("缺失修补没有精确限定在目标病例")
    if any(after[name] != before[name] for name in before if name != target_name):
        raise ValueError("补做缺失病例时改写了其他有效病例")
    if after[target_name]["image.npy"]["sha256"] != before[target_name]["image.npy"]["sha256"]:
        raise ValueError("修补后的影像数组内容不同")
    validate_storage(first, full=True, config=config)
    dataset = CaseDataset(first, config["resources"], log_dir=config["run_dir"], logging=config["logging"])
    loaded = []
    for row in first["manifest"]["cases"]:
        started = perf_counter()
        entry = dataset.get(row["case_id"])
        if not isinstance(entry["image"], np.memmap) or entry["image"].flags.writeable or any(entry["timing"].values()):
            raise ValueError("病例加载未使用只读离线映射")
        loaded.append({"case_id": row["case_id"], "roles": row["roles"], "shape": list(entry["image"].shape),
                       "load_seconds": perf_counter() - started, "readonly_mmap": True,
                       "unknown_classes": entry["unknown_classes"]})
        dataset.clear()
        del entry
    gc.collect()
    return {"cold": cold, "reuse": warm, "repair": repair, "repair_case": target_name,
            "data": data_info(output), "loaded": loaded, "data_snapshot": after,
            "recipe_identity": identity, "fit_ids": verify_selected_statistics(first),
            "consumer_event_offset": len(events(config)),
            "reuse_overrides": {"epochs": variant["training"]["epochs"], "declared_gpu_count": 2,
                                "heartbeat_seconds": variant["logging"]["heartbeat_seconds"]},
            "contract_checks": {"different_fold_separated": True, "different_statistics_source_separated": True,
                                "date_in_numeric_recipe": False},
            "scope": "four_owned_technical_cases_with_one_intentional_missing_array"}


def verify_consumption(config, prepared_path, preprocess_result):
    from chd_segmentation.preprocessing.offline import data_info
    from chd_segmentation.preprocessing.prepare import load_prepared
    prepared = load_prepared(prepared_path)
    rows = prepared["manifest"]["cases"]
    consumed = events(config)[preprocess_result["consumer_event_offset"]:]
    forbidden = [r for r in consumed if r.get("stage") == "crop_normalize_resample" or r["event"] == "preprocess_case_started"]
    if forbidden:
        raise ValueError("训练、恢复或测试期间重新执行了整例预处理")
    batches = [r for r in consumed if r["event"] == "batch_consumed"]
    allowed = {r["case_id"] for r in rows if r["roles"] != ["test"]}
    names = {name for batch in batches for name in batch["case_ids"]}
    if names != allowed:
        raise ValueError("训练/验证实际消费病例与所选角色不符")
    info = data_info(prepared_path)
    if info["active_references"] or info["cleanup_blocked"]:
        raise RuntimeError("离线数据仍被引用，不能结束本次检查")
    return {"state": "PASS", "whole_case_resampling_after_prepare": 0, "batch_count": len(batches),
            "consumed_ids": sorted(names), "data": info, "fit_ids": verify_selected_statistics(prepared),
            "scope": "actual_cpu_or_configured_device_consumption"}
