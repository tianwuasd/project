"""Deterministic patient-grouped folds and leakage-aware prediction selection."""

import random

CV_MODEL_FORMAT = "imagechd7-threefold-v1"
SPLITS = ("train", "val", "test")


def make_plan(data, seed, scope="auto"):
    if data.get("for_prediction"):
        raise ValueError("无标签预测缓存不能用于三折训练")
    if scope not in {"auto", "all", "train_val"}:
        raise ValueError("development_scope 必须为 auto/all/train_val")
    policy = data.get("id_policy", "filename_as_patient")
    if scope == "auto":
        scope = "train_val" if policy == "explicit_patient_split" else "all"
    cohort, patients, seen = [], {}, set()
    for row in sorted(data["cases"], key=lambda r: r["case_id"]):
        case, patient, part = row.get("case_id"), row.get("patient_id"), row.get("split")
        if not case or case in seen or not patient or part not in SPLITS:
            raise ValueError("病例/患者编号或原始划分无效")
        seen.add(case)
        if patients.setdefault(patient, part) != part:
            raise ValueError("同一患者跨原始数据划分")
        cohort.append({"case_id": case, "patient_id": patient, "split": part})
    development = [r for r in cohort if scope == "all" or r["split"] != "test"]
    order = sorted({r["patient_id"] for r in development})
    if len(order) < 3:
        raise ValueError("三折训练至少需要 3 位开发集患者，保证 train/val/test 非空")
    random.Random(seed).shuffle(order)
    folds = []
    for index in range(3):
        test = set(order[index::3])
        remainder = [p for p in order if p not in test]
        random.Random(seed + 1009 * (index + 1)).shuffle(remainder)
        n_val = max(1, round(len(order) * 0.1))
        if n_val >= len(remainder):
            raise ValueError("开发集患者不足以同时提供训练与验证数据")
        groups = {"train": set(remainder[n_val:]), "val": set(remainder[:n_val]), "test": test}
        folds.append(
            {
                "index": index,
                "patients": {part: sorted(group) for part, group in groups.items()},
                "cases": {
                    part: sorted(r["case_id"] for r in development if r["patient_id"] in group)
                    for part, group in groups.items()
                },
            }
        )
    development_ids = {r["case_id"] for r in development}
    return {
        "format": "imagechd7-patient-threefold-v1",
        "seed": seed,
        "id_policy": policy,
        "development_scope": scope,
        "validation_fraction_of_development_patients": 0.1,
        "cohort": cohort,
        "external_case_ids": [r["case_id"] for r in cohort if r["case_id"] not in development_ids],
        "folds": folds,
    }


def validate_plan(plan):
    """Recompute the stored plan rather than trusting edited assignments."""
    try:
        actual = make_plan(
            {"cases": plan["cohort"], "id_policy": plan["id_policy"]},
            plan["seed"],
            plan["development_scope"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("三折划分记录无效") from error
    if actual != plan:
        raise ValueError("三折划分记录已改变或存在患者泄漏")


def apply_fold(data, fold):
    assignment = {}
    for part in SPLITS:
        for case in fold["cases"][part]:
            if case in assignment:
                raise ValueError("同一病例跨折内集合")
            assignment[case] = part
    source = {r["case_id"]: r for r in data["cases"]}
    if not set(assignment).issubset(source):
        raise ValueError("折划分含缓存中不存在的病例")
    rows = [{**source[case], "split": part} for case, part in sorted(assignment.items())]
    for part in SPLITS:
        actual = {r["patient_id"] for r in rows if r["split"] == part}
        if not actual or actual != set(fold["patients"][part]):
            raise ValueError("折划分患者与缓存身份不匹配")
    return {**data, "cases": rows}


def held_out_index(plan, row):
    matches = {
        f["index"]
        for f in plan["folds"]
        if row.get("case_id") in f["cases"]["test"] or row.get("patient_id") in f["patients"]["test"]
    }
    if len(matches) > 1:
        raise ValueError("病例与患者身份指向不同测试折")
    return next(iter(matches)) if matches else None


def select_members(collection, row):
    index = held_out_index(collection["split_plan"], row)
    members = collection["_members"]
    if index is None:
        return members, "threefold_ensemble"
    chosen = [m for m in members if m["index"] == index]
    if len(chosen) != 1:
        raise ValueError("缺少该患者未参与训练的测试折模型")
    return chosen, "out_of_fold"


def select_rows(data, split, plan=None, fold=None):
    if split == "auto":
        split = "cv" if plan and any(held_out_index(plan, r) is not None for r in data["cases"]) else "test"
    if split == "cv":
        if plan is None:
            raise ValueError("--split cv 需要完整三折模型/预测记录")
        rows = [r for r in data["cases"] if held_out_index(plan, r) is not None]
    elif split is None:
        rows = data["cases"]
    elif split in SPLITS:
        rows = (
            [r for r in data["cases"] if r["case_id"] in fold["cases"][split]]
            if fold
            else [r for r in data["cases"] if r.get("split") == split]
        )
    else:
        raise ValueError("未知数据集合")
    return rows, split
