"""Small per-disease trees; native JSON artifacts and exact branch explanations."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_NAMES, FEATURE_SCHEMA, read_features
from .io import DISEASES, case_index, file_hash, new_output, read_artifact, save
from .labels import read_labels


def matrix(rows):
    return np.asarray(
        [[np.nan if r["values"][key] is None else r["values"][key] for key in FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )


def export_tree(estimator, x, y):
    tree = estimator.tree_
    paths = estimator.decision_path(x)
    nodes = []
    for i in range(tree.node_count):
        selected = paths[:, i].toarray().ravel().astype(bool)
        nodes.append(
            {
                "feature": int(tree.feature[i]),
                "threshold": float(tree.threshold[i]),
                "left": int(tree.children_left[i]),
                "right": int(tree.children_right[i]),
                "samples": int(selected.sum()),
                "positive": int(y[selected].sum()),
            }
        )
    return nodes


def train_classifier(features, labels, output, allow_smoke=False, max_depth=4, min_leaf=2, seed=42):
    if not 1 <= max_depth <= 8 or min_leaf < 2:
        raise ValueError("树深度须为 1—8，叶节点最少病例数至少为 2")
    fpath, data = read_features(features)
    lpath, truth = read_labels(labels)
    if data["segmentation_mode"] != "train" and not allow_smoke:
        raise ValueError("短测/来源未确认的分割特征只能用 --allow-smoke 验证软件流程")
    if data.get("segmentation_prepared_sha256") != data["prepared_sha256"] and not allow_smoke:
        raise ValueError("诊断训练须复用分割训练的同一预处理清单与患者划分；请重新预测/提取特征")
    indexed = case_index(truth["cases"])
    rows = data["cases"]
    if any(r["case_id"] not in indexed for r in rows):
        raise ValueError("诊断标签未覆盖特征病例；检查 ImageCHD index 映射")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val:
        raise ValueError("诊断训练需要独立且非空的 train/val 集合")
    xtrain = matrix(train)
    report = {
        "format": "chd-diagnosis-training-v1",
        "status": "running",
        "blank_policy": truth["blank_policy"],
        "features_sha256": file_hash(fpath),
        "labels_sha256": file_hash(lpath),
        "test_cases_used": 0,
        "training_cases": len(train),
        "validation_cases": len(val),
        "diseases": {},
        "trained_diseases": 0,
        "assumed_negative_counts": {
            d: sum(d in indexed[r["case_id"]].get("assumed_negative", []) for r in train) for d in DISEASES
        },
    }
    model = {
        "format": "chd-diagnosis-tree-v1",
        "status": "passed",
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "diseases": list(DISEASES),
        "trees": {},
        "skipped": {},
        "blank_policy": truth["blank_policy"],
        "segmentation_sha256": data["segmentation_sha256"],
        "segmentation_mode": data["segmentation_mode"],
        "prepared_sha256": data["prepared_sha256"],
        "development_patients": sorted({r["patient_id"] for r in train + val}),
        "development_cases": [r["case_id"] for r in train + val],
        "features_sha256": report["features_sha256"],
        "labels_sha256": report["labels_sha256"],
        "threshold": 0.5,
        "probability_note": "Unweighted positive fraction in training leaf; uncalibrated, not clinical risk.",
        "research_only": True,
        "software_only": bool(allow_smoke),
    }
    output = new_output(output)
    try:
        for disease in DISEASES:
            ytrain = np.array([indexed[r["case_id"]]["labels"][disease] for r in train], dtype=float)
            known = np.isfinite(ytrain)
            positives, negatives = int(np.sum(ytrain == 1)), int(np.sum(ytrain == 0))
            summary = {"positive": positives, "negative": negatives, "unknown": int((~known).sum())}
            report["diseases"][disease] = summary
            if positives < 2 or negatives < 2:
                model["skipped"][disease] = "训练集至少需要 2 个阳性和 2 个阴性"
                summary.update(status="not_trainable", reason=model["skipped"][disease])
                continue
            x, y = xtrain[known], ytrain[known].astype(int)
            fill = np.array(
                [float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.0 for col in x.T]
            )
            x = np.where(np.isfinite(x), x, fill).astype(np.float32)
            yval = np.array([indexed[r["case_id"]]["labels"][disease] for r in val], dtype=float)
            observed = np.isfinite(yval)
            can_select = set(yval[observed]) == {0, 1}
            depths = sorted({min(d, max_depth) for d in (2, 3, 4)}) if can_select else [min(2, max_depth)]
            best, best_score, best_depth = None, -1.0, None
            for depth in depths:
                estimator = DecisionTreeClassifier(
                    max_depth=depth, min_samples_leaf=min_leaf, random_state=seed
                )
                estimator.fit(x, y)
                if can_select:
                    exported = {"nodes": export_tree(estimator, x, y)}
                    decisions = [explain_tree(exported, row["values"])["status"] for row in val]
                    # Missing branch evidence abstains at deployment; penalize it during selection too.
                    score = float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        decisions[i] == ("positive" if label else "negative")
                                        for i in np.flatnonzero(yval == label)
                                    ]
                                )
                                for label in (0, 1)
                            ]
                        )
                    )
                else:
                    score = 0.0
                if score > best_score:
                    best, best_score, best_depth = estimator, score, depth
            model["trees"][disease] = {
                "nodes": export_tree(best, x, y),
                "imputation": fill.tolist(),
                "max_depth": best_depth,
                "min_leaf": min_leaf,
                "selection": "validation_balanced_correct_fraction_including_abstentions"
                if can_select
                else "fixed_shallow_default_no_two_class_validation",
            }
            summary.update(
                status="trained",
                max_depth=best_depth,
                validation_balanced_correct_fraction_including_abstentions=best_score if can_select else None,
            )
        report["trained_diseases"] = len(model["trees"])
        report["status"] = "passed" if model["trees"] else "not_trainable"
        if model["trees"]:
            save(output / "classifier.json", model)
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save(output / "trainability.json", report)


def read_classifier(path):
    path, model = read_artifact(path, "classifier.json", "chd-diagnosis-tree-v1")
    if (
        model.get("feature_names") != list(FEATURE_NAMES)
        or model.get("feature_schema") != FEATURE_SCHEMA
        or model.get("diseases") != list(DISEASES)
    ):
        raise ValueError("分类器特征/疾病协议不匹配")
    if not model.get("trees") or set(model["trees"]) | set(model.get("skipped", {})) != set(DISEASES):
        raise ValueError("分类器疾病集合不完整")
    for record in model["trees"].values():
        nodes = record["nodes"]
        reached = set()

        def visit(index):
            if type(index) is not int or index < 0 or index >= len(nodes) or index in reached:
                raise ValueError("分类树存在无效节点或循环")
            reached.add(index)
            node = nodes[index]
            if not 0 <= node["positive"] <= node["samples"] or node["samples"] < 1:
                raise ValueError("分类树样本计数无效")
            if node["left"] == -1 and node["right"] == -1:
                return
            if not 0 <= node["feature"] < len(FEATURE_NAMES) or not np.isfinite(node["threshold"]):
                raise ValueError("分类树特征或阈值无效")
            visit(node["left"])
            visit(node["right"])

        visit(0)
        if len(reached) != len(nodes):
            raise ValueError("分类树有不可达节点")
    return path, model


def explain_tree(record, values):
    index, path = 0, []
    while True:
        node = record["nodes"][index]
        if node["left"] == -1:
            fraction = node["positive"] / node["samples"]
            return {
                "status": "positive" if fraction > 0.5 else "negative",
                "probability": fraction,
                "path": path,
                "leaf": {"id": index, "samples": node["samples"], "positive": node["positive"]},
                "reason": "training_leaf_fraction" if path else "no_split_training_prior",
            }
        name = FEATURE_NAMES[node["feature"]]
        value = values[name]
        if value is None:
            path.append(
                {"feature": name, "observed": None, "threshold": node["threshold"], "condition": "missing"}
            )
            return {
                "status": "indeterminate",
                "probability": None,
                "path": path,
                "leaf": None,
                "reason": "missing_required_feature",
            }
        used = float(np.float32(value))  # Match sklearn's prediction precision.
        left = used <= node["threshold"]
        path.append(
            {
                "feature": name,
                "observed": value,
                "used_value": used,
                "operator": "<=" if left else ">",
                "threshold": node["threshold"],
                "condition": "satisfied",
            }
        )
        index = node["left"] if left else node["right"]
