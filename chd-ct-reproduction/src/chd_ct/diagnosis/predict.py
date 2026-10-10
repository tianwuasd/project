"""Separate rules and tree inference; faithful, escaped human-readable explanations."""

import html
import json
from pathlib import Path

from .classifier import explain_tree, read_classifier
from .features import RULE_KEYS, read_features
from .io import DISEASES, case_index, file_hash, new_output, save
from .rules import evaluate_rules, load_rules, validate_evidence

NAMES = {
    "LV": "左心室",
    "RV": "右心室",
    "LA": "左心房",
    "RA": "右心房",
    "MYO": "心肌",
    "AO": "主动脉",
    "PA": "肺动脉",
}


def feature_title(name):
    for prefix, label in (
        ("present_", "预测结构是否存在"),
        ("fraction_", "占心脏前景的体素比例"),
        ("components_", "预测连通部分数量"),
        ("largest_fraction_", "最大连通部分占比"),
        ("extent_x_", "网格第1轴跨度比例"),
        ("extent_y_", "网格第2轴跨度比例"),
        ("extent_z_", "网格第3轴跨度比例"),
    ):
        if name.startswith(prefix):
            return NAMES[name[len(prefix) :]] + "：" + label
    if name.startswith("contact_"):
        a, b = name.removeprefix("contact_").split("_")
        return NAMES[a] + "—" + NAMES[b] + "标签相邻（连接代理特征）"
    return name


def render_report(report):
    def esc(x):
        return html.escape(str(x))

    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>诊断与判断依据</title><style>body{font:16px/1.7 system-ui;margin:32px auto;max-width:1000px;padding:0 20px;color:#17283b;background:#f5f7fa}section,details{background:white;padding:16px;margin:12px 0;border-radius:10px}summary{cursor:pointer;font-weight:600}small{color:#566}li{margin:5px 0}</style><h1>诊断与判断依据</h1>',
        "<p>研究模型输出。解释来自实际决策路径；表示模型判别依据，不表示疾病成因。叶节点比例未经临床风险校准。</p>",
        "<p>训练标签空白处理："
        + esc(
            {"negative": "空白按阴性（0）", "unknown": "空白保留未知"}.get(
                report.get("classifier_blank_policy"), "未使用分类器"
            )
        )
        + "</p>",
    ]
    if report.get("software_only"):
        parts.append("<p>本报告来自软件短测／合成验证，不能作为真实诊断性能。</p>")
    names = {"positive": "阳性候选", "negative": "阴性判断", "indeterminate": "证据不足"}
    for row in report["cases"]:
        parts.append("<section><h2>" + esc(row["case_id"]) + "</h2>")
        for disease, finding in row["classifier"].items():
            parts.append(
                "<details><summary>" + esc(disease) + "：" + names[finding["status"]] + "</summary><ol>"
            )
            for step in finding.get("path", []):
                text = feature_title(step["feature"])
                if step["observed"] is None:
                    text += "；该特征缺失，停止判定。"
                else:
                    text += f" = {step['observed']:.6g}，满足 {step['operator']} {step['threshold']:.6g}"
                parts.append("<li>" + esc(text) + "</li>")
            parts.append("</ol>")
            if finding.get("reason") == "no_split_training_prior":
                parts.append("<p>此树没有可用的个体判断分支，仅输出训练先验；没有个体解剖分支依据。</p>")
            leaf = finding.get("leaf")
            if leaf:
                parts.append(
                    "<p>"
                    + esc(
                        f"到达叶节点 {leaf['id']}：训练病例 {leaf['samples']} 例，其中阳性 {leaf['positive']} 例，比例 {finding['probability']:.1%}。"
                    )
                    + "</p>"
                )
            elif finding.get("reason"):
                parts.append("<p>" + esc(finding["reason"]) + "</p>")
            parts.append("</details>")
        if row["rules"]:
            parts.append(
                "<details><summary>候选规则结果（独立对照）</summary><pre>"
                + esc(json.dumps(row["rules"], ensure_ascii=False, indent=2))
                + "</pre></details>"
            )
        parts.append("</section>")
    return "".join(parts) + "</html>"


def diagnose(
    features,
    output,
    classifier=None,
    method="both",
    split="test",
    evidence=None,
    rules=None,
    allow_smoke=False,
):
    fpath, data = read_features(features)
    if method not in {"tree", "rules", "both"}:
        raise ValueError("诊断方法须为 tree/rules/both")
    if data["segmentation_mode"] != "train" and not allow_smoke:
        raise ValueError("短测/未确认分割来源需要 --allow-smoke")
    selected = [r for r in data["cases"] if split == "all" or r["split"] == split]
    if not selected:
        raise ValueError("所选诊断集合为空；无标签新病例请使用 --split all")
    model, modelpath = None, None
    if method in {"tree", "both"}:
        if not classifier:
            raise ValueError("树分类需要 --classifier；只运行规则可选择 --method rules")
        modelpath, model = read_classifier(classifier)
        if (model.get("software_only") or model["segmentation_mode"] != "train") and not allow_smoke:
            raise ValueError("软件短测分类器需显式 --allow-smoke")
        if model["segmentation_sha256"] != data["segmentation_sha256"]:
            raise ValueError("分割模型来源与分类器训练特征不同，请重新提取一致来源的特征")
    config = load_rules(rules) if method in {"rules", "both"} else None
    reviewed = {}
    if evidence:
        raw = json.loads(Path(evidence).read_text(encoding="utf-8"))
        reviewed = case_index(raw["cases"])
        if set(reviewed) - set(case_index(data["cases"])):
            raise ValueError("人工解剖证据包含特征清单之外的病例")
        for row in reviewed.values():
            if not isinstance(row.get("source"), str) or not row["source"].strip():
                raise ValueError("人工解剖证据需要 source 记录来源")
            validate_evidence(row["features"])
    result = {
        "format": "chd-diagnosis-predictions-v1",
        "status": "passed",
        "method": method,
        "split": split,
        "features_sha256": file_hash(fpath),
        "classifier_sha256": file_hash(modelpath) if modelpath else None,
        "classifier_blank_policy": model["blank_policy"] if model else None,
        "development_patients": model["development_patients"] if model else [],
        "development_cases": model["development_cases"] if model else [],
        "segmentation_mode": data["segmentation_mode"],
        "research_only": True,
        "rule_set": config.get("name") if config else None,
        "effective_rules": config,
        "software_only": bool(allow_smoke or (model and model.get("software_only"))),
        "evidence_sha256": file_hash(evidence) if evidence else None,
        "cases": [],
    }
    for row in selected:
        trees = {}
        if model:
            for disease in DISEASES:
                trees[disease] = (
                    explain_tree(model["trees"][disease], row["values"])
                    if disease in model["trees"]
                    else {
                        "status": "indeterminate",
                        "probability": None,
                        "path": [],
                        "leaf": None,
                        "reason": model["skipped"][disease],
                    }
                )
        # Do not reinterpret raw voxel contacts as verified anatomical communication.
        known = dict.fromkeys(RULE_KEYS)
        annotation = reviewed.get(row["case_id"])
        if annotation:
            known.update(annotation["features"])
        findings = evaluate_rules(known, config) if config else {}
        result["cases"].append(
            {
                "case_id": row["case_id"],
                "patient_id": row["patient_id"],
                "split": row["split"],
                "classifier": trees,
                "rules": findings,
                "rule_evidence_source": annotation["source"] if annotation else None,
            }
        )
    output = new_output(output)
    save(output / "diagnosis.json", result)
    (output / "diagnosis-report.html").write_text(render_report(result), encoding="utf-8")
    return result
