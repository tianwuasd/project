"""Display existing decision explanations and metrics without importing ML code."""

from .result_io import count, mapping, named, number, read, rows, text

FINDINGS = {"positive": "阳性候选", "negative": "阴性判断", "indeterminate": "证据不足"}
POLICIES = {"negative": "空白按阴性（0）", "unknown": "空白保留未知"}
REASONS = {
    "missing_required_feature": "判断所需特征缺失，停止判定",
    "no_split_training_prior": "没有个体判断分支，仅输出训练先验",
    "training_leaf_fraction": "根据实际到达的训练叶节点作出判断",
    "conflicting_candidate_rules": "候选规则相互冲突，保留为证据不足",
    "missing_evidence": "规则所需证据缺失",
}
OPERATORS = {"eq": "=", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}


def metadata(report):
    print("  记录状态：" + text(report.get("status", "未记录")))
    print("  数据集合：" + text(report.get("split", "未记录")))
    print("  分割特征来源模式：" + text(report.get("segmentation_mode", "未记录")))
    policy = report.get("blank_policy", report.get("classifier_blank_policy"))
    print("  标签空白处理：" + named(policy, POLICIES))
    if report.get("software_only") or report.get("segmentation_mode") in ("smoke", "preflight", "synthetic"):
        print("  此记录仅用于软件验证，不能作为正式诊断性能。")
    print("  疾病名称沿用诊断表代码；不能与同名分割结构代码混用。")


def show_training(path):
    report = read(path, warn=True)
    if not report:
        return
    model = read(path.parent / "classifier.json")
    metadata({**model, **report})
    print(
        f"  训练/验证病例：{count(report.get('training_cases', report.get('train_cases')))}/"
        f"{count(report.get('validation_cases', report.get('val_cases')))}；"
        f"训练使用测试病例：{count(report.get('test_cases_used'))}"
    )
    print("  已训练疾病模型数：" + count(report.get("trained_diseases")))
    for disease, item in mapping(report.get("diseases")).items():
        record = mapping(item)
        print(
            f"  {text(disease)}：{named(record.get('status'), {'trained': '已训练', 'not_trainable': '不能训练'})}；"
            f"阳性/阴性/未知 {count(record.get('positive'))}/{count(record.get('negative'))}/"
            f"{count(record.get('unknown'))}；深度 {count(record.get('max_depth'))}"
        )
        if record.get("reason"):
            print("    原因：" + text(record["reason"]))
        if "validation_balanced_correct_fraction_including_abstentions" in record:
            print(
                "    验证集均衡正确比例（计入弃判）："
                + number(record["validation_balanced_correct_fraction_including_abstentions"])
            )
    print("  训练可用性报告：" + text(path))
    print("  训练完成不代表诊断测试通过；测试指标见菜单 12。")


def feature_name(value):
    names = {
        "LV": "左心室",
        "RV": "右心室",
        "LA": "左心房",
        "RA": "右心房",
        "MYO": "心肌",
        "AO": "主动脉",
        "PA": "肺动脉",
    }
    if not isinstance(value, str):
        return "未记录"
    for prefix, label in (
        ("present_", "预测结构是否存在"),
        ("fraction_", "占前景体素比例"),
        ("components_", "连通部分数量"),
        ("largest_fraction_", "最大连通部分占比"),
        ("extent_x_", "网格第1轴跨度比例"),
        ("extent_y_", "网格第2轴跨度比例"),
        ("extent_z_", "网格第3轴跨度比例"),
    ):
        if value.startswith(prefix) and value[len(prefix) :] in names:
            return f"{names[value[len(prefix) :]]}{label} [{text(value)}]"
    if value.startswith("contact_"):
        return "标签相邻代理特征（不等于已验证的解剖交通）[" + text(value) + "]"
    return text(value)


def show_branch(branch, rule=False):
    label = feature_name(branch.get("feature"))
    if branch.get("observed") is None:
        print("      " + label + "：未记录/缺失，不能补造判断依据。")
        return
    operator = named(branch.get("operator"), OPERATORS)

    def value_text(value):
        return ("1（是）" if value else "0（否）") if isinstance(value, bool) else number(value)

    print(
        f"      {label}：观测 {value_text(branch.get('observed'))}；"
        f"实际比较值 {value_text(branch.get('used_value', branch.get('observed')))} "
        f"{operator} {value_text(branch.get('threshold'))}"
    )
    if rule:
        value = branch.get("satisfied")
        print("        条件结果：" + ("满足" if value is True else "不满足" if value is False else "未知"))


def show_decisions(path):
    report = read(path, warn=True)
    if not report:
        return
    metadata(report)
    print(
        "  决策方法："
        + named(report.get("method"), {"tree": "决策树", "rules": "候选规则", "both": "决策树与规则分别输出"})
    )
    cases = rows(report.get("cases"))
    print(f"  本报告病例记录数：{len(cases)}；每例按疾病分别作出判断。")
    print("  树比例为训练叶节点阳性比例，未经临床风险校准；判断路径不是疾病成因解释。")
    for case in cases:
        print(
            f"\n  病例 {text(case.get('case_id', '未记录'))} | 患者 {text(case.get('patient_id', '未记录'))}"
            f" | 集合 {text(case.get('split', '未记录'))}"
        )
        for group, label in (("classifier", "决策树"), ("rules", "候选规则（独立对照）")):
            findings = mapping(case.get(group))
            print("    " + label + ("：" if findings else "：未记录/未使用"))
            if group == "rules":
                print("    人工证据来源：" + text(case.get("rule_evidence_source") or "未记录/未提供"))
            for disease, raw in findings.items():
                finding = mapping(raw)
                print(f"    {text(disease)}：{named(finding.get('status'), FINDINGS)}")
                reason = finding.get("reason")
                if reason:
                    print("      原因：" + named(reason, REASONS))
                if group == "classifier" and not rows(finding.get("path")):
                    print("      判断路径：未记录/无已保存条件；是否为训练先验以上述来源原因为准。")
                for branch in rows(finding.get("path" if group == "classifier" else "evidence")):
                    show_branch(branch, rule=group == "rules")
                missing = finding.get("missing")
                if isinstance(missing, list):
                    print("      缺失证据：" + "、".join(text(v) for v in missing))
                leaf = mapping(finding.get("leaf"))
                if leaf:
                    print(
                        f"      叶节点 {text(leaf.get('id', '未记录'))}：训练阳性/总数 "
                        f"{count(leaf.get('positive'))}/{count(leaf.get('samples'))}；"
                        f"阳性比例 {number(finding.get('probability'), True)}"
                    )
    print("  决策清单：" + text(path))
    html = path.parent / "diagnosis-report.html"
    if html.is_file():
        print("  可展开的详细报告：" + text(html))


def metrics_detail(record):
    print(
        f"    已知真值 {count(record.get('known'))} | 未知真值 {count(record.get('unknown_truth'))} | "
        f"已判定 {count(record.get('decided'))} | 弃判 {count(record.get('abstained'))}"
    )
    print("    TP/TN/FP/FN：" + "/".join(count(record.get(k)) for k in ("tp", "tn", "fp", "fn")))
    print(
        f"    准确率 {number(record.get('accuracy'), True)} | 覆盖率 {number(record.get('coverage'), True)} | "
        f"F1 {number(record.get('f1'))}"
    )
    print(
        f"    敏感度 {number(record.get('sensitivity'))} | 特异度 {number(record.get('specificity'))} | "
        f"精确率 {number(record.get('precision'))} | AUROC {number(record.get('auroc'))}"
    )
    print(
        "    全部已知真值中的正确比例（包含弃判分母）："
        + number(record.get("correct_fraction_all_known"), True)
    )


def show_metrics(path):
    report = read(path, warn=True)
    if not report:
        return
    metadata(report)
    print("  评估病例数：" + count(report.get("cases_evaluated")))
    if report.get("status") not in ("passed", "complete"):
        print("  此诊断评估尚未完整通过，以下仅为已有记录。")
    print("  统计单位是患者-疾病对；准确率/敏感度/特异度排除弃判和未知真值。")
    print("  覆盖率=已判定/已知真值；TP 真阳性、TN 真阴性、FP 假阳性、FN 假阴性。")
    print("  按标签约定将空白计为阴性的单元格：" + count(report.get("assumed_negative_cells")))
    for group, label in (("classifier", "决策树"), ("rules", "候选规则")):
        print("\n  " + label + "整体汇总")
        metrics_detail(mapping(report.get(group + "_micro")))
        diseases = mapping(report.get(group))
        if not diseases:
            print("  逐疾病指标：未记录")
        for disease, values in diseases.items():
            print("  " + label + " · " + text(disease))
            metrics_detail(mapping(values))
    alignment = mapping(report.get("rules_alignment"))
    for key, label in (
        ("rule_diseases_without_matching_truth", "规则无匹配真值代码"),
        ("label_diseases_without_rule", "真值疾病无对应规则"),
    ):
        values = alignment.get(key)
        if isinstance(values, list):
            print("  " + label + "：" + ("、".join(text(v) for v in values) or "无"))
    print("  诊断评估文件：" + text(path))
