"""Read-only result sections shared by the terminal browser and one-shot CLI."""

from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from . import inspection, results
from .result_decisions import POLICIES, show_decisions, show_metrics, show_training
from .result_io import count, mapping, named, number, read, rows, text
from .result_metrics import fold_details, summary

SECTIONS = {
    "segmentation": "分割结果",
    "decisions": "诊断决策",
    "diagnosis-evaluation": "诊断评估",
    "training": "训练与数据",
    "files": "文件与来源",
}
GROUPS = {
    "segmentation": {"test", "predict", "evaluate", "demo"},
    "decisions": {"diagnose", "diagnosis-demo"},
    "diagnosis-evaluation": {"diagnosis-evaluate", "diagnosis-demo"},
    "training": {
        "environment",
        "preprocess",
        "smoke",
        "preflight",
        "train",
        "diagnosis-labels",
        "diagnosis-features",
        "diagnosis-train",
        "diagnosis-demo",
    },
}
REPORTS = {
    "test": ("test/evaluation/evaluation.json", "test/predictions/prediction-report.json"),
    "predict": ("predict/prediction-report.json",),
    "evaluate": ("evaluate/evaluation.json",),
    "demo": ("demo/evaluation/evaluation.json",),
    "diagnose": ("diagnose/diagnosis.json",),
    "diagnosis-evaluate": ("diagnosis-evaluate/diagnosis-metrics.json",),
    "diagnosis-train": ("diagnosis-train/trainability.json",),
    "diagnosis-demo": (
        "diagnosis-demo/training/trainability.json",
        "diagnosis-demo/prediction/diagnosis.json",
        "diagnosis-demo/evaluation/diagnosis-metrics.json",
    ),
}


def select_steps(job, step="auto"):
    if not job.is_dir():
        raise ValueError("结果目录不存在：" + str(job))
    steps = results.steps_for(job)
    selected = steps if step == "auto" else [s for s in steps if step in (s["name"], s["output"].name)]
    if step != "auto" and not selected:
        raise ValueError("任务没有此步骤：" + step)
    return selected


def header(job):
    steps = select_steps(job)
    current = read(job / "job_state.json")
    done = sum(s.get("status") in ("passed", "complete", "reused") for s in steps)
    print("结果中心：" + results.title(job))
    print(f"状态：{results.state(current.get('status'))} | 已完成步骤 {done}/{len(steps)}")
    print("只读查看；后台任务继续运行。")


def report_notice(report):
    print("报告状态：" + results.state(report.get("status")))
    if report.get("status") not in ("passed", "complete"):
        print("记录尚未完整通过，以下仅为已有结果。")
    if report.get("software_only") or report.get("segmentation_mode") in ("smoke", "preflight", "synthetic"):
        print("仅用于软件验证，不代表正式性能。")


def model_notice(job, item):
    path = results.prediction_source(job, item)
    report = read(path) if path else {}
    mode = report.get("model_mode", report.get("mode"))
    if item["name"] == "demo" or report.get("software_only") or mode in ("smoke", "preflight"):
        print("模型用途：短测/合成软件验证，不代表正式分割性能。")
    elif mode == "train":
        print("模型用途：正式训练模型。")
    else:
        print("模型用途：未记录，无法确认正式/短测。")


def segmentation(job, item, detailed):
    name, output = item["name"], item["output"]
    model_notice(job, item)
    if name == "demo":
        print("数据：程序生成的合成数据。")
    path = output / REPORTS[name][0]
    if detailed:
        if name == "predict":
            results.prediction(path)
        else:
            results.evaluation(path)
            prediction = output / "test/predictions/prediction-report.json"
            if name == "test" and prediction.is_file():
                results.prediction(prediction)
        return
    report = read(path, warn=True)
    if not report:
        return
    report_notice(report)
    print("集合：" + text(report.get("split", "未记录")))
    results.scope(report, prediction_only=name == "predict")
    if name == "predict":
        cases = report.get("cases")
        print("预测记录数：" + count(len(cases) if isinstance(cases, list) else None))
        print("只预测，不计算 Dice。")
        return
    value, derived = summary(report, rows(report.get("cases")), results.LABELS)
    print(
        f"评估 {count(value.get('cases_evaluated'))} 例 | 平均前景 Dice {number(value.get('mean_case_foreground_dice'))}"
    )
    if derived:
        print("缺失摘要由完整的已保存病例数值补算。")
    for fold, metrics, _, computed in fold_details(report, results.LABELS):
        print(
            f"{fold}：{count(metrics.get('cases_evaluated'))} 例 | Dice "
            + number(metrics.get("mean_case_foreground_dice"))
            + ("（明细补算）" if computed else "")
        )
    print("Dice 是分割重叠度；详细页查看七结构、有效分母与病例。")


def diagnosis_path(item, section):
    if item["name"] == "diagnosis-demo":
        suffix = (
            "prediction/diagnosis.json" if section == "decisions" else "evaluation/diagnosis-metrics.json"
        )
        return item["output"] / "diagnosis-demo" / suffix
    return item["output"] / REPORTS[item["name"]][0]


def diagnostic_summary(report):
    report_notice(report)
    print("集合：" + text(report.get("split", "未记录")))
    print("分割特征模式：" + text(report.get("segmentation_mode", "未记录")))
    print("标签空白：" + named(report.get("blank_policy", report.get("classifier_blank_policy")), POLICIES))


def decisions(item, detailed):
    path = diagnosis_path(item, "decisions")
    if detailed:
        show_decisions(path)
        return
    report = read(path, warn=True)
    if not report:
        return
    diagnostic_summary(report)
    cases = report.get("cases")
    print("病例记录数：" + count(len(cases) if isinstance(cases, list) else None))
    print("以下是已保存判断数量，不是准确率；证据不足不算阴性。")
    for group, label in (("classifier", "决策树"), ("rules", "规则")):
        diseases = {}
        for case in rows(cases):
            for disease, raw in mapping(case.get(group)).items():
                status = mapping(raw).get("status")
                key = status if status in ("positive", "negative", "indeterminate") else "unknown"
                diseases.setdefault(disease, Counter())[key] += 1
        if not diseases:
            print(label + "：未记录/未使用")
        for disease, counts in diseases.items():
            print(
                f"{label} · {text(disease)}：阳性候选 {counts['positive']} | 阴性 {counts['negative']}"
                f" | 证据不足 {counts['indeterminate']} | 状态未记录 {counts['unknown']}"
            )
    priors = sum(
        mapping(finding).get("reason") == "no_split_training_prior"
        for case in rows(cases)
        for finding in mapping(case.get("classifier")).values()
    )
    if priors:
        print(f"其中 {priors} 项仅为训练先验，没有个体判断分支；叶节点比例未经临床风险校准。")


def diagnosis_evaluation(item, detailed):
    path = diagnosis_path(item, "diagnosis-evaluation")
    if detailed:
        show_metrics(path)
        return
    report = read(path, warn=True)
    if not report:
        return
    diagnostic_summary(report)
    print("评估病例数：" + count(report.get("cases_evaluated")))
    print("统计单位：患者-疾病对；准确率排除弃判和未知真值，需同时看覆盖率。")
    for group, label in (("classifier_micro", "决策树"), ("rules_micro", "规则")):
        record = mapping(report.get(group))
        if not record:
            print(label + "：未记录/未评估")
            continue
        print(
            f"{label}：准确率 {number(record.get('accuracy'), True)} | 覆盖率 {number(record.get('coverage'), True)}"
        )
        print(
            f"  已知 {count(record.get('known'))} | 已判定 {count(record.get('decided'))}"
            f" | 弃判 {count(record.get('abstained'))} | 未知真值 {count(record.get('unknown_truth'))}"
        )


def training(item, detailed):
    name, output = item["name"], item["output"]
    if name in inspection.MODEL_STEPS:
        if detailed:
            results.training(output, name)
        else:
            report, models = read(output / "train/report.json"), read(output / "train/models.json")
            if name in ("smoke", "preflight") or models.get("mode", report.get("mode")) in (
                "smoke",
                "preflight",
            ):
                print("短测/预检，仅用于软件验证。")
            if not report and not models:
                print("尚无训练结果清单；已保存状态不代表模型文件齐全。")
            elif models.get("format") == "imagechd7-threefold-v1" or report.get("task") == "chd_threefold":
                folds = mapping(report.get("folds"))
                for fold in inspection.FOLDS:
                    record = mapping(folds.get(fold))
                    print(
                        f"{fold}：{results.state(record.get('status'))} | 训练/验证/测试 "
                        f"{count(record.get('train_cases'))}/{count(record.get('val_cases'))}/{count(record.get('test_cases'))}"
                    )
            else:
                print(f"单次划分：{len(mapping(models.get('stages')))}/6 个阶段有模型记录。")
            print("训练完成不等于测试完成；loss 不等于 Dice。")
    elif name in ("diagnosis-train", "diagnosis-demo"):
        path = output / REPORTS[name][0]
        if detailed:
            show_training(path)
        else:
            report = read(path, warn=True)
            if report:
                diagnostic_summary({**read(path.parent / "classifier.json"), **report})
                print(
                    f"训练/验证病例：{count(report.get('training_cases', report.get('train_cases')))}/"
                    f"{count(report.get('validation_cases', report.get('val_cases')))}；"
                    f"已训练疾病模型：{count(report.get('trained_diseases'))}"
                )
    else:
        print("准备步骤状态以上方记录为准；未保存内容不作推断。")
    prepared = results.step_arguments(item).get("prepared")
    if isinstance(prepared, str) and prepared:
        if detailed:
            results.cache_details(prepared)
        elif name == "preprocess":
            data = read(Path(prepared).expanduser() / "dataset.json")
            retained = data.get("cases")
            print(
                f"缓存发现 {count(data.get('discovered'))} 例 | 保留 "
                f"{count(len(retained) if isinstance(retained, list) else None)} 例；不是本次评估数。"
            )


def files(job, steps, detailed):
    print("任务目录：" + text(job))
    for item in steps:
        print(results.STEPS.get(item["name"], text(item["name"])) + "：")
        paths = [item["output"] / p for p in REPORTS.get(item["name"], ())]
        if item["name"] in inspection.MODEL_STEPS:
            paths += [item["output"] / "train/models.json", item["output"] / "train/report.json"]
        existing = [p for p in paths if p.is_file()]
        for path in existing:
            print("  " + text(path))
        if not existing:
            print("  输出目录（尚无已识别报告）：" + text(item["output"]))
        if detailed:
            args = results.step_arguments(item)
            for key, label in (
                ("dataset", "原始数据"),
                ("prepared", "数据/真值缓存"),
                ("models", "分割模型"),
                ("predictions", "预测输入"),
                ("features", "诊断特征"),
                ("classifier", "决策树"),
                ("diagnosis_labels", "诊断真值"),
                ("diagnoses", "诊断判断"),
                ("evidence", "人工证据"),
            ):
                if args.get(key):
                    print("  " + label + "：" + text(args[key]))
            if item["name"] in {"test", "predict", "evaluate", "demo"}:
                results.evaluation_mode(job, item)
    if detailed:
        plan = read(job / "plan.json")
        for key, label in (("input_job", "预处理输入任务"), ("rerun_of", "重跑来源任务")):
            source = mapping(plan.get(key))
            if source:
                print(label + "：" + text(source.get("name")) + " | " + text(source.get("directory")))
        for receipt in sorted((job / "uploads").glob("*.json")):
            url = read(receipt).get("url")
            if isinstance(url, str) and url.startswith("https://github.com/tianwuasd/project/releases/"):
                print("已上传产物：" + text(url))
        results.commands(job, steps)


def render(job, section="all", detail="summary", step="auto"):
    job = Path(job).expanduser().resolve()
    if section not in ("all", *SECTIONS) or detail not in ("summary", "detailed"):
        raise ValueError("未知的结果分区或查看级别")
    steps = select_steps(job, step)
    stream = StringIO()
    with redirect_stdout(stream):
        if section == "all" and detail == "detailed":
            results.show_results(job, step)
        else:
            sections = SECTIONS if section == "all" else (section,)
            for key in sections:
                print("\n" + SECTIONS[key])
                if key == "files":
                    files(job, steps, detail == "detailed")
                    continue
                selected = [s for s in steps if s["name"] in GROUPS[key]]
                if not selected:
                    print("没有此部分的步骤记录。")
                for item in selected:
                    print(
                        "\n"
                        + results.STEPS.get(item["name"], text(item["name"]))
                        + " | "
                        + results.state(item.get("status"))
                    )
                    if item["name"] == "diagnosis-demo":
                        print("合成数据，仅用于软件验证，不代表真实诊断性能。")
                    if key == "segmentation":
                        segmentation(job, item, detail == "detailed")
                    elif key == "decisions":
                        decisions(item, detail == "detailed")
                    elif key == "diagnosis-evaluation":
                        diagnosis_evaluation(item, detail == "detailed")
                    elif key == "training":
                        training(item, detail == "detailed")
    return stream.getvalue()


def show_results(job, step="auto", section="all", detail="summary"):
    if not (section == "all" and detail == "detailed"):
        header(Path(job).expanduser().resolve())
    print(render(job, section, detail, step), end="")
    if detail == "summary":
        print("详细内容：--result-detail detailed；按部分查看：--result-section <分区名>。")
