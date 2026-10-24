"""Human-readable existing results, using only the standard library and read-only I/O."""

import shlex
from pathlib import Path

from . import inspection
from .result_io import count, mapping, named, number, read, rows, text
from .result_metrics import denominator, fold_details, fold_index, summary

LABELS = {
    "LV": "左心室",
    "RV": "右心室",
    "LA": "左心房",
    "RA": "右心房",
    "MYO": "心肌",
    "AO": "主动脉",
    "PA": "肺动脉",
}
STATES = {
    "passed": "已完成",
    "complete": "已完成",
    "reused": "已复用",
    "failed": "失败",
    "interrupted": "已中断",
    "running": "运行中",
    "pending": "未开始",
    "pending_cpu": "计算完成，等待 CPU 后处理",
    "queued": "已提交",
    "waiting_gpu": "等待显卡",
}
STEPS = {
    "environment": "环境检查",
    "preprocess": "预处理",
    "smoke": "软件短测",
    "preflight": "训练预检",
    "train": "正式训练",
    "test": "菜单 5 · 测试集预测和评估",
    "predict": "菜单 6 · 独立预测（不计算 Dice）",
    "evaluate": "菜单 7 · 独立分割评估（读取已有预测）",
    "diagnosis-labels": "菜单 8 · 导入诊断标签",
    "diagnosis-features": "菜单 9 · 提取诊断特征",
    "diagnosis-train": "菜单 10 · 训练诊断树",
    "diagnose": "菜单 11 · 诊断与解释",
    "diagnosis-evaluate": "菜单 12 · 诊断评估",
    "demo": "菜单 13 · 合成分割测试",
    "diagnosis-demo": "菜单 14 · 合成诊断测试",
}
PROTOCOLS = {
    "out_of_fold": "三折独立测试（OOF）",
    "threefold_ensemble": "三折集成",
    "single_split": "单次划分",
}


SPLITS = {
    "cv": "开发集的留出折病例（交叉验证）",
    "train": "原记录 train 集合",
    "val": "原记录 val 集合",
    "test": "原记录 test 集合",
    "all": "全部病例",
}


def case_list(cases):
    ids = [text(r["case_id"]) for r in cases if isinstance(r.get("case_id"), str)]
    print(f"  病例名单：{'、'.join(ids) if ids else '未记录'}")
    if ids and len(ids) != len(cases):
        print("  部分病例编号未记录，名单不完整。")


def scope(report, prediction_only=False):
    split = report.get("split")
    description = (
        "全部病例（此预测清单范围）"
        if prediction_only and "split" in report and split is None
        else named(split, SPLITS)
    )
    print("  数据范围：" + description)
    protocols = report.get("protocols")
    if not isinstance(protocols, list):
        protocols = [r.get("inference_protocol") for r in rows(report.get("cases"))]
    protocols = sorted({p for p in protocols if isinstance(p, str)})
    print("  推理方式：" + ("、".join(named(p, PROTOCOLS) for p in protocols) or "未记录"))
    if "out_of_fold" in protocols:
        print("  OOF：按记录，每例使用其留出测试折模型；分折不是同一批病例跑三次。")
    if "threefold_ensemble" in protocols:
        print("  集成模型的结果，不是三个独立测试折的分数；不能据此拆成三折指标。")
    if split in ("train", "val"):
        print("  集合名不等于独立测试；是否留出以预测协议及原始划分记录为准。")


def state(value):
    return STATES.get(value, text(value or "未记录")) if isinstance(value, str) else "未记录"


def steps_for(job):
    """Use stored paths, falling back to the selected workflow after it was moved."""
    records = {}
    for filename in ("plan.json", "job_state.json"):
        for row in rows(read(job / filename).get("steps")):
            name = row.get("name")
            if isinstance(name, str):
                records[name] = {**records.get(name, {}), **row}
    # Very old/incomplete workflow metadata can still have numbered step folders.
    for folder in sorted((job / "steps").glob("*")):
        if folder.is_dir() and "_" in folder.name:
            name = folder.name.split("_", 1)[1]
            records.setdefault(name, dict(name=name, output=str(folder)))
    result = []
    for row in records.values():
        raw = row.get("output")
        if not isinstance(raw, str) or not raw:
            continue
        old = Path(raw).expanduser()
        output = old if old.is_absolute() else job / old
        local = job / "steps" / old.name
        if local.is_dir():
            output = local
        result.append({**row, "output": output})
    return result


def result_history(runtime, results):
    """Read-only browser may find old results from a different/missing runtime."""
    runtime = Path(runtime)
    saved_pointer = read(runtime / "last-job.json").get("directory")
    pointer = (
        Path(saved_pointer).expanduser().resolve()
        if isinstance(saved_pointer, str) and saved_pointer
        else None
    )
    roots = {Path(results).expanduser()}
    saved = read(runtime / ".server-settings.json").get("results")
    if isinstance(saved, str) and saved:
        roots.add(Path(saved).expanduser())
    if pointer:
        roots.add(pointer.parent)
    candidates = {p for root in roots for p in root.glob("*_workflow") if p.is_dir()}
    candidates.update(
        Path(p).expanduser().resolve()
        for p in read(runtime / "job-history.json", kind=list)
        if isinstance(p, str) and p
    )
    if pointer and pointer.is_dir():
        candidates.add(pointer)
    return sorted(
        (p for p in candidates if any((p / f).is_file() for f in ("job_state.json", "plan.json"))),
        key=lambda p: p.name,
        reverse=True,
    )


def title(job):
    return text(read(job / "job_state.json").get("name") or read(job / "plan.json").get("name") or job.name)


def commands(job, steps):
    from .launcher import ROOT

    print("\n可复制的查询命令（只读，可在任意目录执行）：")
    saved = read(job / "job_state.json")
    original_steps = rows(saved.get("steps"))
    original_paths = {
        r.get("name"): Path(r["output"]).expanduser()
        for r in original_steps
        if isinstance(r.get("name"), str) and isinstance(r.get("output"), str)
    }
    compatible = bool(original_steps) and all(original_paths.get(s["name"]) == s["output"] for s in steps)
    requests = [("results", "结果摘要", [])]
    trained = next((s["name"] for s in reversed(steps) if s["name"] in inspection.MODEL_STEPS), None)
    if trained:
        requests.append(("models" if compatible else "results", "训练记录", ["--step", trained]))
    if compatible:
        requests.append(("logs", "详细日志", ["--step", "launcher"]))
    for query, label, extra in requests:
        args = ["python3", str(ROOT / "start_server.py"), "--query", query, "--job-dir", str(job)]
        print(f"  {label}：\n  {shlex.join(args + extra)}")
    if not compatible:
        logs = [job / "launcher.log", *(s["output"] / "server.log" for s in reversed(steps))]
        path = next((p for p in logs if p.is_file()), None)
        if path:
            print("  现有日志：\n  " + shlex.join(["tail", "-n", "60", str(path)]))
        else:
            print("  暂无可读取的日志文件。")


def step_arguments(item):
    return {**read(item["output"] / "server-settings.json"), **mapping(item.get("arguments"))}


def cache_details(value):
    if not isinstance(value, str) or not value:
        return
    directory = Path(value).expanduser()
    data = read(directory / "dataset.json")
    if not data:
        print("  数据清单不可读：无法确认队列数量与真值可用性。")
        return
    print("  数据清单：" + text(directory / "dataset.json"))
    retained, excluded = data.get("cases"), data.get("excluded")
    print(
        f"  缓存记录：发现 {count(data.get('discovered'))} 例，"
        f"保留 {count(len(retained) if isinstance(retained, list) else None)} 例；"
        f"排除记录 {count(len(excluded) if isinstance(excluded, list) else None)} 条。"
    )
    prediction_only = data.get("for_prediction")
    print(
        "  真值用途："
        + (
            "无标签预测缓存，不能计算 Dice。"
            if prediction_only is True
            else "按带标签缓存处理；本次查询不读取标签影像。"
            if prediction_only is False
            else "未记录，无法确认真值可用性。"
        )
    )
    sources = read(directory / "preprocess-contract.json").get("sources")
    if isinstance(sources, list):
        roots = sorted(
            {
                str(Path(r[1]).parent)
                for r in sources
                if isinstance(r, list) and len(r) > 1 and isinstance(r[1], str)
            }
        )
        print("  原始影像来源目录（记录值）：" + ("；".join(text(p) for p in roots) or "未记录"))
    print("  缓存总数不等于本次评估数；具体名单以本报告为准。")


def prediction_source(job, item):
    output = item["output"]
    if item["name"] == "test":
        return output / "test/predictions/prediction-report.json"
    if item["name"] == "predict":
        return output / "predict/prediction-report.json"
    if item["name"] == "demo":
        return output / "demo/predictions/prediction-report.json"
    source = step_arguments(item).get("predictions")
    if not isinstance(source, str) or not source:
        return None
    folder = Path(source).expanduser()
    if not folder.is_absolute():
        folder = job / folder
    # Remap only sources owned by this workflow, not another task's same-named step.
    for producer in rows(read(job / "plan.json").get("steps")):
        raw, kind = producer.get("output"), producer.get("name")
        if isinstance(raw, str) and kind in {"test", "predict"}:
            suffix = "test/predictions" if kind == "test" else "predict"
            if Path(raw) / suffix == folder:
                local = job / "steps" / Path(raw).name / suffix / "prediction-report.json"
                if local.is_file():
                    return local
    return folder / "prediction-report.json"


def evaluation_mode(job, item):
    path = prediction_source(job, item)
    report = read(path) if path else {}
    mode = report.get("model_mode", report.get("mode"))
    if report.get("software_only") or mode in ("smoke", "preflight"):
        print("  模型来源：短测/预检，仅用于软件验证；以下分数不能当作正式实验结果。")
    elif mode == "train":
        print("  模型来源模式：正式训练（train）。")
    else:
        print("  模型用途未记录或来源清单不可读，无法确认是否为短测结果。")
    args = step_arguments(item)
    if item["name"] == "evaluate" and path:
        for producer in steps_for(job):
            if producer["name"] in {"test", "predict"} and prediction_source(job, producer) == path:
                args = {**step_arguments(producer), **args}
                break
    for key, label in (("prepared", "数据/真值缓存"), ("models", "分割模型来源")):
        print(f"  {label}：{text(args.get(key) or '未记录')}")
    cache_details(args.get("prepared"))
    print("  预测来源清单：" + text(path or "未记录"))
    if report.get("models_sha256"):
        print("  模型清单摘要（历史记录，未重新核验）：" + text(report["models_sha256"]))
    plan = mapping(report.get("split_plan"))
    if plan:
        print(
            f"  划分来源：上述预测清单 split_plan；身份策略 {text(plan.get('id_policy', '未记录'))}；"
            f"开发范围 {text(plan.get('development_scope', '未记录'))}"
        )
        for fold in rows(plan.get("folds")):
            cases = mapping(fold.get("cases"))
            values = [
                str(len(cases[p])) if isinstance(cases.get(p), list) else "未记录"
                for p in ("train", "val", "test")
            ]
            print(f"  划分记录 fold_{text(fold.get('index', '?'))}：训练/验证/测试 {'/'.join(values)} 例")
        print("  实际评估范围以下方报告为准；未重新验证划分或患者身份。")


def metric_table(saved, cases):
    for code, label in LABELS.items():
        print(
            f"  {label} ({code})： {number(mapping(saved.get('dice_by_class')).get(code))}"
            f" | 有效病例 {denominator(cases, code, saved.get('cases_evaluated'))}"
        )


def case_metrics(cases):
    case_list(cases)
    if not cases:
        return
    print("  逐病例已保存指标：病例 | 前景均值 | " + " | ".join(LABELS))
    for case in cases:
        values = [number(mapping(case.get("dice")).get(code)) for code in LABELS]
        print(
            f"    {text(case.get('case_id', '未记录'))} | {number(case.get('mean_foreground_dice'))} | "
            + " | ".join(values)
        )


def evaluation(path):
    report = read(path, warn=True)
    if not report:
        return
    print(
        f"  评估记录：{state(report.get('status'))} | 集合：{text(report.get('split', '未记录'))}"
        f" | 病例：{text(report.get('cases_evaluated', '未记录'))}"
    )
    if report.get("status") not in {"passed", "complete"}:
        print("  此评估尚未完整通过，以下仅为已有记录。")
    scope(report)
    cases = rows(report.get("cases"))
    overall, derived = summary(report, cases, LABELS)
    print("  整体汇总（本评估报告范围）")
    print("  平均前景 Dice：" + number(overall.get("mean_case_foreground_dice"), percent=True))
    print("  Dice 为分割重叠度，越接近 1 越好；这是按病例平均的前景指标，不是诊断准确率。")
    print("  汇总口径：先取每例前景均值，再对有效病例平均；不对各折均值等权平均。")
    if derived:
        print("  （旧版清单：缺失汇总字段按已保存的逐病例数值补算，不重新评估影像）")
    metric_table(overall, cases)
    print("  — 表示未记录/不适用，不能据此认定结构不存在；当前算法双方都缺失时为 null。")
    print("  有效病例分母仅在明细完整且该字段均已记录时显示；ignore=255 不参与当前 Dice。")
    details = fold_details(report, LABELS)
    if details:
        print("\n  每折分割明细（保留原折摘要；逐病例仅按明确的 OOF 记录归属）")
    for name, fold, members, recovered in details:
        if "cases_evaluated" not in fold and "mean_case_foreground_dice" not in fold and not members:
            print(f"  {name}：未记录，不能据此认定该折未运行。")
            continue
        print(
            f"\n  {name}：{count(fold.get('cases_evaluated'))} 例；"
            f"平均前景 Dice {number(fold.get('mean_case_foreground_dice'))}"
        )
        if recovered:
            print("  缺失汇总由本折已保存的逐病例数值补算。")
        if members and len(members) != fold.get("cases_evaluated"):
            print(f"  注意：已保存明细 {len(members)} 例，与登记总数不一致；明细不代表完整结果。")
        metric_table(fold, members)
        case_metrics(members)
    other = [r for r in cases if fold_index(r) is None] if details else cases
    if other or not details:
        print("\n  未按 OOF 折归属的病例（单次划分/集成/归属未记录，以逐例协议为准）")
        case_metrics(other)
    if report.get("space") == "prepared_grid":
        print("  评估网格：预处理网格。")
    if report.get("error"):
        print("  原错误：" + text(report["error"]))
    print("  评估文件：" + text(path))


def prediction(path):
    report = read(path, warn=True)
    if not report:
        return
    cases = rows(report.get("cases"))
    print(f"  预测记录：{state(report.get('status'))} | 清单记录已保存 {len(cases)} 例")
    scope(report, prediction_only=True)
    print("  此部分只列预测产物，不计算 Dice；分割评分见测试/独立评估。")
    case_list(cases)
    print("  预测目录：" + text(path.parent))
    print("  预测清单：" + text(path))
    missing = 0
    for case in cases:
        value = case.get("prediction")
        if isinstance(value, str):
            target = (path.parent / value).resolve()
            if not target.is_relative_to(path.parent.resolve()) or not target.is_file():
                missing += 1
    if missing:
        print(f"  注意：清单中 {missing} 例的最终预测文件目前不可访问。")
    if report.get("status") == "pending_cpu":
        print("  中间计算已结束，最终影像仍需 CPU 后处理；不能当作最终预测完成。")
    if report.get("software_only") or report.get("model_mode", report.get("mode")) in ("smoke", "preflight"):
        print("  此记录来自短测/预检，仅用于软件验证。")


def training(output, name):
    run = output / "train"
    summary = read(run / "report.json")
    models = read(run / "models.json")
    if not summary and not models:
        print("  尚无训练结果清单；可用下方训练明细命令查看已有 epoch/loss。")
        return
    mode = models.get("mode", summary.get("mode"))
    if name in {"smoke", "preflight"} or mode in {"smoke", "preflight"}:
        print("  短测/预检结果，仅用于软件验证。")
    if models.get("format") == "imagechd7-threefold-v1" or summary.get("task") == "chd_threefold":
        folds = mapping(summary.get("folds"))
        complete = sum(mapping(v).get("status") == "complete" for v in folds.values())
        print(f"  三折训练：记录已完成 {complete}/3 折")
        for key in inspection.FOLDS:
            record = mapping(folds.get(key))
            print(
                f"  {key}：{state(record.get('status'))}；"
                f"训练/验证/测试病例 {record.get('train_cases', '?')}/"
                f"{record.get('val_cases', '?')}/{record.get('test_cases', '?')}"
            )
    else:
        stages = mapping(models.get("stages"))
        print(f"  单次划分模型清单：{len(stages)}/6 个阶段有记录")
        for key in inspection.MODELS:
            record = mapping(stages.get(key))
            if record:
                print(
                    f"  {key}：已完成 {text(record.get('epochs_completed', '?'))} epoch；"
                    f"最佳验证 loss {number(record.get('best_val_loss'))}"
                )
    print("  模型目录：" + text(run))
    print("  训练完成不代表测试/评估已完成；loss 不等于 Dice。")


def diagnosis(path):
    from .result_decisions import show_metrics

    show_metrics(path)


def diagnosis_sources(item):
    args = step_arguments(item)
    for key, label in (
        ("features", "分割特征来源"),
        ("classifier", "决策树来源"),
        ("diagnosis_labels", "诊断真值来源"),
        ("diagnoses", "诊断判断来源"),
        ("evidence", "人工证据输入"),
    ):
        if args.get(key):
            print(f"  {label}：{text(args[key])}")
    print("  来源按历史任务记录展示，未重新核验文件摘要。")


def show_results(job, step="auto"):
    job = Path(job).expanduser().resolve()
    if not job.is_dir():
        raise ValueError("结果目录不存在：" + str(job))
    current = read(job / "job_state.json", warn=(job / "job_state.json").exists())
    steps = steps_for(job)
    selected = steps if step == "auto" else [s for s in steps if step in {s["name"], s["output"].name}]
    if step != "auto" and not selected:
        raise ValueError("任务没有此步骤：" + step)
    print(f"\n{'=' * 56}\n结果概览：{title(job)}")
    print(
        f"流程记录：{state(current.get('status'))} | 完成步骤 "
        f"{sum(s.get('status') in {'passed', 'reused'} for s in steps)}/{len(steps)}"
    )
    print("任务目录：" + text(job))
    plan = read(job / "plan.json")
    for key, label in (("input_job", "预处理输入任务"), ("rerun_of", "重跑来源任务")):
        source = mapping(plan.get(key))
        if source:
            print(label + "：" + text(source.get("name")) + " | " + text(source.get("directory")))
    if current.get("ended_at"):
        print("结束时间：" + text(current["ended_at"]))
    for receipt in sorted((job / "uploads").glob("*.json")):
        record = read(receipt)
        url = record.get("url", "")
        if isinstance(url, str) and url.startswith("https://github.com/tianwuasd/project/releases/"):
            print("已上传结果与权重：" + text(url))
    print("读取已有结果，不重新训练或评估；文件哈希未重新校验。")
    if not selected:
        print("未找到可读取的步骤记录；请核对原 workflow 目录。")
    for item in selected:
        name, output = item["name"], item["output"]
        print(f"\n{STEPS.get(name, text(name))} | {state(item.get('status'))}")
        if name in inspection.MODEL_STEPS:
            training(output, name)
        elif name in {"test", "evaluate"}:
            path = output / (
                "test/evaluation/evaluation.json" if name == "test" else "evaluate/evaluation.json"
            )
            if path.is_file():
                evaluation_mode(job, item)
                evaluation(path)
            else:
                print("  尚无可读取的评估文件，暂不能给出 Dice；已训练完成也不会自动产生此指标。")
                print("  预期评估文件：" + text(path))
            if name == "test" and (output / "test/predictions/prediction-report.json").is_file():
                prediction(output / "test/predictions/prediction-report.json")
        elif name == "predict":
            evaluation_mode(job, item)
            path = output / "predict/prediction-report.json"
            if path.is_file():
                prediction(path)
            else:
                print("  尚无最终预测清单；请查看该步骤日志。")
        elif name == "demo":
            print("  数据类型：程序生成的合成数据；不代表真实 CT 分割精度。")
            evaluation_mode(job, item)
            path = output / "demo/evaluation/evaluation.json"
            if path.is_file():
                evaluation(path)
            else:
                print("  尚无合成分割评估清单。")
        elif name == "diagnosis-evaluate":
            diagnosis_sources(item)
            path = output / "diagnosis-evaluate/diagnosis-metrics.json"
            if path.is_file():
                diagnosis(path)
            else:
                print("  尚无诊断评估文件。")
        elif name in {"diagnosis-train", "diagnose", "diagnosis-demo"}:
            from .result_decisions import show_decisions, show_training

            diagnosis_sources(item)
            if name == "diagnosis-train":
                show_training(output / "diagnosis-train/trainability.json")
            elif name == "diagnose":
                show_decisions(output / "diagnose/diagnosis.json")
            else:
                print("  数据类型：合成数据，仅验证诊断软件流程，不代表真实诊断性能。")
                show_training(output / "diagnosis-demo/training/trainability.json")
                show_decisions(output / "diagnosis-demo/prediction/diagnosis.json")
                diagnosis(output / "diagnosis-demo/evaluation/diagnosis-metrics.json")
        else:
            print("  输出目录：" + text(output))
        if item.get("status") in {"failed", "interrupted"}:
            print("  排障日志：" + text(output / "server.log"))
    commands(job, steps)
