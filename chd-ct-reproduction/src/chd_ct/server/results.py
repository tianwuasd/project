"""Human-readable existing results, using only the standard library and read-only I/O."""

import json
import math
import shlex
import unicodedata
from pathlib import Path

from . import inspection

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
    "test": "测试",
    "predict": "预测",
    "evaluate": "分割评估",
    "diagnosis-evaluate": "诊断评估",
}
PROTOCOLS = {
    "out_of_fold": "三折独立测试（OOF）",
    "threefold_ensemble": "三折集成",
    "single_split": "单次划分",
}


def text(value):
    """Keep persisted names/errors from injecting terminal control sequences."""
    return "".join(c if not unicodedata.category(c).startswith("C") else " " for c in str(value))


def read(path, warn=False, kind=dict):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("结果清单超过 16 MiB，请直接查看原文件")
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, kind):
            raise ValueError("结果清单格式不正确")
        return value
    except (OSError, ValueError) as error:
        if warn:
            print(f"  无法读取 {text(path)}：{text(error)}")
        return kind()


def mapping(value):
    return value if isinstance(value, dict) else {}


def rows(value):
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def number(value, percent=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—（未记录/不适用）"
    return f"{value:.4f}" + (f"（{value * 100:.2f}%）" if percent else "")


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


def evaluation_mode(job, item):
    output = item["output"]
    if item["name"] == "test":
        path = output / "test/predictions/prediction-report.json"
    else:
        settings = read(output / "server-settings.json")
        source = mapping(item.get("arguments")).get("predictions") or settings.get("predictions")
        path = None
        if isinstance(source, str) and source:
            folder = Path(source).expanduser()
            path = folder / "prediction-report.json"
            # A copied workflow may retain its old absolute input path in the plan.
            if "steps" in folder.parts:
                relative = folder.parts[folder.parts.index("steps") :]
                local = job.joinpath(*relative) / "prediction-report.json"
                if local.is_file():
                    path = local
    report = read(path) if path else {}
    mode = report.get("model_mode", report.get("mode"))
    if report.get("software_only") or mode in {"smoke", "preflight"}:
        print("  模型来源：短测/预检，仅用于软件验证；以下分数不能当作正式实验结果。")
    elif mode == "train":
        print("  模型来源模式：正式训练（train）。")
    else:
        print("  模型用途未记录或来源清单不可读，无法确认是否为短测结果。")


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
    protocols = report.get("protocols", [])
    if isinstance(protocols, list) and protocols:
        print("  推理方式：" + "、".join(PROTOCOLS.get(p, text(p)) for p in protocols if isinstance(p, str)))
    print("  平均前景 Dice：" + number(report.get("mean_case_foreground_dice"), percent=True))
    print("  Dice 为分割重叠度，越接近 1 越好；这是按病例平均的前景指标，不是诊断准确率。")
    print("  结构                 Dice")
    by_class = mapping(report.get("dice_by_class"))
    if not by_class and rows(report.get("cases")):
        # Old releases stored case Dice without a class-level summary.
        for code in LABELS:
            values = [mapping(r.get("dice")).get(code) for r in rows(report["cases"])]
            values = [
                v
                for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            ]
            by_class[code] = sum(values) / len(values) if values else None
        print("  （旧版清单：按已保存的逐病例 Dice 汇总，不重新评估影像）")
    for code, label in LABELS.items():
        print(f"  {label} ({code})： {number(by_class.get(code))}")
    print("  — 表示未记录或不适用；真值和预测都没有该结构时不计入其平均值。")
    for name, fold in mapping(report.get("folds")).items():
        if isinstance(fold, dict):
            print(
                f"  {text(name)}：{text(fold.get('cases_evaluated', '?'))} 例；"
                f"平均前景 Dice {number(fold.get('mean_case_foreground_dice'))}"
            )
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
    if report.get("software_only") or report.get("model_mode", report.get("mode")) in {"smoke", "preflight"}:
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
    report = read(path, warn=True)
    if not report:
        return
    print(
        f"  诊断评估：{state(report.get('status'))} | 集合：{text(report.get('split', '未记录'))}"
        f" | 病例：{text(report.get('cases_evaluated', '未记录'))}"
    )
    for key, label in (("classifier_micro", "分类器"), ("rules_micro", "规则")):
        metrics = mapping(report.get(key))
        print(
            f"  {label}：准确率 {number(metrics.get('accuracy'), True)}；"
            f"覆盖率 {number(metrics.get('coverage'), True)}；"
            f"F1 {number(metrics.get('f1'))}"
        )
    print("  准确率仅统计已作出判断且真值已知的患者-疾病对；覆盖率反映弃权情况。")
    if report.get("software_only"):
        print("  此记录仅用于软件验证。")
    print("  诊断评估文件：" + text(path))


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
            path = output / "predict/prediction-report.json"
            if path.is_file():
                prediction(path)
            else:
                print("  尚无最终预测清单；请查看该步骤日志。")
        elif name == "diagnosis-evaluate":
            path = output / "diagnosis-evaluate/diagnosis-metrics.json"
            if path.is_file():
                diagnosis(path)
            else:
                print("  尚无诊断评估文件。")
        else:
            print("  输出目录：" + text(output))
        if item.get("status") in {"failed", "interrupted"}:
            print("  排障日志：" + text(output / "server.log"))
    commands(job, steps)
