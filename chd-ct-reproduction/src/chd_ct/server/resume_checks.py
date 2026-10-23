"""User-selected resume checks, scoped to direct dependencies in simple mode."""

from pathlib import Path

from .prompts import ask

INPUTS = {
    "environment": (),
    "preprocess": ("dataset", "split_file", "spacing_excel"),
    "train": ("prepared", "config"),
    "test": ("prepared", "models"),
    "predict": ("prepared", "models"),
    "evaluate": ("prepared", "predictions"),
    "diagnosis-labels": ("diagnosis_source",),
    "diagnosis-features": ("prepared", "predictions"),
    "diagnosis-train": ("features", "diagnosis_labels"),
    "diagnose": ("features", "classifier", "rules", "evidence"),
    "diagnosis-evaluate": ("features", "diagnosis_labels", "diagnoses"),
    "diagnosis-demo": (),
    "demo": (),
}
ARTIFACT_FILES = {
    "features": "features.json",
    "diagnosis_labels": "labels.json",
    "classifier": "classifier.json",
    "diagnoses": "diagnosis.json",
}
NAMES = {"full": "全面检查", "simple": "简单检查", "none": "不做检查"}


def choose_mode(value, interactive):
    if value is not None:
        if value not in NAMES:
            raise ValueError("续跑检查只能为 full/simple/none")
        return value
    if not interactive:
        return "simple"
    print(
        "续跑前检查：\n  1 全面检查：所有已完成步骤的输入/输出，完整读取内容。\n"
        "  2 简单检查：前置结果及直接依赖；小文件核验内容，大文件只查状态（默认）。\n"
        "  3 不做检查：跳过本次产物校验；仍保留任务运行锁和各步骤自身检查。"
    )
    choices = {"1": "full", "2": "simple", "3": "none", "": "simple", **{k: k for k in NAMES}}
    while True:
        selected = ask("检查方式 [2]（b 返回，q 退出）：").lower()
        if selected in choices:
            return choices[selected]
        print("请输入 1、2 或 3。")


def pending_inputs(plan):
    """Only existing direct inputs; outputs of earlier pending steps are future inputs."""
    roots, future = {}, []
    for step in plan["steps"]:
        if step.get("reuse"):
            continue
        arguments = step["arguments"]
        task = arguments["task"]
        if task not in INPUTS:
            raise ValueError("未知步骤不能推断简单检查范围，请选择全面检查：" + task)
        for key in INPUTS[task]:
            if not arguments.get(key) or (key == "classifier" and arguments.get("method") == "rules"):
                continue
            path = Path(arguments[key]).expanduser().resolve()
            if any(path == root or path.is_relative_to(root) for root in future):
                continue
            if key in ARTIFACT_FILES and path.is_dir():
                path /= ARTIFACT_FILES[key]
            roots.setdefault(str(path), []).append(step["name"] + "/" + key)
        produced = arguments.get("prepared") if task == "preprocess" else str(Path(step["output"]) / task)
        if produced:
            future.append(Path(produced).expanduser().resolve())
    return roots


def simple_records(plan):
    needed = pending_inputs(plan)
    snapshots = [s["reuse"].get("snapshot", {}) for s in plan["steps"] if s.get("reuse")]
    snapshots.append({"inputs": plan.get("frozen_files", {})})
    records, unrecorded = [], []
    for root, users in needed.items():
        selected = {}
        prefix = Path(root)
        for snapshot in snapshots:
            for group in ("inputs", "outputs"):
                for name, digest in snapshot.get(group, {}).items():
                    path = Path(name)
                    if path == prefix or path.is_relative_to(prefix):
                        if name in selected and selected[name] != digest:
                            raise ValueError("已完成步骤校验记录冲突：" + name)
                        selected[name] = digest
        record = dict(inputs=selected, outputs={}, output_roots=[])
        if selected:
            record["input_roots"] = [root]
        else:
            # Never manufacture a new trusted digest for a legacy/external dependency.
            record["unrecorded_roots"] = [root]
            unrecorded.append(root)
        records.append((", ".join(users), record))
    return records, unrecorded


def check_plan(plan, mode):
    from .verification import verify_snapshots

    print("本次续跑：" + NAMES[mode] + "。", flush=True)
    if mode == "none":
        print("已跳过历史输入/产物校验；不代表文件完整，成功步骤仍保持复用。", flush=True)
        return dict(mode=mode, status="skipped", files=0, bytes=0)
    unrecorded = []
    if mode == "full":
        records = []
        for step in plan["steps"]:
            if not step.get("reuse"):
                continue
            record = step["reuse"].get("snapshot")
            if not record and step["arguments"]["task"] != "environment":
                raise ValueError("已完成步骤缺少校验记录，无法全面检查：" + step["name"])
            if record:
                records.append((step["name"], record))
        if plan.get("frozen_files"):
            records.append(("冻结配置", dict(inputs=plan["frozen_files"], outputs={})))
    else:
        records, unrecorded = simple_records(plan)
        print("仅检查后续步骤直接依赖的前置结果；不追溯读取整条历史链路。", flush=True)
        if unrecorded:
            print(
                "以下依赖没有历史摘要，仅检查存在与本次状态稳定：\n  " + "\n  ".join(unrecorded), flush=True
            )
    result = verify_snapshots(records, mode=mode)
    return dict(
        result,
        mode=mode,
        status="passed",
        checked_dependencies=[name for name, _ in records],
        unrecorded_roots=unrecorded,
    )
