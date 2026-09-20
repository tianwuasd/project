"""汇总成功/失败短测并估算正式预算，绝不从短测 Dice 推断方法效果。"""
from pathlib import Path
from statistics import mean
from data_pipeline import ROOT, load_json, save_json


def main():
    # 只接受 suite.json 中完整成功的四个运行，不能挑选单个较好结果。
    suites = sorted((ROOT / "09_reports").glob("smoke_suite_*/suite.json"))
    successful = [p for p in suites if len(load_json(p)) == 4 and all(r["status"] == "passed" for r in load_json(p))]
    if not successful:
        raise RuntimeError("尚没有完整通过的四组短测")
    suite = successful[-1]
    runs = {}
    for entry in load_json(suite):
        # 每个日志中写入唯一 run.json 路径，避免误用另一次手动运行。
        text = Path(entry["log"]).read_text(encoding="utf-8")
        report_lines = [s[len("REPORT: "):].strip() for s in text.splitlines() if s.startswith("REPORT: ")]
        if len(report_lines) != 1:
            raise ValueError(f"无法唯一定位运行报告: {entry['log']}")
        r = load_json(report_lines[0])
        assert r["status"] == "passed" and r["smoke_only"]
        runs[f"{entry['fold']}_{entry['method']}"] = r
    rows, total_hours = [], 0
    for fold, epochs, folds in [("ct_holdG", 250, 3), ("mr_holdE", 300, 2)]:
        a, b = runs[fold+"_B0"], runs[fold+"_B1"]
        assert a["initial_weight_sha256"] == b["initial_weight_sha256"], "B0/B1 初始化不同"
        assert a["plan_sha256"] == b["plan_sha256"], "B0/B1 规划不同"
        for method in ["B0", "B1"]:
            r = runs[fold+"_"+method]
            # 每轮250训练步和50开发验证步，乘以固定epoch；尚不含全部原图推理。
            average_step = mean(r["train_seconds_excluding_first_two"])
            hours = epochs * (250*average_step + 50*r["validation_step_seconds"])/3600
            total_hours += hours * folds
            rows.append({"fold": fold, "method": method, "train_step_seconds": average_step, "validation_step_seconds": r["validation_step_seconds"], "peak_allocated_gib": r["peak_allocated_gib"], "one_training_hours_extrapolated": hours, "outer_folds_assumed": folds})
    result = {"status": "passed", "suite": str(suite), "same_initialization_and_plans": True, "rows": rows, "ten_runs_core_hours_extrapolated": total_hours, "limitation": "Only 10 timed steady-state steps per condition and 2 validation steps; other folds have different plans. Excludes preparation, full-volume inference, checkpoint IO, thermal/desktop load changes. Not a guaranteed runtime or performance result."}
    save_json(ROOT / "09_reports/smoke_summary.json", result)
    lines = ["# 短测结果与正式训练预算", "", "## Material Passport", "", "Origin: academic_tools experiment execution / project1；Status: 工程短测，非效果验证。", "", "四组短测通过，同模态两组的初始权重哈希和计划哈希相同。每次12个训练步骤，加1个恢复后的步骤；每组2个开发验证步骤。", "", "| 方向 | 组别 | 每训练步秒数 | 峰值显存 GiB | 单次正式训练估计小时 |", "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['fold']} | {r['method']} | {r['train_step_seconds']:.3f} | {r['peak_allocated_gib']:.2f} | {r['one_training_hours_extrapolated']:.1f} |")
    lines += ["", f"按CT 3方向、MRI 2方向外推，10次训练核心计算合计约 **{total_hours:.1f} 小时（{total_hours/24:.1f} 天连续运行）**。", "", "这是小样本吞吐量外推，不是保证。尚未包含全部预处理、完整体积推理、检查点写入和电脑其他负载；其余方向的网络规划可能不同。正式开启前需确认可接受的 GPU 时间，并将实际完成步数完整记录。", "", "没有用短测分数比较增强收益，没有读取外层目标中心进行选模。CT、MRI 各保存一个来源开发病例的完整三维预测并恢复原始网格；当前预测未训练充分，仅验证文件流程。", "", "P3/P4 正式训练尚未开始，不能报告研究结论。"]
    (ROOT / "09_reports/短测与预算.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
