"""依据真实报告更新项目状态；缺少证据时直接失败，不写虚构完成状态。"""
import importlib.metadata as metadata
import sys
from data_pipeline import ROOT, load_json, save_json


def main():
    summary = load_json(ROOT / "09_reports/smoke_summary.json")
    assert summary["status"] == "passed" and summary["same_initialization_and_plans"]
    probe = load_json(ROOT / "09_reports/gpu_probe.json")
    assert probe["status"] == "passed"
    decisions = load_json(ROOT / "00_admin/decisions.json")
    decisions["confirmed"]["protocol_approval"] = "2026-09-19 用户：好的，按照研究方案进行"
    decisions["confirmed"]["initialization"] = "按获准方案的建议，B0/B1 统一随机初始化；短测哈希相同"
    decisions["pending"] = {"full_training_budget_hours": None, "remaining_three_folds_preparation": True}
    decisions["training_started"] = True
    decisions["training_scope"] = "仅四组12步工程短测及各1步恢复验证；正式研究训练未开始"
    save_json(ROOT / "00_admin/decisions.json", decisions)
    save_json(ROOT / "00_admin/stage_status.json", {"updated": "2026-09-19", "stage": "preflight_and_smoke_complete_for_two_directions", "frozen_outer_splits": 5, "prepared_outer_splits": ["ct_holdG", "mr_holdE"], "successful_smoke_conditions": 4, "formal_training_started": False, "pending": ["full_gpu_budget_confirmation", "remaining_three_directions_preparation", "formal_training_and_evaluation"], "independent_peer_review_performed": False, "formal_ars_schema_validated": False, "no_effectiveness_results_yet": True})
    packages = {p: metadata.version(p) for p in ["numpy", "scipy", "nibabel", "matplotlib", "pandas", "PyYAML", "torch", "torchvision", "nnunetv2", "SimpleITK", "batchgeneratorsv2"]}
    save_json(ROOT / "06_configs/environment_status.json", {"checked_on": "2026-09-19", "environment_name": "whole_heart_project1", "python": sys.version, "executable": sys.executable, "packages": packages, "pytorch_installed_by_this_project": True, "nnunet_installed_by_this_project": True, "gpu_training_verified": True, "formal_training_started": False, "gpu_evidence": "09_reports/gpu_probe.json and smoke_summary.json", "blas": "OpenBLAS", "runtime_conflict_bypassed": False})
    root_readme = ROOT / "README.md"
    text = root_readme.read_text(encoding="utf-8-sig")
    text = text.replace("当前阶段：研究方案 v0.1；尚未训练模型、尚无效果结果。", "当前阶段：研究方案已获准；5个外层划分已冻结，CT-holdG/MR-holdE 完成准备和四组工程短测。正式训练尚未开始，尚无方法效果结论。\n\n最新进度见 [短测与预算](09_reports/短测与预算.md)；复现环境见 [训练环境复现](06_configs/训练环境复现.md)。")
    text = text.replace("项目完整性检查工具；后续训练、预测、评价代码，未实现训练程序", "数据转换、来源专用规划、受控训练器、增强、指标及短测入口")
    root_readme.write_text(text, encoding="utf-8")
    documents = {
        "tests/README.md": "# 实现验证\n\n10 项自动检查通过，日志见 ../09_reports/tests.log。覆盖标签往返、未知标签拒绝、中心与病例隔离、增强不改标签、与固定参考代码一致、单调性、毫米距离、空掩膜，以及阻止官方训练器自动改用随机划分。\n\n另有真实 GPU 架构检查、四组真实病例短测、权重/优化器恢复后再训练一步、两个完整三维预测的原网格检查。它们验证工程流程，不验证跨中心效果。配对统计、批量目标中心评价、长期训练稳定性仍待后续。\n",
        "07_experiments/README.md": "# 实验记录\n\n当前只有 gpu_probe 与四个 smoke_* 工程短测，正式模型训练未开始。每个短测目录保存配置与文件哈希、来源划分、运行状态、日志、检查点；B0 还保存一个来源开发病例的完整预测。不要把这些仅13步的检查点当成正式模型。\n\n后续正式实验另建目录，不覆盖短测或失败记录。\n",
        "08_results/README.md": "# 研究结果\n\n当前没有正式研究指标或方法效果结论。工程短测预测保存在 ../07_experiments/smoke_*，预算报告在 ../09_reports。此目录以后保存外层目标中心的每例每类指标、汇总及配对分析，不能用短测分数或论文数字充当研究结果。\n",
        "09_reports/README.md": "# 阶段报告\n\n优先阅读 [短测与预算](短测与预算.md)。smoke_summary.json 为四组汇总；gpu_probe.json 为真实架构的合成输入显存检查；tests.log 为10项测试。两个 *_preparation.json 记录来源专用准备。原 ct_holdG_preparation.log 保留线程参数错误，修复续行记录为 ct_holdG_preparation_r1.log。\n\n结果只支持流程可运行，不能证明增强有效。\n",
        "06_configs/环境说明.md": "# 环境说明\n\n本机为 RTX 5060 Ti 8151 MiB，驱动591.44，Windows。独立 Conda 环境 whole_heart_project1，位置 C:/programming/anaconda/envs/whole_heart_project1。\n\n已安装并实测 PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128 和 nnU-Net 2.8.1。CT/MRI 真实病例均完成有限步数训练、开发集验证、检查点恢复与继续一步训练；GPU训练栈可运行。尚未验证长期稳定性或正式效果。\n\n基础 conda_explicit.txt 保留最初环境记录；当前训练环境应查看 conda_explicit_training.txt、pip_freeze_training.txt 和 environment_status.json。[环境重建说明](训练环境复现.md)解释安装顺序与 OpenBLAS 修复。\n\n使用 conda activate whole_heart_project1 激活。训练时只运行一个 GPU 任务。原始影像不修改，未上传外部服务，未修改系统代理。\n",
    }
    for relative, content in documents.items():
        (ROOT / relative).write_text(content, encoding="utf-8")
    with (ROOT / "00_admin/implementation_log.md").open("a", encoding="utf-8") as stream:
        stream.write("\nP1/P2 更新：两个先行方向的来源准备及四组短测已通过；两模态均验证完整三维预测恢复原网格。正式预算见短测与预算.md。其余三个方向仅冻结划分，未预处理；正式10次训练尚未开始。\n")
    print("Project records updated from verified reports.")


if __name__ == "__main__":
    main()
