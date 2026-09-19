# GitHub 归档说明

归档日期：2026-09-19。用户选择暂不继续正式训练，本次仅上传当前研究工作。

已包含方案、来源与 skill 记录、病例/隔离/划分清单、中文注释代码、环境锁定文件、四组短测的配置/日志/结果记录和进度图。短测只验证工程流程，不代表正式分割效果。

未提交原始影像、派生影像/预处理缓存、预测 NIfTI、约 2 GB 的短测权重。原件仍在本机。原始数据不在仓库中，克隆项目后需要自行准备数据并核对清单中的本机路径。

作者参考脚本沿用项目 .gitignore，不随仓库分发；来源 commit 记录在 `../02_materials/code_reference/source.json`。`tests/test_appearance.py` 的参考数值一致性测试需要从该 commit 获取 `trainers/bias_field_transform.py` 和 `trainers/bezier_dualnorm_transform.py`，放入本地 `02_materials/code_reference/`。另外两份参考文件是 `trainers/nnUNetTrainerLayer2BezierDualNorm.py` 与仓库 README。它们不是训练运行时依赖；当前报告中的10项检查是在本地参考文件齐备时完成的。

部分历史运行日志、清单和哈希记录包含本机绝对路径，作为原始运行记录保留。网页阅读优先使用项目 README 的相对链接；本机路径不是 GitHub 下载链接。

本次仅改变 GitHub 副本的日志忽略例外，以便归档可复核的小型日志；没有启动、续跑或修改本地训练。
