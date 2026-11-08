# Dataset1282 Prediction Implementation Plan

**Goal:** 使用训练阶段冻结的预处理参数与模型，在原名单的三组留出 CT 上分别预测和评价。
**Architecture:** prediction/source.py 核对并冻结来源；prediction/workflow.py 执行逐病例可恢复预测；prediction/cli.py 调用既有隔离与资源监督；scripts/xu_predict.py 提供入口。
**Tech Stack:** 复用 PyTorch、NumPy、SciPy、Blosc2、nibabel 和 psutil。
**Spec:** ../specs/2026-11-08-prediction-design.md

## 共同约束

- 只读原件和训练结果，独立应用冻结计划，source_test/target_primary/target_secondary 保持原分组，原生局部毫米网格。
- 原训练源码保持不变；完整模型要求对应完成收据。
- 单 GPU、CPU 预取、排除 0/1 与外部占用；资源恢复保持模型和精度。

## 实施和验证

- [x] source.freeze_source(training, cache, output, mode)：核验原始 run 身份、阶段收据、模型摘要和 ROI；复制所需权重。测试未完成训练、损坏权重与源文件不变。
- [x] heldout.prepare_case(output, name)：核对原 cohort/政策/Excel/原件摘要，调用官方预处理器应用训练计划且不读取标签。测试冻结计划、三组角色互斥、排除和重复扫描、损坏输入。
- [x] workflow.predict(output, device)：复用原预测器、投票与后处理，逐病例摘要提交。测试真实短训练、几何、重复恢复和损坏产物拒绝。
- [x] cli.main(argv)：独立 run/status/worker，复用隔离启动、预算与资源恢复。测试真实 CPU 子进程入口及现有隔离检查。
- [x] evaluate：预测文件完整提交后读取标签，按三组及设备分别汇总指标、有效分母和拒绝数量。

发布目标：使用说明、清单和验收代码随 `feat/dataset1282-prediction` 提交，main 由单独的合并流程管理。
