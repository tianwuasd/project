# STU-Net-S Prediction Implementation Plan

> **For agentic workers:** 使用当前会话执行，各任务先验证失败再实现，最终安排独立分支审查。

**Goal:** 复用已有训练与预处理，为留出测试集提供可恢复、有资源预算的整幅七结构预测。

**Architecture:** prediction 包负责模型、输入、滑窗、导出和任务控制；acceleration/prediction.py 使用既有 V2 资源接口。CLI 和 smoke 调用同一实现，计算进程使用冻结代码。

**Tech Stack:** Python、PyTorch、nnU-Net 2.8.1、NumPy、SciPy、NiBabel、Blosc2、psutil。

**Spec:** ../specs/2026-11-08-prediction-design.md

## Global Constraints

- 仅修改 chd-ct-panel-v2；不修改训练算法、数据、旧检查点或签名。
- test 默认；max_gpus=2；tile_step=0.5；TTA 显式配置；CPU/RAM80%；GPU 保留 max(4GiB,10%)。
- 参数来源、权限范围、完成标记及结果 hash 必须验证。无 mock，使用真实模型/库/子进程。

## Review Focus

- 多折/未知病例出现训练泄漏：角色交叉检查和无标签新病例测试。
- 旧来源损坏、checkpoint 指针变化：冻结模型身份、完整源核验、恢复拒绝差异。
- 显存不足及 GPU 被其他进程占用：有限调整、重新排队、只管理本进程。
- CPU 预取与导出同时驻留：整体内存预算限制，保留等待原因。
- 原始头 spacing 不可信、ignore 和空预测：参考 CT 几何一致、指标明确缺失语义。

### Task 1: 来源和输入

Files: prediction/model.py、inputs.py、tests/test_prediction_contracts.py。

Interfaces: inspect_training(run, checkpoint, smoke, allow_intermediate) -> dict；select_cases(prepared, models, selected, split) -> list；prepare_case(record, model, destination, threads) -> dict。

- [x] 写角色泄漏、错误来源、真实旧检查点和图像-only预处理测试；确认未实现时失败。
- [x] 实现源代码/模型校验、预测计划适配、历史角色检查、缓存复用和独立新输入。
- [x] 运行 test_prediction_contracts.py，要求全部通过。

### Task 2: 滑窗和评价

Files: prediction/infer.py、export.py、tests/test_prediction_numerics.py。

Interfaces: sliding_logits(network, image, patch, device, batch, tta, step, progress) -> logits/report；export_case(record, prepared, model, logits, output, evaluate) -> dict。

- [x] 使用真实 STU-Net、边缘 padding、批量等价、全 affine HD95、空预测测试验证失败。
- [x] 复用官方 Gaussian/滑窗定位和导出函数；实现 FP32 融合及多折共同网格概率平均。
- [x] 运行数值测试并与单窗口直接 forward 对照。

### Task 3: 调度和恢复

Files: acceleration/prediction.py、prediction/run.py、worker.py、tests/test_prediction_runtime.py。

Interfaces: run_prediction(options) -> summary；worker(job)；request_stop(output)。

- [x] 添加真实进程、已完成结果恢复、输出损坏、后台退出独立性检查。
- [x] 实现显存探测/CPU预算、驻留worker、受控下一例预取、独立导出、心跳、停止和恢复。
- [x] 使用真实 CPU 子进程跑通任务；CUDA/多卡按实际硬件分别列明。

### Task 4: CLI、短测和交付

Files: cli.py、smoke.py、config.py、configs/prediction/stunet-s.json、docs/prediction-usage.md、tests/test_prediction_cli.py。

- [x] CLI 支持 predict、模型目录发现、test/all folds、背景运行、check-only、resume、停止和新病例。
- [x] 真模型/真实 CT 完整预测；同参数串行和优化路径重复比较并保存证据。
- [x] 检查训练回归、文档、源文件摘要，独立审查后修复重要问题。
- [x] 更新发布清单并在本地分支提交；不合并 main、不上传数据或模型。
