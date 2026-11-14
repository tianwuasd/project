# Model Ordered Supply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 两模型按当前批次公平准备数据，完成 CPU 就绪检查后进入设备通信。

**Architecture:** 使用父进程创建的共享 CPU 控制对象和有序病例构建对象。Q 保持同步采样；MedSAM2 包装官方 sampler、数据集与迭代器。

**Tech Stack:** Python、multiprocessing、PyTorch、Gloo、官方 MedSAM2。

**Spec:** ../specs/2026-11-15-model-ordered-supply-design.md

## Global Constraints

- 不改变 Q r2、官方 MedSAM2 序列、随机调用与全局 batch。
- 最多两例真实病例、每阶段两次更新；CPU 验证、禁止登录服务器。
- 使用批准的 t17 目录并清理；报告总量不超过 10 MiB。
- 保留助手2分支与既有恢复边界。

## Review Focus

- 不均分 Q 份额时当前样本公平获得构建名额。
- MedSAM2 多 worker 的后续任务不能阻塞当前批次。
- rank0 验证和保存期间其他 rank 不提前提交 collective。
- 官方 sampler、增强随机数及恢复 worker 布局保持一致。
- CUDA 同步后释放缓冲；本机只做 CPU，CUDA 部分须明确未覆盖。

### Task 1: 供数协议与模型接入

**Files:** 新建 models/supply.py、models/medsam2/supply.py；修改 models/execution.py、models/workflow.py、models/medsam2/data.py、models/medsam2/worker.py；新建 tests/verify_model_supply_real.py。

**Interfaces:** CpuReady.wait(rank, phase, device=None) 检查阶段并等待；OrderedSupply.start(rank, phase, quotas, device=None)、build(rank, batch, position)、finish(rank, phase, device=None) 管理当前批次；MedSAM2 包装器保留官方索引与数据。

- [x] 写真实多进程供数验证，缺少协议时应失败。
- [x] 实现 CPU 就绪与有序构建，在 Q 三阶段和官方 MedSAM2 路径使用。
- [x] 运行一例真实 Q 两进程三阶段训练及恢复，每阶段合计两次更新。
- [x] 运行官方真实 DataLoader 多 worker 的两个批次，核对增强结果、批次顺序、队列上限及背压，零次训练更新。
- [ ] 助手1提交确认后整合适用公共改动，更新兼容清单并运行受影响接口。
- [ ] 完成一次独立整体审阅，修复重要发现；记录未覆盖范围与清理结果并提交。
