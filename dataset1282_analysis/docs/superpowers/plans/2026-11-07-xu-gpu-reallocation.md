# Xu GPU Reallocation Implementation Plan

> **For agentic workers:** Use native execution with a fresh final reviewer; user authorized implementation in this conversation.

**Goal:** 排除物理 GPU 0、1，先统一登记和放行，再在外部占用变化时有界重新分配并恢复。

**Architecture:** 保留现有分支调度及训练器。查询快照、进程放行、设备重新分配分开实现，阶段研究身份独立于设备成员。

**Tech Stack:** Python、psutil、PyTorch、CUDA driver 只读查询、NVIDIA CLI、unittest。

**Spec:** `docs/superpowers/specs/2026-11-07-xu-gpu-reallocation-design.md`

## Global Constraints

- 保留原始数据、预处理、模型、batch、loss、学习率和轮数。
- 默认排除物理 0、1；最多 3 次重新分配；等待沿用原参数。
- 所有修改通过文件编辑工具；新增测试使用真实功能，不使用 mock。

## Review Focus

- CUDA 可见序号重新排序后仍按物理 UUID 排除。
- 某个分支启动失败时，等待放行的另一个分支退出。
- 首次名单之外的本任务进程复查后正确识别，PID 复用不获授权。
- 改变卡数只改变运行记录，恢复仍验证完整研究合同。
- 阶段已完成或只有部分文件时，重分配分别保持产物或重新执行未完成步骤。

## Task 1: 查询与排除

Files: `xu_gpu_probe.py`, `xu_fleet.py`, `xu_pipeline.py`, `tests/test_xu_reallocation.py`.

- [x] 验证物理排除与可见序号重排，先观察测试失败。
- [x] 实现 `filter_physical_devices(devices, excluded, inventory)`，统一快照供分配使用。
- [x] 增加 CLI 排除参数和运行合同记录，相应测试通过。

## Task 2: 登记与放行

Files: `xu_dispatch.py`, `xu_runtime_guard.py`, `xu_pipeline.py`, `scripts/xu_worker.py`, `tests/fixtures/xu_launch_child.py`.

- [x] 真实双子进程测试统一登记、退出与外部进程保留，先观察失败。
- [x] 实现 `Children.configure_launch(count)` 与管道启动包装器；两道同步点保证全部登记及复查结束。
- [x] GPU PID 判定重新核实归属，新增带设备证据的 `GpuResourcesChanged`。
- [x] 实际进程测试通过，包括原子创建登记和取消。

## Task 3: 有界重新分配与恢复

Files: `xu_reallocation.py`, `xu_pipeline.py`, `xu_fleet.py`, `xu_training.py`, `xu_init.py`.

- [x] 验证阶段身份不受卡号变化影响、已完成产物保持、有界重试和实际训练恢复，先观察失败。
- [x] 只对确认的 GPU 资源变化重新调度；停止并确认本组退出后重新分配。
- [x] 设备成员变化写运行历史；新执行策略保持固定 batch 科研合同。
- [x] 实际训练恢复验证权重与 Adam 状态，未完成融合归档后重新计算。

## Task 4: 验收与发布

- [x] 运行相关测试及全套测试，记录条件跳过与现场验证边界。
- [x] 独立审查完整分支，修复重要问题并验证。
- [x] 更新文档、release manifest，核对发布目录与实测源码一致。
- 发布 GitHub 分支、合入 main，给出简短更新和训练命令；发布完成状态由 GitHub 合并记录确认。
