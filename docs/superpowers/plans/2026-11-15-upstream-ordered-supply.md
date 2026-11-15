# Upstream Ordered Supply Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 助手1完成本轮测试与发布后，助手2继承其最终确定版本并核对Q、MedSAM2适用性。

**Architecture:** 以助手1完成回执的本地提交与公开main提交为来源，比较完整子树及逐文件差异。公共STU供数、主机协调和构建预算沿用该版本，模型预算与官方采样接口保留各自约定。

**Tech Stack:** Python、PyTorch、multiprocessing、Git、官方MedSAM2。

**Spec:** ../specs/2026-11-15-model-ordered-supply-design.md；补充用户顺序要求：助手1修复、测试、发布全部结束后才执行最终整合。

## Global Constraints

- 必须核对本轮最终本地与公开提交，不用e5264c2或210d80b代替。
- 只编辑助手2目录，保留独立分支与索引；不登录服务器，不自行使用GPU。
- Q保持r2随机调用与全局batch；MedSAM2保持官方序列、增强、优化及恢复worker布局。
- 真实一至两例、各阶段最多两次更新，产物在已批准t17目录生成并清理，报告总量不超过10MiB。
- 保留实际耗时日志及测量设备、数据范围，CPU短测不作为服务器训练速度。

## Review Focus

- 新的worker数量上限不能阻止MedSAM2恢复旧worker布局。
- 模型工作区预算、官方CPU训练内存下限及病例字节额度不能被通用预算替换。
- 通用供数/主机协调新签名须在共享engine中完整继承，已有设备恢复检查保留。
- 通信超时配置用于Q时不通过增大超时掩盖慢数据准备；MedSAM2保留官方默认值。
- 精确源码兼容清单覆盖64f5510及既有来源，未知改动继续拒绝。

### Task 1: 最终版本同步与验收

**Files:** 公共config、data/supply、training/budget、build_gate、coordinator、engine与相关测试、说明；模型execution和兼容清单；原供数计划的待同步项。

**Interfaces:** cpu_plan返回workspace_capacity_bytes及闭合breakdown；模型保持memory_profile、fixed_workers与retained_case_bytes；通用HostCoordinator按最终源码使用；Q从公共资源配置取得通信超时。

- [x] 读取助手1完成状态、最终本地与公开SHA，并核对公开子树来源。
- [x] 保留模型专属部分，继承公共代码并逐项审查差异；先运行能揭示布局或预算问题的检查。
- [x] 运行真实病例预算检查、Q训练及恢复、官方MedSAM2数据接口，保存计时和队列日志并清理。
- [x] 更新精确兼容记录、材料说明、本地提交和来源追踪；记录远端与服务器状态。
- [x] 一次独立整体审阅并修复重要发现，完成最终检查与报告。
