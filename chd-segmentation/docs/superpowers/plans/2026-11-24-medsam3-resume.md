# MedSAM3 Resume Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 遵循main的科学合同与运行资源分离原则，允许MedSAM3旧训练在资源调整后恢复，并说明现有预处理流程。

**Architecture:** 检查点记录完整科学合同及来源代码。科学合同校验后保留原训练身份；资源字段不影响身份。完整优化器更新边界可在固定全局批量下调整rank数，沿用main的新增rank随机状态派生方式。旧摘要格式仅接受可核验的已登记来源和原合同，不改写源检查点签名。

**Tech Stack:** 项目main恢复合同、PyTorch优化器/RNG、MedSAM3官方训练。

## Global Constraints

- 保留数据、类别、提示、预处理语义、精度和优化设置检查。
- 不将任意未知旧摘要作为兼容依据；缺少原合同和身份证明时明确拒绝。
- 单轮相同两例、总共两次更新，先由旧实现保存一次，再由新实现恢复一次；已有500文件/4 GiB/16 GiB/20分钟及清理规则继续生效。
- 不发布远端main，不修改外部已有文件。

## Review Focus

- worker、预取、CPU/RAM/显存限额变化不误拒绝。
- 输入内容、标签语义、精度、global batch或优化配方变化继续拒绝。
- 换卡仅发生在完整更新边界，旧rank恢复原RNG，新rank确定性派生。
- 来源身份和迁移记录保留，优化器、调度器和步数真实恢复。
- 预处理说明分别覆盖当前原图读取、已有NPZ复用、模型输入变换和空间假设。

## Tasks

- [x] 用原实现、真实病例保存旧检查点与可核验原合同。
- [x] 新增科学合同及旧格式核验，运行配置不绑定训练身份。
- [x] 完整更新边界支持新拓扑，记录恢复性质和来源；CPU/Gloo检查通过，完整多GPU/NCCL未测。
- [x] 改变资源后恢复并执行第二次真实更新，验证优化状态与拒绝分支；清理副产物。
- [x] 更新恢复与预处理说明，提交本地main。

验收：resume-compat-regression/resume-final-report.json及resume-boundaries.json。真实更新共两次；最终步补验证未新增更新，同一病例两片合同构造通过。检查点、切片及缓存78个文件已自动删除，释放616,788,270字节；日志与报告按原授权保留。
