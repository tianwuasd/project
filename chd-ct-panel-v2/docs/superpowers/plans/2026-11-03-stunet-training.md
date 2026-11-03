# STU-Net-S Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 独立分支实现完整/部分监督STU-Net-S训练，真实本机短训与恢复验收。

**Architecture:** 原生STU骨干和nnU-Net增强，标注合同贯穿块加载与损失；确定性批次种子、CPU有界供数、单卡容量确认与原子检查点。

**Tech Stack:** Python3.11、PyTorch、nnunetv2 2.8.1、Blosc2、现有V2资源采集。

**Spec:** ../specs/2026-11-03-stunet-training-design.md

## Global Constraints

- 分支codex/chd-stunet-training-20261103；不修改原工作区或reproduction，不合main。
- 预处理/原始输入只读；只有已知生产者的三列坐标缓存允许明确适配。
- 默认微调1000轮、250/50、lr.001、B4、128³；短测配置另存且明确标记。
- 原始数据、权重、真实报告和运行日志不入Git。新验收/临时输出遵循D:/projects/AGENTS.md落到back分类目录。

## Review Focus

- 完整/部分样本混合与all-ignore：损失和梯度必须有限且不造MYO负例。
- test或未完整缓存混入：正式加载拒绝，短测必须显式选择并保留角色。
- 坐标列数、轴序、增强padding/ignore及深监督：原生接口和真实patch检查。
- 预取未消费/AMP跳步/半轮停止：游标和完整检查点必须可重放。
- 不同缓存、权重或配方恢复：写旧目录前拒绝；内存压力不暗改科学参数。

### Task 1: 输入与loss/model

**Files:** preprocessing/stunet_s.py、configs/preprocessing/*.json、training/contracts.py、data.py、model.py、losses.py、_vendor/stunet_architecture.py、tests/test_training_*.py。

- [x] 先写失败检查：原生四列坐标、旧缓存适配、角色/文件hash拒绝；完整loss原生梯度对照、部分监督边界。
- [x] 实现输入、官方模型和安全权重加载、marginal CE/有效类别Dice及原生增强取块。
- [x] 运行科学数值及接口检查，保存红→绿证据。

### Task 2: 资源/训练/恢复入口

**Files:** acceleration/training_cpu.py、training/engine.py、checkpoints.py、run.py、cli.py、smoke.py、setup_env.py、configs/training/stunet-s.json、requirements-training.txt。

- [x] 固定种子的小模型检查证明连续和恢复一致；有界供数、未消费重放与内存护栏可测试。
- [x] 实现真实CPU/单卡训练、验证、安全检查点、环境检查与薄CLI；正式与短测同一执行器。
- [x] 第一轮独立审查，修正实质问题并回归。

### Task 3: 本机验收/复核/交付

**Files:** docs/training-usage.md、README.md、pyproject.toml、代码清单；back/sync-reports下私有记录。

- [x] 复核现有预处理；按原角色补齐小样本main val缓存；真实原生取块及增强检查。
- [x] 作者骨干精确加载，真实GPU固定预算短训与验证，确认发生真实优化器更新；恢复并验证游标/状态。
- [x] 第二轮独立审查，必要修正后重跑；完整回归、编译、白名单打包和分支提交。

## Execution Ledger

- 复用干净的既有隔离工作副本，新建训练分支；原目录0.3.0白名单60文件校验后复制。51项基线通过，0.3.0先独立提交776652aa。作者身份沿用已有提交，仅命令级设置，不改全局Git配置。
- 当前本机RTX4060 Laptop8GiB可用；STU小模型作者权重已在研究权重目录。正式容量仍须真实规格验证，短测不能证明B4/128³可入场。
- 最终回归68项通过，编译与已有研究环境API检查通过；两轮独立审查发现的SIGINT、串行内存准入、模糊算法选择、GPU UUID、恢复动量预留及压力恢复顺序均已修正并覆盖检查。
- 真实3例短测使用原冻结train/aux_train/val角色，64³/B1、2轮、每轮8训练迭代/1验证迭代；16次尝试、12次真实更新、4次AMP溢出跳步。完整与辅助病例均发生更新。
- 第6次尝试暂停后用1个供数进程恢复，与2个供数进程连续运行逐值比较：130个模型tensor、128项优化器状态、scaler、RNG、训练状态和签名完全相等。更换精度的恢复请求被拒绝，原运行83份文件字节未变。真实证据保存在私有back/sync-reports任务目录，不随代码发布。
- 新坐标协议采用原生四列，旧两份已知生产者仅只读适配；按实际Blosc2块组织坐标核验，每个涉及块只解压一次，所有采样点仍核对实际标签。
- 交付0.4.0训练开发分支及代码白名单包；不合并main、不修改原reproduction。未进行正式长训练、全量三折、全体积预测、Linux负载或新环境安装验收。
