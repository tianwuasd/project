# MedSAM3 Multigpu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 复用main公共资源策略，支持MedSAM3多卡训练及按病例并行预测，完成有限真实检查后合并main。

**Architecture:** 公共ResourceSession统一准入，多rank复用当前原生模型训练及预测。固定global batch和公共梯度求和保持每次更新目标，检查点保存逐rank随机状态及完整拓扑。

**Tech Stack:** Python、PyTorch distributed、MedSAM3官方模型/损失、项目公共acceleration。

**Spec:** ../specs/2026-11-24-medsam3-multigpu-design.md

## Global Constraints

- 原生1008输入、官方损失、医学LoRA和冻结病例角色不变。
- 1至4卡固定拓扑，不自动增加全局批量或降为单卡。
- 主机80%、至少1 GiB显存预留及有限时间/文件/磁盘上限继续生效。
- 本机每轮使用相同2例、最多2次更新；CPU/Gloo不替代多GPU/NCCL实测。
- 保护外部已有修改；只发布代码。

## Review Focus

- 不均等份额、未参与当前样本的LoRA参数：梯度归约与单进程同global batch一致。
- rank故障或超时：父进程停止同组子进程，无进程残留。
- 恢复拓扑/RNG：逐rank恢复，global batch或卡数变化拒绝继续。
- 预测分片：病例不重复、不遗漏，恢复复用完整结果。
- CPU/RAM：多rank总额不超过公共准入量。

## Task 1: 同步公共机制及固定执行合同

**Files:** config.py；models/medsam3/config.py、runtime.py、finetune.py；configs/models/medsam3*.yaml。

- [x] 合并固定main，配置冲突采用main；公共config.py与固定main一致。
- [x] 解析1至4卡、固定全局batch和每卡份额；构建覆盖全部rank的ResourceDemand。
- [x] 核验无模型检查、CPU/RAM预算和无效拓扑拒绝。

## Task 2: 训练及预测的多rank执行

**Files:** models/medsam3/training_worker.py、worker.py、runtime.py、finetune.py；新增launcher.py和distributed.py。

- [x] 共用隔离解释器启动与父进程监督，登记每个子进程出生身份，退出时停止本组。
- [x] 使用公共后端选择、协调和梯度同步；实现固定global batch采样与正确归约。
- [x] 保存/恢复逐rank RNG、拓扑和优化状态；在共同边界处理资源暂停。
- [x] 按病例分配预测并合并各rank结果；CPU评价沿用单进程。

## Task 3: 真实有限检查及合并

**Files:** tests/verify_medsam3_distributed.py、verify_medsam3_local_workflow.py；docs/MEDSAM3.md。

- [x] 官方LoRA实际权重和真实医学文本的CPU/Gloo两rank检查，覆盖不均等归约、RNG、拓扑拒绝，不更新完整模型。
- [x] 单GPU两例、每轮两次更新、退出恢复、验证及训练后单片预测；记录峰值/阶段耗时并清理副产物。
- [x] 整体代码审查和修正后的有限回归通过，远端基点未变化；本地main已快进合并，代码提交222eaa53。多GPU/NCCL实机边界已记录，远端发布结果以本次发布回执为准。

## 执行记录

- main基点678f4d23，整合提交0fdf28df；公共配置冲突依用户要求采用main。
- 审查后补充全组预测额度、逐轮验证边界、读取池互斥和精度/公共归约身份。最终同样两例的完整回归255.97秒通过；两轮累计4次更新，均为独立局部测试。
- 两CPU/Gloo进程实际通过公共启动器，包含[1,1]/[2,1]/[2,2]梯度归约、RNG文件保存恢复、空验证rank、额度和真实缺失检查点故障清理。
- Ruling: 训练采用main已有手动梯度求和的数据并行机制，固定global batch逐样本平均；目标多GPU/NCCL、完整资源暂停恢复和持续性能尚未实测，不能据本机检查宣称已验收。
