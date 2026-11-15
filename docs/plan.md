# 小样本流程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供独立的2/1/1真实流程、计时、恢复与清理工具。

**Architecture:** 父进程管理版本、预算、进程和保留策略；工作子进程调用指定生产项目接口。所有产物在工具自己的运行目录。

**Tech Stack:** Python、psutil、filelock，外部生产环境中的STU-Net/MONAI/nnU-Net。

**Spec:** docs/design.md

## Global Constraints

- 生产基点60d6e5f17c81f1fea78690021fbf403b1ee61f37；生产目录只读。
- 2例训练、1例验证、1例测试，患者互斥；只拟合训练病例。
- 全局batch 4；最多2轮、每轮2次尝试，总计最多4次；不追加AMP跳步。
- .runs最多500文件、4 GiB工作占用；必要日志最多20 MiB；副产物自动清理。
- 不登录服务器、不自动发布、不合并main。

## Review Focus

- 患者缺失、重复患者及源角色改变：选择必须拒绝。
- 生产提交相同但源码有修改：阶段前后拒绝执行。
- 运行目录链接、越界、来源重叠：写出及清理前拒绝。
- 子进程失败、超时、容量超限：停止所属进程并清理，保留真实错误。
- 恢复与测试范围：累计尝试有界，fit_ids及预测病例精确匹配。

### Task 1: 输入、版本和预算合同

**Files:** run.py、src/chd_mini/contracts.py、tests/test_contracts.py。
**Interfaces:** validate_settings(dict)->dict；select_cases(manifest,seed)->list；verify_engine(root,commit)->dict。

- [x] 运行真实单元断言，确认缺少模块时失败。
- [x] 实现配置、冻结角色选择、患者互斥及版本检查。
- [x] 运行 `python -B -m unittest discover -s tests -v`，合同检查通过。
- [x] 提交本任务明确文件。

### Task 2: 完整流程与收尾

**Files:** src/chd_mini/worker.py、runner.py、cli.py、configs/example.json。
**Interfaces:** worker.execute(request,stage)->JSON记录；runner.run(settings,run_id)->summary。

- [x] 先验证入口缺失时失败。
- [x] 接通选例、拟合、准备、逐例预处理、一次尝试保存、恢复到终点、测试预测评价。
- [x] 实际执行四例CPU技术检查，保存计时、恢复和来源证据；CUDA外部依赖异常单列。
- [x] 核对清理记录、资源退出和留下的日志容量。
- [x] 提交本任务明确文件。

### Task 3: 独立更新与交付

**Files:** update.py、README.md、tests/test_update.py。
**Interfaces:** update只接收独立Git根目录，拒绝生产项目和活动运行。

- [x] 验证真实生产目录被更新器拒绝。
- [x] 用本工具真实已提交版本在独立测试检出验证检查与同版本快进，结束清理测试检出。
- [x] 进行独立全分支审查，修复重要问题并运行针对性验证。
- [x] 记录独立分支、提交、运行命令与实际检查限制。
