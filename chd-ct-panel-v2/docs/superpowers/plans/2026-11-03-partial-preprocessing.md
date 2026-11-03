# Partial-label Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 完成独立的main96+aux13预处理协议，并保留完整监督基线。

**Architecture:** 新方法配置共享已有适配器；cohort负责科学分流与角色，annotation合同贯穿任务/缓存/pkl。只在96主队列train拟合，其余使用冻结参数。

**Tech Stack:** Python3.11、nnunetv2 2.8.1、NumPy/SciPy/Nibabel、Blosc2及既有V2 CPU执行器。

**Spec:** ../specs/2026-11-03-partial-preprocessing-design.md

## Global Constraints

- 原始图像/标签只读；96主队列原角色不重排；源灰度问题病例继续排除。
- 新增13例只用于辅助训练，不扩充验证/测试真值；本阶段无训练损失实现、GPU或面板。
- 标注0的背景歧义必须保存在合同内，额外标签仍为ignore8。
- 不覆盖其他工程或用户既有修改；代码修改前已保存精确字节备份。

## Review Focus

- 不同缺标模式不能都被当作MYO未标：分流单测拒绝。
- 已标小结构重采样后消失：I/O测试要求报错。
- 辅助病例极端灰度或spacing：拟合测试证明不影响主训练参数。
- 元数据丢失或被篡改：真实pkl读取、完成记录和缓存校验。
- 新旧方法/输出选择不一致：CLI、独立缓存、恢复只读拒绝测试。

### Task 1: 科学分流与标注合同

**Files:** configs/preprocessing/stunet-s-partial.json；preprocessing/annotations.py、cohort.py、registry.py；tests/test_partial_preprocessing.py。

- [x] 新测试先失败：96主+13辅、aux_train三折角色、缺MYO背景语义、其他缺标拒绝。
- [x] 实现标注合同和版本化方法，基线方法不改变纳入规则。
- [x] 通过科学分流测试；记录检查结果。

### Task 2: 预处理、拟合与消费边界

**Files:** preprocessing/fingerprint.py、stunet_s.py、run.py；cli.py、smoke.py；tests/test_partial_preprocessing.py、test_preprocessing_io.py、test_preprocessing_cli.py。

- [x] 新测试先失败：辅助不参与拟合、源未标允许、已标消失拒绝、实际pkl合同往返。
- [x] 逐折拆分aux_train，生成消费合同，使用相同主train规划；元数据贯穿任务/输出/缓存。
- [x] 增加cohort输出选择和完整+部分标注的短测；更新日志与摘要。
- [x] 运行完整合成回归，修正兼容问题。

### Task 3: 本机验收与交付

**Files:** docs/preprocessing-usage.md、docs/partial-preprocessing.md、README.md；pyproject.toml、scripts/package_code.py；outputs下私有验证记录。

- [x] 本机走真实短测：全量审计/完整拟合，输出完整与缺MYO病例，再检验恢复。
- [x] 读取标签/pkl/完成记录，核对角色、拟合集和原件哈希；对辅助池做适量真实验收。
- [x] 运行回归、编译、代码清单核验和白名单打包，保存验收范围与剩余服务器工作。

## Execution Ledger

- 2026-11-03：用户已明确要求执行并设立目标；当前对话直接实施，不增加重复审批。修改原项目目录；它在旧monorepo检出中为未跟踪目录，发布0.2.0逐文件SHA匹配。相邻reproduction存在另一个任务的修改，完全不触碰。
- 本次沿用既有资源执行器，不重构CPU策略。新配置与合同显式隔离，不绕过旧运行的严格源码签名。
- 科学合同与实际Blosc2往返测试经历失败→实现→通过；完整回归50项通过。补充了辅助角色校验和缓存annotation检查；重复进程内合成测试暴露PyTorch interop线程只允许设置一次，已改为仅在尚未为1时设置，不改变实际worker的单线程策略。
- 已完成独立只读源码审查，未发现实质缺陷；该审查未独立重跑测试，真实验收仍在执行。环境实际导入/API检查和编译通过，未安装依赖或申请GPU。
- 真实验收完成：110对审计通过，main96/aux13/excluded1；fold0完整标注与缺MYO各一例通过原尺寸变换、逐块回读、pkl/完成记录/缓存合同检查。恢复复用110项审计与2例产物，新处理0、失败0。
- 新旧fold0主队列fit_ids、CT统计和目标spacing完全相同；一个完整标注病例的影像与标签全部数组值逐块比较一致。辅助不进入参数拟合、验证、测试。
- 针对真正改变网格的部分标注合成输入补充了验证，最终51项回归通过。验收只覆盖2例真实重采样，不冒充109例三折完成；Linux新环境/实际负载与正式训练未执行。私有详细记录留在outputs/validation，不放入代码包。
