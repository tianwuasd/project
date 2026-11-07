# Xu Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使正式入口实际使用有界 CPU 并行预取和经过核验的 GPU，提供可复核的实际性能证据。

**Architecture:** 保留现有训练与分布式损失实现。供数、输入缓存、只读 GPU 查询和运行保护各自独立，现有入口负责参数传递；训练完成边界仍保存原检查点。

**Tech Stack:** Python、PyTorch、SciPy、NumPy、Blosc2、psutil、NVIDIA cuda-bindings、系统 Git。

**Spec:** ../../acceleration-design.md

## Global Constraints

- 固定模型、全局 batch、病例顺序、增强随机数及 loss。
- 外部 GPU 任务不纳入本任务归属；只终止实际持有且身份匹配的子进程。
- 所有新增验证使用真实依赖和实际计算；中间产物仅写入已忽略的 local-validation。
- 旧运行目录和服务器已完成的两轮检查点保持原样。

## Review Focus

- 有 GPU PID 后出现新进程：持续检查必须停止本任务，不误杀外部进程。
- 慢的下一批和快的后续批：消费顺序保持，当前已就绪批次立即交付。
- 内存余量下降：停止新增；低于运行保留量时记录并终止本任务。
- 缓存损坏或来源变化：校验失败必须停止，不复用未知内容。
- 提前结束与异常：预取线程、CUDA复制和子进程全部收尾，完整 checkpoint 可恢复。

### Task 1: 有界 CPU 并行供数与缓存

**Files:** xu_supply.py、xu_sample_cache.py、xu_workflow.py、xu_init.py、xu_dispatch.py、xu_pipeline.py、scripts/xu_trial.sh；对应实际计算测试。

**Interfaces:** BatchSupply 保留 batches()/transfer()；增加可配置 worker 和资源/等待统计。CachedPrepared 代理原 PreparedCache，缓存原尺寸变换的结果。

- [x] 写实际计算测试并确认现有实现缺失的行为。
- [x] 实现顺序线程供数、预算约束、缓存校验和默认参数传递。
- [x] 验证原张量/更新、顺序、错误传播与恢复，记录实测结果。

### Task 2: 无计算上下文的 GPU 检查与持续保护

**Files:** xu_gpu_probe.py、xu_runtime_guard.py、xu_resources.py、xu_dispatch.py、xu_pipeline.py、环境依赖配置。

**Interfaces:** visible_devices()/snapshot() 仅返回只读 GPU 证据；RuntimeGuard.check() 核对自身进程身份及资源，错误直接向调用者传播。

- [x] 验证 CUDA Driver UUID 枚举与进程列表，不创建 Torch CUDA 上下文。
- [x] 接入筛选、启动二次检查及父进程运行检查。
- [x] 用实际进程验证归属、冲突/资源保护、日志与收尾。

### Task 3: 实测、审阅与发布

**Files:** scripts/xu_acceleration_check.py、tests 中对应验证、docs/xu-model-run.md、发布清单。

- [x] 使用本机已完成预处理病例，比较原读取和缓存/并行路径，验证真实更新一致。
- [x] 运行受影响的回归与短流程，记录验证范围、缺少的设备条件及计时。
- [x] 完成独立审阅，处理影响正确性和资源边界的发现。
- [ ] 校验发布文件，仅提交代码，发布并给出简短服务器命令。

## Execution ledger

- 2026-11-07：用户已明确要求设立目标并实施 v1/v2 优化，沿用会话中已确定的科研参数与工程实施授权。
- Pre-flight：供数/缓存接入 train 与 init；GPU 查询及保护共享 UUID/PID 身份格式；发布前核对所有入口。
- Task 1：顺序、实际 SciPy 插值、真实 PyTorch loss/参数/Adam/RNG 一致性通过；缓存校验和实际缺失文件失败已验证。
- Task 2：CUDA Driver 查询前后无上下文；真实外部计算进程拒绝、实际子进程身份与私有内存、内存保留值拒绝通过。环境安装实际复用原 Python，只写隔离查询依赖目录。
- 验证边界：本机可见显卡存在外部计算进程，遵照占用规则只进行只读 GPU 检查和 CPU 计算。GPU 固定页复制验证已接入服务器 inference 短测，成功结果必须包含实际复制统计。
- 真实 CT：三例 64³ 输入探针串行 4.38 秒，并行首次缓存 2.37 秒，缓存复用 0.72 秒；计时含操作系统缓存影响，不作为完整模型提速结论。真实 CT 的 64³、128³ smoke 模型各一轮已通过正式入口。
- 恢复：实际 smoke 网络在第二轮读取失败后，从完整第一轮恢复；最终参数、Adam 状态和最优损失与连续两轮逐项一致。
- Final: fixed CPU 共享预算 — test_actual_stage_limits_consumer_and_prefetch_together RED→GREEN，计算线程与供数线程共同满足真实预算；单工作槽明确拒绝并行预取。
- Final review：一名独立审阅者检查全部生产差异；没有 Critical 或延后 Minor。
- Final: Ruling: GPU 计算验证保留为服务器短测门槛 — 本机显卡存在外部进程，遵守拒绝占用规则 — 复制与多卡实际吞吐在本机无法确认。
- Final: Ruling: MIG 保持明确不支持 — 本次使用完整物理 GPU 的可见编号和 UUID — MIG 环境无法运行此资源策略。
- Final: Ruling: 保留现有子进程退出方式 — 独立审阅者实际验证 Windows venv 启动器终止时其 Python 子进程一同退出 — 其他启动器需要各自环境验收。
- 最终研究环境回归 127 项全部通过；数据分析环境回归共 46 项，其中 45 项通过、既有 1 项按条件跳过。
- 最终输入计时：三例真实 CT 的串行热读取 1.698 秒，缓存并行复用 0.541 秒，约 3.14 倍；对应同一小型 Conv3d 输入探针，loss 和参数逐项一致。
