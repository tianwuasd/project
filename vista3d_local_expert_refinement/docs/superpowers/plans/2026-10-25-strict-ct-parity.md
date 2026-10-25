# Strict CT Parity Implementation Plan

> **For agentic workers:** 依用户已指定完整CT母版逐项实施；当前会话实施，独立审查复核，不另请求已明确的模板选择。

**Goal:** 修复v0.2的GPU/路径/发布缺口并逐项核对内部执行方法。
**Architecture:** 扩展Vista控制面，适配CT原版网络与发布核心；保留B0协议及原任务快照。
**Tech Stack:** Python3.11标准库、Bash、Git/SSH/Git LFS、nnU-Net现有独立环境。
**Spec:** ../specs/2026-10-25-strict-ct-parity.md

## Global Constraints

- 不运行真实患者训练或上传真实患者产物；不改现有8fit/4cal/250epoch/评价协议。
- CPU前置不因GPU不足等待，auto不越分配范围；原任务快照和原runtime优先。
- u/w/v统一官方认证、同密钥、子进程代理；发布前预览且不强推。

## Review Focus

1. 自动候选与并发上限独立；少卡可跑、无卡等待、CPU任务不问卡。
2. 自定义runtime/旧设置覆盖服务器默认路径；取消无半份保存。
3. 私有URL重写或错误origin不可把认证导向第三方。
4. 非项目提交、敏感文件、活动/篡改产物不得通过w/v。
5. 旧计划、产物依赖、CPU尾部和正式尺寸预检不能被新菜单改变语义。

### 1. 新建/GPU/路径

Files: task_panel.py、task_runtime.py、evaluation_gpu.py、gpu_queue.py、server_profile.py、06_configs/servers/zmic44.json；tests/test_strict_panel.py。

- [x] 先测试命名与路径在GPU之前、auto/候选数独立、配额与坏输入、CPU不问卡。
- [x] 按CT settings/launcher重整表单与展示；预填真实数据路径和独立目录约定。
- [x] 贯通worker与队列参数，少卡不拒绝；回归旧配置和历史任务。

### 2. CT原版发布核心与网络

Files: 05_src/panel_publication/*、task_update.py、task_panel.py；tests/test_publication.py。

- [x] 为CT原版网络/发布源码建立来源清单与适配说明。
- [x] 实现c/w/v，接通u，脚本参数与预览/确认/同包重试。
- [x] 本地合成Git、代理隔离、文件范围、归档完整性和活跃锁测试。

### 3. 子任务内部契约与逐项审计

Files: task_worker.py、task_inputs.py、server_launch.py、run_smoke_suite.py、任务契约/资源模块与审计文档；相关回归测试。

- [x] 逐条比对环境/预处理/软件短测/正式尺寸预检/训练/预测/CPU汇总。
- [x] 修复缺少的依赖门槛、资源预算、实际参数传递和完成/恢复边界。
- [x] 对照每项给测试证据和B0方法差异原因；不得只照搬CHD算法。

### 4. 验证与交付

- [x] 适用回归、Bash语法、纯标准库只读入口、独立复核。
- [x] 更新HANDOFF/STATE/DECISIONS与完整严格审计表；修订上次排除项。
- [x] GitHub增量提交/核验；交付直接Git升级与面板网络/代码/产物发布操作。
