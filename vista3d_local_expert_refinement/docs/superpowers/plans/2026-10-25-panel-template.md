# CT 母版面板适配 Implementation Plan

> **For agentic workers:** 按本计划逐项实施；本轮由当前会话直接完成并做独立复核。用户已要求完成后上传 GitHub。

**Goal:** 提供与 CT 母版相似的中文操作、环境安装和完整可核实的任务过程。

**Architecture:** 复用标准库控制面和冻结任务。新增只读过程聚合模块，安装子阶段独立落盘；现有 B0 执行器不变。

**Tech Stack:** Python 3.11+、Bash、unittest、GitHub Actions Linux。

**Spec:** ../specs/2026-10-25-panel-template-design.md

## Global Constraints

- 不修改 B0 8 fit / 4 cal、250 epochs、标签和指标；不启动患者训练。
- 查询只用标准库且不安装环境、不创建任务；原任务快照不变。
- 私有仓库使用官方认证原站；本机代理不当作服务器默认值。

## Review Focus

1. 旧任务缺少安装记录：显示未记录，不猜失败。
2. 有 PID 但身份已改变：不算活动任务，不显示别人的进程。
3. 新日志/截断/大日志：有界尾读、重新发现文件，不一次读全文件。
4. 安装中断或依赖配置改变：阶段/错误保留，不信任无版本旧凭证。
5. 空闲/等待 GPU/进程僵尸：状态与日志分开，不将心跳当模型推进。

### Task 1: 真实过程与日志查询

Files: 创建 05_src/task_progress.py；修改 task_queries.py；测试 tests/test_panel_progress.py。

Interfaces: `task_progress.step_lines(job, plan, work, state)`、`process_lines(job)`、`task_queries.log_sources(job)`；query 增加 log_source='all' 和 processes scope。

- [x] 加入旧记录、安装阶段、预处理完成病例、过期等待、身份检查和有界日志测试，先确认失败。
- [x] 实现只读聚合，中文状态、时间和路径，保留 models 原计时。
- [x] 通过新测试及 test_task_panel.py 回归。

### Task 2: 环境阶段与可恢复安装

Files: 修改 runtime_setup.py、task_worker.py、bootstrap_server.sh；测试 tests/test_environment_progress.py。

Interfaces: `runtime_setup.report_progress(env, phase, message, **fields)` 写 job/environment.json；安装器保存依赖签名凭证，仅成功后标记。

- [x] 测试 pip 配置隔离、下载字节、校验失败、环境失败阶段和 CPU 前置顺序。
- [x] 加入安装子阶段、时间戳和下载计数，worker 设置 current_step=installation 后再运行安装。
- [x] Bash 按依赖配置摘要决定检查/安装；支持已存在独立环境的核验；shell 语法与控制面回归通过。

### Task 3: 友好导航与发布

Files: 修改 task_panel.py、docs/SERVER_PANEL.md；新增 .github/workflows/vista3d-panel.yml；更新 .codex 交接。

- [x] 加入日志选择/持续刷新/环境入口/取消不提交的菜单测试。
- [x] 实现分层查询、分组新建表单、环境独立入口和提交后查询提示。
- [x] 本地运行控制面与协议回归；Linux CI 运行实际进程测试；独立审查修改。
- [x] 将本次增量文件发布 tianwuasd/project main，核验远端提交与 CI，给服务器受管更新命令。

执行注记：本地46项中42通过、4项Linux专属跳过；Linux CI已提交但被GitHub账户付款/支出上限在启动前阻止。独立审查问题已修复。代码已发布，Linux实际运行仍待平台恢复或服务器执行，不将此项标作通过。
