# Whole-heart Panel Implementation Plan

**Goal:** 依据现行 server-engineering 模板升级服务器任务入口，训练和预测共享可扩展模型多选。
**Architecture:** 标准库面板与任务生命周期层调用既有训练环境；科学算法保持，选择与来源显式持久化。后台使用每任务源码快照；旧记录只读或经独立任务复用。
**Tech Stack:** Python 标准库、Linux flock/proc、现有 nnU-Net/PyTorch 环境与 unittest。
**Spec:** `docs/server-panel-design.md`。
**Execution:** 本会话直接实施用户已指定的模板适配。

## Global Constraints
- 仅 B0、B1；七结构、冻结划分、250/300轮与指标定义保持。
- 标准库查询，无训练、网络、安装副作用。新任务独立目录，原模型与旧结果不覆盖。
- 明确模型选择贯通训练、短测和预测；未知/空选择拒绝；旧无选择清单仍按完整十模型校验。
- CPU前置不申请GPU，选择不能突破继承可见范围；查询/退出不会停止任务。

## Review Focus
1. 只选B1时不能偷偷训练B0，也不能因缺B0使汇总失败。
2. 单CT范围不能因旧“十模型”假设等待MRI。
3. 两个入口、历史恢复和新任务不混用保存的选择。
4. 父进程结束但子进程仍活跃时锁不能提前释放，PID复用不得误杀。
5. 旧模型签名/缓存变化必须明确拒绝或在新目录回退，不能改旧记录。

## Task 1: 模型与范围契约
- Files: `05_src/model_catalog.py`, `server_launch.py`, `server_train.py`, `run_smoke_suite.py`, `evaluation.py`, `evaluation_summary.py`。
- Interface: `parse_models(value) -> list[str]`; `selection(work, folds=None, methods=None) -> dict`，只有显式选择才改变旧全量语义。
- [x] 添加模型选择、单模型/单模态评价测试。
- [x] 实现共享注册表、选择冻结、队列传递、条件配对汇总。
- [x] 运行新测试与可在本机执行的既有训练/评价回归；完整nnU-Net与真实GPU回归待目标环境。

## Task 2: 任务与缓存生命周期
- Files: `05_src/task_runtime.py`, `task_inputs.py`, `task_worker.py`, `gpu_devices.py`, `server_data.py`。
- Interface: `submit(plan, foreground=False)`, `resume(job)`, `stop(job)`, `history(runtime)`；plan记录name/methods/folds/tasks/source/code。
- [x] 添加唯一目录、查询不写、来源复用、取消/锁竞争与显卡范围测试。
- [x] 实现冻结快照、原子状态、共享锁继承、独立重跑及后台逐步执行。
- [ ] 目标Linux合成任务验收退出查看、中断和恢复（本机无可用Linux，已在验收记录列明）。

## Task 3: 面板、查询与更新
- Files: `05_src/task_panel.py`, `task_queries.py`, `task_update.py`, `start_server.sh`, `run_zmic44.sh`, `bootstrap_server.sh`, `zmic44_setup.py`。
- [x] 实现新建/历史/返回/退出、多选、按步骤设置；CLI共用同一契约。
- [x] 实现只读状态、逐模型日志、分层结果；更新独立版本，不改共享monorepo其它项目。
- [x] 验证 `python -S` 查询、菜单取消、命令帮助、单模型选项和更新失败保留旧版。

## Task 4: 交付
- [x] 更新入口说明、方法适配、配置示例、兼容边界与验收记录。
- [x] 运行相关回归，检查差异仅限whole-heart，报告本地验证和服务器未部署的区别。
