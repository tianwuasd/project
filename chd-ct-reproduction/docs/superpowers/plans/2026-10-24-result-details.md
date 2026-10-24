# 分割结果明细 Implementation Plan

**Goal:** 让旧/新 CT 任务的结果页显示可追溯的整体与逐折分割明细，并展开最后的诊断决策与逐疾病评估。

**Architecture:** 保留 results.py 的查询入口；用独立标准库模块整理保存的指标与折归属，来源定位与中文显示接入现有结果页。

**Tech Stack:** Python 3.11+ 标准库、pytest、Ruff；Markdown skill 模板。

**Spec:** ../specs/2026-10-24-result-details-design.md

## Constraints and review focus

- 不改变训练、预处理、推理或 Dice 算法；不要求旧任务重跑。
- 逐折缺失与集成不能伪造分折；数字类型损坏不导致整个查询崩溃。
- 旧记录可以按病例汇总，但缺失不能变成零，分母不能按固定病例数猜测。
- 来源来自所选任务的记录，外部预测不可误连到同名本地步骤。
- 查询须能在 python -S 下执行，保留文件字节、mtime 和最近任务指针。

## Tasks

- [x] 保存修改前文件与 SHA256，核对远端及 skill 安装版差异。
- [x] 扩展 tests/test_result_viewer.py：分折结构分数/有效分母/病例名单；旧逐病例回退；集成与缺失折；坏类型；来源和合成步骤；无 ML 依赖只读查询。
- [x] 实现 src/chd_ct/server/result_metrics.py 的只读指标整理，扩展 results.py 的来源与步骤解释，验证上述用例先失败后通过。
- [x] 按用户补充要求实现 result_decisions.py：诊断训练可用性、真实决策树路径、规则条件/缺失证据、叶节点比例与逐疾病混淆/覆盖统计；核对生产端字段并新增回归。
- [x] 更新版本、README、server-start 和维护约定；补充实际验证记录。
- [x] 修改 server-engineering 的结果模板及整体模板路由，用实际场景复核，验证 skill 格式和源码/安装版哈希。
- [x] 运行相关面板/查询/工作流回归与 Ruff；结果见 docs/validation-result-details.md。
- [ ] 检查差异后分别提交推送 CT 和 skill，核验远端与镜像，报告服务器未部署（状态以对应提交的发布记录为准）。
