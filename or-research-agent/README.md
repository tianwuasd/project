# 运筹学研究 Agent

基于论文证据推进 **找场景 → 数学建模 → 求解算法与验证**。包含可安装的 `$or-research` 技能、可追溯文献索引工具，以及已运行的低空物流强化学习研究试验。由当前宿主agent推理，本地工具管理证据与计算；没有独立常驻模型服务，不需要另外配置付费模型API。

公开代码位于 [tianwuasd/project 的 or-research-agent 子目录](https://github.com/tianwuasd/project/tree/main/or-research-agent)。

## 项目包括什么

| 部分 | 现有内容 | 入口 |
|---|---|---|
| 研究 Agent | 找场景、建模、算法设计三阶段流程，证据规则和实验规范 | [skill/SKILL.md](skill/SKILL.md) |
| 文献工具 | 文献建库、关键词检索、来源追溯、正文片段提取；个人数据库仅保留在本机 | [skill/scripts](skill/scripts) |
| 低空物流案例 | 共享起降、机队、充电预约，联合情景与部分相关扰动下的概率约束 | [研究说明](studies/trc_capacity_rl/README.md) |
| 求解与实验 | 后决策TD、单调及资源组合特征、风险插入特征；贪心、容量定价、条件LP前瞻及先知MILP对照 | [src](studies/trc_capacity_rl/src) |
| 数据与审计 | LaDe公开配送任务、209份公开飞行日志的派生统计、模型检查点、逐日结果、可行性审计 | [最新结果](studies/trc_capacity_rl/development/RESULTS_03.md) |
| 论文与目标 | 英文LaTeX研究初稿、最邻近文献对照、创新审查及持续研究验收要求 | [论文](studies/trc_capacity_rl/manuscript.tex)、[目标](RESEARCH_GOAL.md) |

截至2026-10-22，项目完成了三轮开发试验，研究目标仍在进行。第三轮事后价值缩放诊断的最好均值为81.13，同约束条件LP基线为82.17；尚无可靠的方法优势。补充回放审计覆盖400个训练日和330个选型验证日，均满足所建模型的可行性检查；这不构成真实飞行安全验证。全新确认实验尚未启动。

## 先看成果

- [持续研究目标](RESEARCH_GOAL.md)：建模区别、结构匹配方法、独立数据上的可靠优势，三项同时验收；目标进行中。
- [第二轮开发模型](studies/trc_capacity_rl/development/scenario_model.md)、[全文对照](studies/trc_capacity_rl/literature/fulltext_comparison.md)：共同扰动下的返航预约与资源组合价值。当前仅探索性候选，尚未确认创新。
- [第三轮风险模型与结果](studies/trc_capacity_rl/development/RESULTS_03.md)：209份实飞日志审计、部分相关扰动反例和联合概率约束。新特征79.99对同约束强基线82.17，未达优势目标。

- [Agent 技能入口](skill/SKILL.md)：三阶段流程、证据规则与工具调用。
- [低空物流强化学习研究](studies/trc_capacity_rl/README.md)：公开LaDe数据、共享起降/机队/充电预约、单调后决策TD、基线、消融、五训练种子。
- [英文论文 PDF](studies/trc_capacity_rl/manuscript.pdf)、[LaTeX 源码](studies/trc_capacity_rl/manuscript.tex)：完整模型、证明、方法、实验及局限。2026-10-22 已用独立 Tectonic 0.17.0 编译并检查全部15页；宿主编辑器的内置预览仍有 Windows 应用包执行权限问题。
- [创新审查](studies/trc_capacity_rl/NOVELTY.md)、[审查修复记录](studies/trc_capacity_rl/REVIEW.md)、[实验结果](studies/trc_capacity_rl/results/summary.json)。

本次单调RL测试日均节省69.05，贪心63.93，容量定价74.93（均为模拟广义成本单位）。研究初稿如实报告：RL尚未超过最强已实现在线基线，创新性与TRC投稿水平尚未确证。

## 使用 Agent

将 `skill` 文件夹安装到个人技能目录，目录名设为 `or-research`。在支持本地技能的宿主中调用：

> 使用 $or-research，结合我的论文库，找三个低空物流研究场景，核对最邻近工作，选择一个并给出完整模型、算法与可复现实验。

也可以直接让当前agent读取 `skill/SKILL.md`。技能脚本相对于该文件定位；数据和输出相对于项目根目录，或由环境变量 `OR_RESEARCH_HOME` 指定。安装后的个人资料路径在本机配置，公共仓库不含个人文献或私人路径。

## 配置自己的论文库

将 `config/sources.example.json` 复制为 `config/sources.json` 并填写本机路径。支持现有JSON记录格式、WoS文本记录，以及PDF/Markdown精读目录。索引保留来源哈希、证据层级和读取定位；PDF清单不代表已读取全文。

在本项目目录运行：

```sh
python -X utf8 skill/scripts/corpus.py build --config config/sources.json
python -X utf8 skill/scripts/corpus.py stats
python -X utf8 skill/scripts/corpus.py search "无人机 骑手" --limit 8
python -X utf8 skill/scripts/new_run.py "站点容量下的协同配送"
python -X utf8 skill/scripts/solve_demo.py --output runs/demo.json
python -X utf8 -m unittest discover -s tests -v
```

外部数据库可通过全局 `--db` 参数指定，放在 `search/build` 等子命令前。索引默认使用Python标准库；PDF提取需要PyMuPDF，MILP需要SciPy。检索是中英文别名与文本匹配，不是向量语义检索。论文收藏不等于本人著作，笔记不替代原文核验。

## 研究复现与公开范围

依赖见 [requirements.txt](requirements.txt)，完整训练命令见[研究说明](studies/trc_capacity_rl/README.md)。原始公开数据由下载脚本重建；论文数据库、收藏PDF、个人阅读笔记及本机配置不进入公共仓库。公开包保留代码、技能、检查点、汇总与逐日结果，便于审查。

Agent工具测试13项、研究测试35项通过（含第三轮概率约束与完整训练审计检查）。图表数值来自实际运行。单调命题的条件、额外凹性假设、部分观测近似、合成揭示时序与统计局限均在论文中明确。
