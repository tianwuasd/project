# 运筹学研究 Agent

基于论文证据推进 **找场景 → 数学建模 → 求解算法与验证**。包含可安装的 `$or-research` 技能、可追溯文献索引工具，以及已运行的低空物流强化学习研究试验。由当前宿主agent推理，本地工具管理证据与计算；没有独立常驻模型服务，不需要另外配置付费模型API。

## 先看成果

- [Agent 技能入口](skill/SKILL.md)：三阶段流程、证据规则与工具调用。
- [低空物流强化学习研究](studies/trc_capacity_rl/README.md)：公开LaDe数据、共享起降/机队/充电预约、单调后决策TD、基线、消融、五训练种子。
- [英文论文初稿](studies/trc_capacity_rl/manuscript.tex)：完整模型、证明、方法、实验及局限；LaTeX可编辑，当前宿主编译器环境错误，排版编译尚未验证。
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

Agent工具测试13项、研究测试9项通过。图表数值来自实际运行。单调命题的条件、额外凹性假设、部分观测近似、合成揭示时序与统计局限均在论文中明确。
