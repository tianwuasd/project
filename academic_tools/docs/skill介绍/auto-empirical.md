# Auto-Empirical Research Skills（AERS）

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[brycewang-stanford/Auto-Empirical-Research-Skills](https://github.com/brycewang-stanford/Auto-Empirical-Research-Skills)
- **本地目录**：[`skills/auto-empirical`](../../skills/auto-empirical/)
- **项目类型**：Stanford REAP × CoPaper.AI 方向的社科实证研究 skill 集合
- **仓库 README 当前说明**：以中文为默认入口，包含多个合集、9 阶段流水线、数值 benchmark 和行为评测
- **文章评价**：第二篇文章评为“顶级”

## 一句话定位

AERS 把社科实证研究的选题、文献、数据、识别策略、估计、稳健性、表图、写作、同行评审和投稿串成一条可介入的流水线，覆盖 Python、Stata、R 和 Quarto 等实证环境。

## 核心功能

- **选题精炼**：把模糊想法收紧为可证伪、可执行的研究问题。
- **文献综述**：检索、筛选、PRISMA 流程、批判性阅读和主题分析。
- **数据与识别策略**：支持公开数据/API/清洗，并覆盖 DiD、RD、IV、SCM、DML、matching 等方法。
- **三语言计量栈**：提供 Python、Stata、R 的 Full Empirical 合集，配合 pandas、statsmodels、linearmodels、pyfixest、reghdfe、csdid、fixest、did 等工具。
- **稳健性与复现**：包含复现包检查、HonestDiD、事件研究和稳健性审计方向。
- **表格与图形**：面向期刊的表格、图形、LaTeX 嵌入和结果产物。
- **论文写作与同行评审**：LaTeX/Quarto、审稿模拟、逐点回复和学术表达。
- **降 AIGC 与投稿**：包含多种中英文 AI 痕迹审查/改写 skill，以及投稿材料组织。
- **质量验证**：README 提供 numeric benchmark、行为评测、fixture 和外部 scoreboard 等质量信号；这些信号仍需结合具体版本理解。

## 适用场景

适合经济学、政治学、社会学、公共管理、心理学和其他需要计量实证的社科研究。尤其适合希望把 Stata/Python/R 分析流程标准化、保存可复现产物并减少手动表图工作的研究者。

## 优势

1. 从研究问题到投稿的流程覆盖完整，且每一步允许人工介入后继续。
2. 计量方法和软件栈具体，适合真实面板数据、因果推断和稳健性分析。
3. README 公开 benchmark、行为评测和安全信号，质量说明比单纯宣传功能更完整。
4. 能把分析代码、表格、图形、论文和审稿回应放入同一流程。

## 局限与风险

- skill 数量、合集数量和统计数字会随版本更新；不能把 README 的汇总数字当成每个模块都同等成熟。
- Agent 可以生成估计代码，但不能替代识别假设、变量构造、样本选择和统计推断的研究判断。
- Stata、R、Python、Quarto 及数据源有不同依赖、许可证和版本限制。
- “可复现”需要真实数据、固定代码、环境、随机性和完整日志；仅生成一份表格不够。

## 文章评价

第二篇文章评为“顶级”，称赞其从数据清洗到建模、表图和投稿的覆盖以及较强的流程质控。这个评价与仓库公开的 9 阶段流水线、benchmark 和 eval 结构相互印证，但不等同于对所有合集逐一复现。

## 推荐组合

- AERS 负责社科估计和稳健性，Claude Scholar 负责文献和项目知识。
- 用 WenyuChiou 版本先做 gap/feasibility gate，再让 AERS 执行通过门禁的实证设计。
- 用 PaperSpine 组织贡献、Results 映射和最终投稿，用 Nature Skills 改善英文表达。

## 使用前核查

1. 先选择 Python、Stata 或 R 的明确栈，不要让 Agent 在三种语法间隐式切换。
2. 在估计前确认识别假设、处理组/对照组、时间窗口、聚类层级和缺失值处理。
3. 保存原始数据、清洗脚本、估计代码、日志、环境文件和图表源文件。
4. 对 benchmark 和 Agent 输出做独立复算，并由熟悉计量方法的人审阅。

## 参考

- [仓库 README](../../skills/auto-empirical/README.md)
- [中文扩展正文](../../skills/auto-empirical/docs/CONTENT_ZH.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
