# Scientific Agent Skills

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)
- **本地目录**：[`skills/scientific-agent-skills`](../../skills/scientific-agent-skills/)
- **项目类型**：面向 Agent Skills 标准的科学计算与科研技能集合
- **仓库 README 当前标注**：163 个 skill、100+ 数据库、70+ Python 包技能；数量会随上游更新变化
- **文章评价**：两篇文章均评为“夯”

## 一句话定位

Scientific Agent Skills 为 AI Agent 提供跨生物信息学、化学、医学、材料、物理、工程、地理和数据分析等领域的可复用文档、代码示例、数据库接入和工作流指导。

## 核心功能

- **生物与医学**：序列分析、单细胞 RNA-seq、基因调控、变异注释、蛋白质工程、临床研究和医学影像研究。
- **化学与药物发现**：RDKit、DeepChem、分子性质、虚拟筛选、ADMET、分子对接和分子动力学。
- **科学数据库**：统一的数据库查询 skill，以及 PubChem、ChEMBL、UniProt、COSMIC、ClinicalTrials.gov 等多类数据源。
- **数据分析与可视化**：统计分析、网络分析、时间序列、EDA、出版级图表和大规模数据处理。
- **科学包工作流**：覆盖 Scanpy、scikit-learn、PyTorch Lightning、GeoPandas、pymatgen、Qiskit、OpenMM/MDAnalysis 等包的版本化用法与例子。
- **实验与平台集成**：Benchling、DNAnexus、OMERO、Protocols.io、Opentrons 等实验或数据平台。
- **科学沟通**：证据可追溯写作、文献综合、同行评审、PPTX/海报/示意图和文档处理。
- **质量与安全**：仓库为带脚本的 skill 配套测试，并提供结构校验、脚本解析和安全扫描说明。

## 适用场景

适合有真实科学数据、专业 Python 包或数据库需求的研究工作。生物信息、药物发现、医学研究、科学机器学习和多学科实验自动化是它的强项；单纯的中文论文润色不是它的主要目标。

## 优势

1. 覆盖面广，能减少查包文档、拼 API 和反复配置环境的时间。
2. 每个 skill 通常带 `SKILL.md`、示例、用例、最佳实践和参考资料，便于 Agent 按任务选择。
3. 兼容 Cursor、Claude Code、Codex、Google Antigravity 等支持标准的宿主。
4. README 明确给出安全边界，提醒用户阅读 skill、查看贡献历史并自行扫描。

## 局限与风险

- “163 个 skill”和“100+ 数据库”是当前仓库 README 的宣传/统计口径，不代表每个 skill 都适合你的领域，也不代表所有接口持续可用。
- 很多功能依赖 Python 包、API、账号、网络或本地计算资源；安装 skill 不等于安装这些运行依赖。
- 临床、诊断、治疗和监管相关模块必须限定为研究或证据准备，不能把 Agent 输出当作医疗或合规决定。
- 社区贡献的 skill 审查深度可能不同。不要一次性全量安装；先阅读目标 `SKILL.md` 和外部调用。

## 文章评价

文章作者将其评为“夯”，称赞其封装 RDKit、Scanpy 等专业工具、能直接用于复杂科学计算。这个评价说明它作为科学计算工具箱的覆盖价值，但不替代对具体包版本、数据和计算结果的验证。

## 推荐组合

- 与 Claude Scholar 组合：Scholar 管理文献和项目知识，Scientific Agent Skills 执行数据分析。
- 与 ARS-Codex 或 PaperSpine 组合：前者定义研究问题和证据边界，后者把计算结果映射到论文贡献和 Results。
- 与 Paper Craft 或 Nature Skills 组合：生成可解释的科学图、图解和论文级表达。

## 使用前核查

1. 只安装目标领域或目标包，先查看对应 `SKILL.md`、版本和测试。
2. 确认数据库条款、API 额度、联网权限和数据隐私要求。
3. 固定环境和数据版本，保存运行脚本、参数、随机种子和原始输出。
4. 对统计、结构预测、实验结论和临床相关输出做领域专家复核。

## 参考

- [仓库 README](../../skills/scientific-agent-skills/README.md)
- [安全说明](../../skills/scientific-agent-skills/SECURITY.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
