# PaperSpine

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[WUBING2023/PaperSpine](https://github.com/WUBING2023/PaperSpine)
- **本地目录**：[`skills/PaperSpine`](../../skills/PaperSpine/)
- **项目类型**：以贡献为先、面向审稿人的端到端学术写作编排 skill
- **仓库当前说明**：V4 是一个 `paper-spine` 编排 skill，内含 12 阶段路由；V5 为显式选择的发布候选版
- **文章评价**：第一篇文章介绍了该项目，但未给出分档；第二篇文章未重点测评

## 一句话定位

PaperSpine 先学习目标期刊/会议、优秀样例和用户的核心贡献，再逐单元设计论文，最后生成或重建 LaTeX、PDF、Word 和投稿材料。它的关键不是“把句子改顺”，而是保持贡献、结果、审稿意见与最终稿之间的可追踪关系。

## 核心功能

- **两种主流程**：Rewrite Existing 改进已有论文/报告；Build From Materials 从 PDF、图片、数据摘要、实验描述和初稿素材构建稿件。
- **目标场景**：期刊论文、会议论文、课程/技术报告、综述和竞赛论文。
- **12 阶段编排**：Intake、Research、Citation、Motivation Confirmation、Humanize、Writing/Drafting、Integrity Audit、LaTeX/PDF/Word、Submission Package、Translation Package、Review Response、Final Audit。
- **贡献优先**：没有确认的 contribution 不开始实质写作；Results 子节需要映射到贡献承诺。
- **审稿人视角**：在声称可投稿前生成 reviewer audit，包括审稿人价值图、异议登记和场景匹配。
- **可审计产物**：配置、参考材料索引、研究档案、样例学习、风格档案、gap map、引用支持库和各阶段状态落盘。
- **引用与翻译**：提供引用支持库、英文引用核验、中文翻译包和审稿意见回复流程。

## 适用场景

适合重视目标 venue 格式、论文贡献表达、审稿风险和多人协作交接的学术写作。特别适合已有一批素材但需要重新组织成完整稿件，或需要反复修订、保留动机记录的项目。

## 优势

1. 把“贡献是什么”置于写作前，避免围绕零散结果堆砌文章。
2. 阶段 gate 和断点续跑让复杂写作任务比一次性生成更可控。
3. 同时覆盖 LaTeX、PDF、Word、投稿材料、翻译和 rebuttal。
4. 以审稿人价值图、异议登记和最终审计形成完整的投稿前检查链。

## 局限与风险

- 过程较重，需要准备配置、参考材料和阶段产物；只想润色一段话时可能过度。
- gate 能检查结构和文件产物，不能替作者证明研究结论或期刊适配性。
- 生成 PDF、Word、LaTeX 和图表仍依赖本地工具链；失败时要检查环境和版本。
- 仓库明确提醒发布软件不等于授权外部投稿，最终学术责任仍由作者承担。

## 文章评价

第一篇文章强调 PaperSpine 的“贡献为先、面向审稿人”和动机追踪，但未给出夯/顶级等评级。这里将其作为结构化写作和投稿控制工具介绍，不把文章描述当成独立质量结论。

## 推荐组合

- 用 WenyuChiou 或 ARS-Codex 先做研究问题、缺口和证据边界，再进入 PaperSpine。
- 用 Scientific Agent Skills、AI Research Skills 或 Auto-Empirical 产生真实实验结果与图表。
- 用 Nature Skills 处理发表风格英文，用 Paper Craft 生成汇报或方法图。

## 使用前核查

1. 先确认 contribution、motivation、目标 venue 和可用素材，再启动阶段流程。
2. 阅读 `src/skill/SKILL.md`、对应 `references/` 和最终审计脚本，不要手改 `dist/`。
3. 核对引用支持库中的来源、年份、BibTeX 和可支撑句子。
4. 任何“可投稿”结论都要通过人工领域审查与目标 venue 要求核对。

## 参考

- [仓库 README](../../skills/PaperSpine/README.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
