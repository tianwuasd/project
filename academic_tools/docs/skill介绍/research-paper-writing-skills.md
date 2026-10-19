# Research Paper Writing Skills

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[Master-cai/Research-Paper-Writing-Skills](https://github.com/Master-cai/Research-Paper-Writing-Skills)
- **本地目录**：[`skills/Research-Paper-Writing-Skills`](../../skills/Research-Paper-Writing-Skills/)
- **项目类型**：面向 ML/CV/NLP 论文写作的规则、模板和检查清单 skill
- **知识来源**：仓库说明大部分方法论来自彭思达老师公开学习笔记，并由仓库作者整理、结构化和封装
- **文章评价**：第二篇文章评为“NPC”

## 一句话定位

Research Paper Writing Skills 更像一份可调用的论文写作方法论手册，而不是自主完成研究的代理。它帮助作者把已有草稿、实验和论据组织成更清晰的摘要、引言、方法、实验、结论和审稿回复。

## 核心功能

- 撰写或重写 Abstract、Introduction、Related Work、Method、Experiments、Discussion 和 Conclusion。
- 检查段落之间的逻辑、章节结构、claim-evidence 对齐和术语一致性。
- 以 reviewer 视角做提交前自审，发现论证缺口、证据不足和表达问题。
- 提供各章节修辞结构指南、写作清单、图表规划模板和 brainstorming guide。
- 按仓库规则区分已有内容、用户确认内容、推断和建议性扩展。
- 明确要求不编造数据、DOI、期刊信息、实验结果，保留公式、变量、模型名和参考文献的原始形式。

## 适用场景

适合已经有实验结果、论文草稿或结构化笔记，尤其是 ML/CV/NLP 论文的章节改写、逻辑审查和 rebuttal 准备。它不负责替代真实实验、文献检索或研究设计。

## 优势

1. 规则集中、上手成本低，适合作为写作时的稳定检查表。
2. 关注 claim-evidence 对齐和审稿人视角，能减少“句子漂亮但论证空”的问题。
3. 对摘要、引言、方法和实验等论文单元有明确的章节化参考。
4. 证据规范直接写进项目说明，能约束模型不要虚构数据或文献元信息。

## 局限与风险

- 文章作者认为它主要是公开笔记和提示规则的集合，适应具体项目和卡点的能力有限；这里属于文章体验评价。
- 它不会自动为你的研究发现新证据，也不能验证实验是否正确。
- 如果输入素材本身有虚构引用或错误结论，写作 skill 可能只能改善表达，不能替代引用核验。
- 公开笔记的适用领域和写作风格不一定匹配所有学科、期刊或中文学位论文规范。

## 文章评价

第二篇文章评为“NPC”：作者认为它像一份高质量考试笔记，照着执行不容易出大错，但不会针对具体研究情境动态调整。这个标签是文章作者的主观分档，不代表项目没有使用价值。

## 推荐组合

- 先用 Claude Scholar 或 Scientific Agent Skills 收集和核验文献、数据与实验结果，再交给本 skill 写作。
- 用 ARS-Codex 或 PaperSpine 做更严格的证据、贡献和审稿关卡。
- 需要英文学术表达时，可与 Nature Skills 做互补：本 skill 负责论证结构，Nature Skills 负责风格层面的英文表达。

## 使用前核查

1. 提供真实的实验结果、表格、引用和目标会议/期刊要求。
2. 把需要核验的 claim 单独列出，不让润色过程掩盖证据缺口。
3. 写完后检查引用、数值、图表编号、方法名和术语是否与源材料一致。
4. 不把语言流畅度当作研究有效性或可投稿性的证明。

## 参考

- [仓库中文 README](../../skills/Research-Paper-Writing-Skills/README_zh.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
