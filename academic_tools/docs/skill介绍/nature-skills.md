# Nature Skills

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills)
- **本地目录**：[`skills/nature-skills`](../../skills/nature-skills/)
- **项目类型**：面向学术阅读、Nature 风格英文、审稿回复和科研绘图的技能库
- **仓库 README 当前标注**：19 个 skill，支持 Claude Code、Codex、OpenClaw、OpenCode 和 Hermes 等宿主
- **文章评价**：第二篇文章评为“顶级”

## 一句话定位

Nature Skills 把学术写作中的段落表达、论文阅读、文献和他引整理、审稿模拟、审稿回复、科研图和 PPT 等任务封装成可安装的 `nature-*` skill，强调可直接交付的科研产物。

## 核心功能

- **nature-reader**：将论文做成图文对应的中英文 Markdown reader，辅助阅读和理解。
- **nature-polishing / nature-writing**：在保持学术含义的前提下改善 Nature 风格英文的段落组织、措辞和句式；也可辅助摘要、引言和讨论。
- **nature-reviewer / nature-response**：从目标期刊审稿人角度进行预投稿评估，并生成逐点回复和 cover letter 草稿。
- **nature-figure**：根据方法和结果规划、生成科研图或论文示意图草稿。
- **nature-paper2ppt**：把论文转为中文组会或学术汇报 PPT，保留关键图件和来源标注。
- **文献与引用辅助**：整理引用数、严格他引、DOI 和引用者画像；实际数据仍需通过原始数据库核验。
- **共享参考包**：`nature-shared` 为多个技能提供共同参考资料，不能简单把目录名当成独立功能页面。
- **多宿主安装**：README 提供 `npx skills`、Claude Code wrapper 和 Codex 安装路径，支持按技能或全量安装。

## 适用场景

适合需要英文论文润色、Nature/高水平期刊风格表达、科研绘图、论文阅读器或组会汇报的研究者。其重点是科研表达和产物呈现；它不替代研究设计、统计审计和引用真实性检查。

## 优势

1. 关注整段表达和学术语域，不只是拼写与语法修正。
2. 把英文学术表达、审稿回复、科研图和文献阅读放在同一套技能体系中。
3. 支持中英文读者和多个 Agent 宿主，且可单独安装某一个 `nature-*` skill。
4. README 给出较具体的输入范例，便于从真实论文、段落、审稿意见和图件开始。

## 局限与风险

- “Nature 风格”是表达目标，不代表内容符合 Nature 的真实投稿要求或已经达到发表标准。
- 语言润色可能无意改变 claim 强度、因果含义、限定条件和专业术语；必须逐句与原稿比对。
- 科研图生成可能错误表达结构、流程、数据点、比例、单位或引用，不能直接作为投稿图。
- 引用数、他引、DOI、研究者身份等动态事实必须回到 Crossref、期刊页面或其他权威来源核验。
- 仓库 README 同时包含社区、商业服务和联系方式；使用 skill 时只需关注 `skills/`、共享参考和安装说明，并审查外部链接。

## 文章评价

第二篇文章评为“顶级”，用中式英文段落示例说明其改写不止于语法纠错，而是改善整段学术表达，并称赞其发表级配图能力。该评价是体验性判断；实际使用中应优先保护原文科学含义和证据边界。

## 推荐组合

- 用 Claude Scholar 或 ARS-Codex 先完成文献与引用核验，再用 Nature Skills 做表达层润色。
- 用 Scientific Agent Skills、AI Research Skills 或 AERS 生成真实结果和数据图，再用 `nature-figure` 做版式草稿。
- 用 PaperSpine 先确定贡献和 reviewer audit，用 `nature-response` 起草逐点回复。
- 用 Paper Craft 处理更完整的视觉故事、方法图或 PPT deck，Nature Skills 负责英文图注和风格一致性。

## 使用前核查

1. 先保存原文和结果版本，要求润色“保持数值、限定词、因果方向和术语不变”。
2. 对每处改写检查是否增加了原稿没有的 claim、确定性或因果解释。
3. 对图片检查流程、箭头、标签、单位、图例和数据来源；不要让生成图替代真实实验图。
4. 阅读目标期刊最新作者指南，不能把“Nature 风格”当作投稿政策。

## 参考

- [仓库 README](../../skills/nature-skills/README.md)
- [在线技能索引](https://yuan1z0825.github.io/nature-skills/)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
