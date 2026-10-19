# 科研 Skill 详细使用文档

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

本目录是 [skill介绍](../skill介绍/) 的深化层：介绍页回答“它是什么、适合什么”，本目录回答“怎样安装、怎样输入、怎样分阶段使用、怎样核查输出”。

## 已完成轮次

### 第 1 轮：核心研究工作流

- [Academic Research Skills（Claude Code 版）](academic-research-skills-claude/00-原始完整说明.md)
- [ARS-Codex](ars-codex/00-原始完整说明.md)
- [Claude Scholar](claude-scholar/00-原始完整说明.md)

### 独立目录第 1 轮：核心研究工作流专题文档

以下三个 skill 已建立独立文件夹；每个文件夹包含功能说明、详细使用指南、源码结构与构建、本地使用指南、验证与风险，以及目录导航 README。

- [Academic Research Skills（Claude Code 版）专题目录](academic-research-skills-claude/)
- [ARS-Codex 专题目录](ars-codex/)
- [Claude Scholar 专题目录](claude-scholar/)

下一轮候选：Scientific Agent Skills、AI Research Skills、Auto-Empirical Research Skills。

### 独立目录第 2 轮：科学计算与实证专题文档

- [Scientific Agent Skills 专题目录](scientific-agent-skills/)
- [AI Research Skills 专题目录](ai-research-skills/)
- [Auto-Empirical Research Skills 专题目录](auto-empirical/)

本轮每个目录均包含六个文件，源码分析覆盖真实 `SKILL.md`、路由/插件入口、验证脚本和依赖边界；本地指南只引用当前项目源码，不做完整安装。

### 独立目录第 3 轮：论文生产与视觉表达专题文档

- [PaperSpine 专题目录](paper-spine/)
- [Research Paper Writing Skills 专题目录](research-paper-writing-skills/)
- [Paper Craft Skills 专题目录](paper-craft-skills/)

本轮每个目录均包含六个文件；PaperSpine 聚焦贡献和证据链，Research Paper Writing Skills 聚焦章节写作，Paper Craft 聚焦图解、HTML 分析和 deck。

### 独立目录第 4 轮：表达、自动推进与研究治理专题文档

- [Nature Skills 专题目录](nature-skills/)
- [ARIS 专题目录](auto-claude-code-research-in-sleep/)
- [WenyuChiou AI Research Skills 专题目录](wenyu-chiou-ai-research-skills/)

本轮特别记录了 Nature/ARIS 的外部工具与自动推进边界，以及 WenyuChiou 当前快照没有 `SKILL.md`、只能作为 registry 审计的事实。

### 第 2 轮：计算与实证工作流

- [Scientific Agent Skills](scientific-agent-skills/00-原始完整说明.md)
- [AI Research Skills](ai-research-skills/00-原始完整说明.md)
- [Auto-Empirical Research Skills](auto-empirical/00-原始完整说明.md)

### 第 3 轮：论文生产与视觉表达

- [PaperSpine](paper-spine/00-原始完整说明.md)
- [Research Paper Writing Skills](research-paper-writing-skills/00-原始完整说明.md)
- [Paper Craft Skills](paper-craft-skills/00-原始完整说明.md)

### 第 4 轮：表达、自动推进与研究治理

- [Nature Skills](nature-skills/00-原始完整说明.md)
- [ARIS](auto-claude-code-research-in-sleep/00-原始完整说明.md)
- [WenyuChiou AI Research Skills](wenyu-chiou-ai-research-skills/00-原始完整说明.md)

每篇指南都使用同一结构：定位、适用边界、前置条件、安装、输入准备、逐步操作、可复制提示词、预期产物、结果核查、失败与风险、与其他 skill 的衔接、来源。

## 增量推进规则

- 每一轮只新增或完善 2–3 个 skill 的详细指南；本轮结束先核验文件和链接，再开始下一轮。
- 后续文件名使用小写短横线，例如 `scientific-agent-skills.md`、`paper-spine.md`。
- 介绍文档保持稳定的快速选型信息；详细文档负责版本、命令、输入输出和操作变化。
- 功能描述以本地仓库的 README、`SKILL.md` 和仓库文档为准；找不到的信息写明“仓库资料未说明”，不补造能力。
- 文章评价是体验性参考，不能替代代码、数据、引用和实验核查。
- 本目录是使用说明，不是自动安装脚本；命令执行前要确认宿主、权限、网络、密钥和作用域。

## 统一核查清单

使用任何指南后，至少确认：

- [ ] 使用的是正确的宿主和上游分支。
- [ ] 论文、数据、引用、代码和实验结果都有来源或明确的待验证状态。
- [ ] Agent 没有新增输入材料中不存在的数字、因果关系、作者、DOI 或结论。
- [ ] 生成的 Markdown、LaTeX、Word、PPT、脚本和日志已保存到明确位置。
- [ ] 对外发送、跨模型审查、联网检索和写入文件前已确认权限与数据范围。

## 当前状态

两篇文章中列出的 12 个科研 skill 均已有对应的详细使用文档；第 1–4 轮的 12 个 skill 已进一步建立独立专题目录。本目录后续如需更新，应按版本变化、实际使用反馈或新增 skill 开启新的增量轮次，不再一次性扩写全部内容。
