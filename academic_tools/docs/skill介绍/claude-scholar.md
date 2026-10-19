# Claude Scholar

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[Galaxy-Dawn/claude-scholar](https://github.com/Galaxy-Dawn/claude-scholar)
- **本地目录**：[`skills/claude-scholar`](../../skills/claude-scholar/)
- **项目类型**：面向计算机科学/AI 研究者的半自动研究助理
- **文章评价**：第二篇文章评为“顶级”

## 一句话定位

Claude Scholar 把研究问题、文献、代码实验、结果分析、报告、论文和项目知识管理串成可追踪流程，尤其强调 Zotero 与 Obsidian 的连接，同时保留研究者对选题、证据、假设和发表的最终判断。

## 核心功能

- **研究构思**：把模糊主题转成研究问题、研究方向、缺口和初步计划。
- **文献工作流**：从 Zotero 集合导入、组织和阅读论文，支持 DOI/标识符导入、去重和引用导出等工作流。
- **结构化笔记**：把论文转成核心贡献、方法差异、证据和可复用 claim，方便后续综合。
- **Obsidian 知识库**：按 `Sources/Papers`、`Knowledge`、`Experiments`、`Results`、`Writing` 等区域管理项目知识和研究记忆。
- **实验与分析**：跟踪 hypothesis、实验线、run history、findings 和 next actions，使用 `results-analysis` 生成统计和科学图，使用 `results-report` 生成实验报告。
- **写作与发表**：辅助文献综述、论文段落、审稿回复、slides、poster 和发表后材料。
- **辅助工程能力**：包含 daily coding、架构设计、bug detective、LaTeX 模板和安全 guard 等配套 skill。
- **多平台分支**：主分支面向 Claude Code，也有 Codex CLI、Kimi Code CLI 和 OpenCode 分支；不要混用不同分支的安装说明。

## 适用场景

适合需要频繁在文献、代码、实验结果和论文之间切换的 CS/AI 研究者、研究生和研究工程师。需要持久文献库或项目知识库时价值更高；只做一次简单改写时可只选单个 skill。

## 优势

1. Zotero、Obsidian、实验记录和写作阶段之间有明确的数据流。
2. 项目定位清楚：半自动研究助理，不宣称替代研究者。
3. 安装支持全量、最小化和选择性模式，便于控制权限与上下文量。
4. README 将关键判断留给人：哪些问题值得做、哪些论文重要、哪些结果可信以及是否投稿。

## 局限与风险

- 重点偏 CS/AI 和软件型研究，其他学科需要额外 skill 或自定义流程。
- Zotero、Obsidian、MCP、Python 和不同 CLI 有独立配置；装好仓库不等于集成已可用。
- 本地 Markdown 知识库与已有文件可能需要人工合并，不能默认安装器会理解你的现有结构。
- 自动摘要、引用和实验分析仍可能错；结构化笔记只是提高可追溯性，不是事实证明。

## 文章评价

第二篇文章评为“顶级”，重点称赞批量导入 Zotero、阅读论文、生成结构化文献笔记和标注方法差异/引用关系的效率。文章作者的体验说明它适合文献苦力活，但连接 Zotero 或 Obsidian 时仍要检查权限、去重和原文来源。

## 推荐组合

- 用 Claude Scholar 管理论文和研究记忆，再用 ARS-Codex 做引用/证据门禁。
- 用 Scientific Agent Skills 或 AI Research Skills 运行真实实验，把日志和结果写回知识库。
- 用 PaperSpine 或 Research Paper Writing Skills 处理贡献、章节逻辑与投稿回复。

## 使用前核查

1. 先确认使用的宿主和分支，优先从最小安装开始。
2. 配置 Zotero MCP 或 Obsidian 前，检查本地权限、库路径、备份和隐私范围。
3. 对论文摘要、方法差异、引用关系和 claim 回到原文核对。
4. 分清 `Sources/Papers` 的原始材料和 `Knowledge/Writing` 中已经提升的判断，避免把推断当事实。

## 参考

- [仓库 README](../../skills/claude-scholar/README.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
