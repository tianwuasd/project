# Academic Research Skills（ARS-Codex）

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[Imbad0202/academic-research-skills-codex](https://github.com/Imbad0202/academic-research-skills-codex)
- **本地目录**：[`skills/academic-research-skills-codex`](../../skills/academic-research-skills-codex/)
- **项目类型**：Codex 原生学术研究套件；是 Claude Code ARS 的独立 Codex 分发版
- **文章评价**：第二篇文章评为“夯”

## 一句话定位

ARS-Codex 把学术研究拆成一个可路由的 `academic-research-suite` skill，覆盖深度研究、论文写作、同行评审、端到端论文流水线和实验代理五类工作流，强调人类确认、证据边界和研究完整性。

## 核心功能

- **deep-research**：细化研究问题、文献综述、系统综述、元分析、事实核查和缺口分析。
- **academic-paper**：从素材生成论文结构、章节草稿、摘要、引文格式和 AI 使用披露。
- **academic-paper-reviewer**：模拟同行评审、形成编辑决定信、复审和修改建议。
- **academic-pipeline**：把选题、研究、写作、审查、修改和最终检查串成有阶段的流水线。
- **experiment-agent**：规划代码实验或人类研究，解释统计结果，并定义可复现性检查。
- **证据与引用控制**：README 明确要求不能验证的引用、数据或期刊政策标记为未验证，不得编造；显式程序化引用门禁可使用 Crossref、OpenAlex、Semantic Scholar，并在有 arXiv ID 时查询 arXiv。
- **Codex 适配**：以一个路由 skill 代替多个自动发现的 Claude skill，支持 `ars-plan`、`ars-reviewer`、`ars-full` 等意图别名。

## 适用场景

适合需要从研究问题推进到论文或实验的研究生、科研工程师和需要结构化审稿流程的作者，尤其适合使用 Codex CLI、希望保留人工确认节点的任务。它不是只做句子润色的工具，而是研究过程编排器。

## 优势

1. 工作流边界清楚，五类研究路径和路由规则写在仓库中。
2. 将引用、证据、审稿、修订和可复现性作为完整性关卡，而不是最后才提醒。
3. Codex 版本明确区分了 Claude 插件行为、代理调用、hooks 和外部服务权限，适合审慎使用。
4. 支持从“只有题目”开始的 Socratic narrowing，要求研究问题收敛后再进入大纲或草稿。

## 局限与风险

- 当前仓库是 Codex 分发版，不能假设其 Claude Code 插件命令、hooks 或 agent-team 行为会原样工作。
- 文章提到的“四库交叉验证”是体验性描述；实际验证范围取决于引用类型、网络、缓存和脚本触发条件，仍需打开原始文献确认。
- 研究流程很重。只想快速改一段文字时，使用完整 pipeline 会产生不必要的开销。
- 外部联网、跨模型审查和脚本调用可能涉及数据上传或凭据，使用前应确认同意范围。

## 文章评价

第二篇文章把 Academic Research Skills 评为“夯”，重点称赞其对 AI 虚构引用的阻断和多来源核查。该评价应理解为文章作者的实测印象，不是本目录独立复现的结论。

## 推荐组合

- 先用 ARS-Codex 的 `deep-research` 收敛研究问题和证据范围，再用 Claude Scholar 管理文献。
- 用 `academic-paper-reviewer` 做结构化审查，再用 PaperSpine 的贡献/投稿关卡组织最终稿。
- ARIS 适合无人值守推进，但可在早晨把产物交给 ARS-Codex 做引用、证据和审稿复核。

## 使用前核查

1. 明确宿主是 Codex CLI 还是 Claude Code，不要把两个发行版混装。
2. 阅读 `skills/academic-research-suite/SKILL.md`、`ars/docs/SETUP.md` 和 `ars/docs/ARCHITECTURE.md`。
3. 对每条关键引用记录 DOI、原文 URL 或数据库结果；无法验证的条目保持未验证状态。
4. 任何跨模型审查或外部 API 调用先确认模型、数据范围、费用和上传权限。

## 参考

- [仓库 README](../../skills/academic-research-skills-codex/README.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
