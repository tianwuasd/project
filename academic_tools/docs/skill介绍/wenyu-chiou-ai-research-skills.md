# WenyuChiou AI Research Skills

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[WenyuChiou/ai-research-skills](https://github.com/WenyuChiou/ai-research-skills)
- **本地目录**：[`skills/WenyuChiou-ai-research-skills`](../../skills/WenyuChiou-ai-research-skills/)
- **项目类型**：围绕研究缺口、证据边界和跨工具协作的 Claude Code skill catalog
- **仓库 README 当前标注**：17 个 skill、5 个插件（catalog v1.7.7）；提交树实际有 0 个 `SKILL.md`，具体技能在其他源仓库中。
- **文章评价**：第一篇文章介绍了该项目，但没有给出分档；第二篇文章未重点测评

## 一句话定位

这个项目首先追问“研究缺口是否真的值得做”，再让文献、设计、实验、写作和投稿沿着可检查的阶段流转。它的特色不是单个算法，而是用 manifest、YAML schema 和 gate 让研究状态跨会话、跨工具保留下来。

## 核心功能

- **研究选题与缺口**：通过 open、contribution、feasibility 三道 gate 形成 topic decision dossier，过滤已被解决、贡献不足或不可执行的题目。
- **文献阶段**：`research-hub`、`paper-summarize`、`literature-triage-matrix`、`zotero-library-curator` 等用于发现、筛选、摘要和组织文献。
- **研究设计**：`research-design-helper` 将通过缺口门禁的候选转为研究问题、设计 brief 和可执行计划。
- **项目状态**：`research-context-compressor`、`research-project-orienter` 和 manifest 记录研究问题、基线、实验、结果和下一步，降低新会话重新解释成本。
- **写作与证据**：`paper-memory-builder`、`academic-writing-skills` 生成论文资产；claim 没有证据时必须保持 gap 状态，不能写成 supported。
- **跨 AI 交接**：按任务把机械工作、长上下文工作和判断/治理工作交给 Codex、Gemini 或 Claude，并通过 coordination 文件交接。
- **可选运行时**：仓库把 prompt/skill catalog 与 `research-hub` Python runtime 分开，避免把纯推理 skill 和联网自动化混为一谈。

## 适用场景

适合研究生、博士后、研究支持人员，以及经常在 Claude、Codex、Gemini 之间切换的计算型科研项目。尤其适合课题刚起步、研究缺口不清晰、需要跨会话恢复或团队协作的任务。

## 优势

1. 把研究缺口、证据和可行性放在写论文之前，减少“先写再发现题目不成立”。
2. 产物以 YAML/Markdown manifest 落盘，跨会话和跨工具恢复比临时聊天更稳定。
3. schema 的 anti-leakage 规则会阻止无证据 claim 被标记为 supported。
4. 插件分层安装，可先使用研究工作区，再按需要添加写作、Zotero 和多 CLI 委托。

## 局限与风险

- 项目包含 catalog、插件和可选 Python runtime，安装时要区分只加载 `SKILL.md` 与运行 `research-hub`。
- gate 可以约束流程，但不能证明研究缺口或因果结论本身正确；领域专家判断仍不可省略。
- Zotero、Codex、Gemini、Python 和外部 API 都有各自的权限、版本和配置要求。
- 仓库 README 的 skill 数量与插件列表可能随上游更新变化，不能把当前统计当成固定版本承诺。

## 文章评价

第一篇文章把该项目描述为解决跨会话状态、幻觉引用、模糊 gap 分析、AI 味文风和跨工具交接的实战型项目，但没有像第二篇文章那样给出等级。这里保留其问题导向的观察，不将其扩展为未经验证的质量排名。

## 推荐组合

- 用本项目做选题和项目状态，用 Claude Scholar 做 Zotero/Obsidian 文献管理。
- 用 Scientific Agent Skills 或 AI Research Skills 执行真实数据/模型工作，再把日志和结果写回 manifest。
- 用 PaperSpine 做贡献优先的最终稿，用 ARS-Codex 做引用和证据门禁。

## 使用前核查

1. 先阅读 `docs/pipeline.md`、`docs/runtime-contract.md` 和 `docs/install.md`，明确哪些功能需要 runtime。
2. 检查 `.research/`、`.paper/` 和 YAML schema 是否进入版本控制并可被团队共享。
3. 对 `worth-pursuing`、`supported` 等状态逐项确认其来源和判定理由。
4. 外部委托前确认数据、文件、API 密钥和跨模型传输范围。

## 参考

- [仓库 README](../../skills/WenyuChiou-ai-research-skills/README.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
