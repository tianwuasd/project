# 学术工作 Agent 调研与多轮查验记录

日期：2026-09-07。性质：方案依据及静态审查，不是已完成产品的测试报告。

[架构方案](D:/code_project/academic_tools/docs/superpowers/specs/2026-09-07-academic-agent-design.md) · [建设计划](D:/code_project/academic_tools/docs/superpowers/plans/2026-09-07-academic-agent-roadmap.md) · [接口探测摘要](D:/code_project/academic_tools/docs/academic-agent/2026-09-07-source-probes.json)

## 1. 本轮调查范围

主审读取当前本地目录结构及代表文件；独立审查者深入抽查 ARS-Codex、ARS、PaperSpine、ARIS、tech_information，浅查 Claude Scholar 和 RadioPath。网络调查优先项目官方仓库和接口官方文档。没有安装这些项目、执行其脚本、运行其测试、复制其代码、读取密钥或调用付费模型。

判断等级：

- **源码核实：** 读到了实现或约束；不代表实际运行成功。
- **官方文档核实：** 官方页面明确描述；本机可用性还需实测。
- **单次连通抽查：** 真实公共请求及响应状态；不代表批量服务质量。
- **设计建议：** 根据以上材料提出的本项目工程选择；不是来源声称的产品事实。

## 2. 本地可复用资产

| 资产 | 核实证据 | 推荐复用方式 | 不能直接假定的能力 |
|---|---|---|---|
| ARS-Codex | [路由脚本](D:/code_project/academic_tools/skills/academic-research-skills-codex/skills/academic-research-suite/codex/scripts/ars_codex_full_runtime.py:4) | 参考技能路由、工作流和 reviewer 角色分配 | 脚本说明它不派生 agent、不执行 hook；只是确定性规划器 |
| ARS 证据契约 | [证据处理](D:/code_project/academic_tools/skills/academic-research-skills/scripts/evidence_rows.py)、[schema](D:/code_project/academic_tools/skills/academic-research-skills/shared/contracts/evidence/evidence_row.schema.json) | 参考主张、定位、摘录、哈希和核验状态分离 | 摘录原文匹配不等于支持某个论断 |
| ARS 文献发现 | [四源发现脚本](D:/code_project/academic_tools/skills/academic-research-skills/scripts/claim_standing_discovery.py) | 参考来源适配和查询接口；重新做实测和规范化 | 源码声明尚未 live 验证；相关性和无 DOI 作者字段映射仍需完善 |
| ARS 单篇核验 | [S2 客户端](D:/code_project/academic_tools/skills/academic-research-skills/scripts/semantic_scholar_client.py)、[中文核验](D:/code_project/academic_tools/skills/academic-research-skills/scripts/chinese_literature_client.py) | 参考 DOI 优先、限流退避、中文查不到不等于伪造 | 单篇核验器不是关键词检索器；中文核验尚未全部接入总流程 |
| ARIS 状态与收据 | [run_state.py](D:/code_project/academic_tools/skills/Auto-claude-code-research-in-sleep/tools/run_state.py:35)、[idea_discovery_gate.py](D:/code_project/academic_tools/skills/Auto-claude-code-research-in-sleep/tools/idea_discovery_gate.py) | 参考阶段完成与审核接受分离、审核收据、恢复状态 | 文件锁依赖 POSIX fcntl，Windows 下退化；本项目用事务和单写者 |
| PaperSpine 写作检查 | [结果检查](D:/code_project/academic_tools/skills/PaperSpine/src/scripts/results_validation_check.py)、[引用表检查](D:/code_project/academic_tools/skills/PaperSpine/src/scripts/citation_bank_check.py) | 参考贡献—结果映射、写作工序和结构检查；消费真正证据 ID | 非空 evidence 字段不证明实验真实或语义支持；不混用 src/dist/RC 版本 |
| tech_information | [Source 接口](D:/code_project/tech_information/internal/collector/source.go)、[arXiv 采集](D:/code_project/tech_information/internal/collector/arxiv.go)、[主入口](D:/code_project/tech_information/cmd/techinfo/main.go:23) | 参考来源接口、错误隔离、并行收集 | 主入口的运行流水线和调度仍未实现；现有 Article 不足以追踪证据和版本 |
| Claude Scholar | [Research Contract](D:/code_project/academic_tools/skills/claude-scholar/skills/research-ideation/references/research-contract.md) | 参考问题卡、证据卡、允许/禁止的论断措辞 | 本次只核实文档；main 的 Claude Code 行为不保证在 Codex 原样可用 |

`RadioPath` 的实际内容是影像/病理模型工程，可作为以后接入真实实验产物的案例，不能当通用文献 agent 运行底座。[项目说明](D:/code_project/RadioPath/README.md)

主审二次抽查确认：ARS-Codex 脚本开头的“只规划”声明、tech_information 主入口两处未实现标记、ARIS 在 Windows 下的无锁分支，以及 ARS/PaperSpine 根许可证。其余细节来自独立审查者的只读报告，不包装成已执行结果。

## 3. 本地许可证边界

以下是当前根 LICENSE 文件标识，不是对全部依赖或附带数据的法律审计。

| 本地仓库 | 根许可标识 | 方案处理 |
|---|---|---|
| academic-research-skills | CC BY-NC 4.0 | 方法参考；复制内容前明确使用/分发范围及署名 |
| academic-research-skills-codex | CC BY-NC 4.0 | 作为可选技能层，避免混入计划中的可分发核心 |
| PaperSpine | MIT | 优先评估其小型结构检查组件，保留许可及来源 |
| Auto-claude-code-research-in-sleep（ARIS） | MIT | 参考状态和检查设计，平台适配另做 |
| claude-scholar | MIT | 参考研究卡和流程，逐项登记 vendored 内容 |
| tech_information / RadioPath | 顶层未发现 LICENSE | 确认归属和复用范围后再复制/发布 |

直接证据：[ARS LICENSE](D:/code_project/academic_tools/skills/academic-research-skills/LICENSE:4)、[ARS-Codex LICENSE](D:/code_project/academic_tools/skills/academic-research-skills-codex/LICENSE:4)、[PaperSpine LICENSE](D:/code_project/academic_tools/skills/PaperSpine/LICENSE:1)、[ARIS LICENSE](D:/code_project/academic_tools/skills/Auto-claude-code-research-in-sleep/LICENSE:1)、[Claude Scholar LICENSE](D:/code_project/academic_tools/skills/claude-scholar/LICENSE:1)。

## 4. 网络开源项目比较

已阅读各项目官方仓库；下表前两列描述文档可确认能力，第三列是本项目的选型判断。未安装运行，也未根据 Star 数排序。

| 项目 | 官方资料体现的能力 | 本方案取舍 |
|---|---|---|
| [PaperQA2](https://github.com/Future-House/paper-qa) | 科学文档 RAG、证据检索和引用问答，多后端；Apache-2.0 标识 | 借鉴证据检索/上下文重排，可作为评测基线或可选问答引擎。自建的检索日志、主张库和文献获取仍需独立存在 |
| [OpenScholar](https://github.com/AkariAsai/OpenScholar) | 科学检索、重排、自反馈生成，附 ScholarQABench 入口；Apache-2.0 标识 | 借鉴检索—修订—引用归因和基准设计，第一版不训练模型或自建庞大论文索引 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | 网页/本地资料研究、规划与执行角色、报告和来源追踪；Apache-2.0 标识 | 参考任务分解和报告交互；学术论文身份、版本及主张支持核验由本项目补足 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 状态化工作流、持久执行及人工介入；MIT 标识 | 作为唯一顶层编排候选，业务数据独立；不要求云端部署 |
| [Docling](https://github.com/docling-project/docling) | 文档结构、PDF 版式、OCR、表格、结构化导出、本地运行；MIT 标识 | 可替换解析器候选，先对实际论文测解析与定位质量 |
| [AI Scientist](https://github.com/SakanaAI/AI-Scientist) | 基于实验模板的想法、实验和论文生成；原仓库描述 Linux/NVIDIA GPU 环境 | 作为后期实验闭环参考。其领域和环境假设不适合作为当前 Windows 通用文献 MVP 底座 |

没有采用上游宣传的“超越人类”等表达作为选型证据。各项目的论文结果与用户将来的学科、语料和模型不同，需在同一套任务中比较。

## 5. 官方接口与 Codex 核实

| 事实 | 来源 | 对计划的影响 |
|---|---|---|
| Codex 现有稳定 Python SDK，通过本地 app-server 通信，发布包含固定运行时依赖 | [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk) | Python 后端可优先验证；不再默认额外建设 TS 桥接 |
| Codex SDK 页面把 codex mcp-server 标为弃用 | [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk) | 新系统不以该旧接口作底座；项目自己的工具 MCP 是另一概念 |
| codex exec 支持 JSONL 事件与 JSON Schema 最终输出 | [非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode) | 保留独立进程适配路径，结果仍需业务校验 |
| Agents SDK 提供局部 agent 循环、工具和交接等能力；Responses API 可自管循环 | [Agents SDK](https://developers.openai.com/api/docs/guides/agents) | API 后端选一种局部运行方式，避免双顶层调度 |
| OpenAlex 当前允许基本无密钥使用，免费密钥提高预算；配额及分页规则有更新 | [认证文档](https://help.openalex.org/api/authentication/) | 不能照搬旧文档的免密钥/固定大配额假设；运行时读取当前规则 |
| Crossref 返回元数据及出版后更新，含可信来源补充 | [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | 用于身份与出版状态；不能替代全文语义核验 |
| arXiv 官方手册说明 Atom/XML、分页和连续请求延迟建议 | [官方手册](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md) | 保留版本、独立限流、失败降级 |
| Unpaywall 的官方 API 文档搜索索引说明需要 email 参数 | [Unpaywall API](https://unpaywall.org/api/v2) | 可选 OA 位置查询；无用户邮箱时不伪造身份调用 |

资料边界：Unpaywall 直接网页提取只返回 JavaScript 提示，Semantic Scholar 直接产品页提取失败、API 文档页只有导航。本轮仅以官方搜索索引支持基础接口描述，并以实际 S2 查询验证最低连通；没有把具体配额作为已实测事实。正式连接器实现时须再读可用官方 schema 和账户说明。

查询中出现的 OpenScholar 错误组织地址已更正为作者官方仓库 `AkariAsai/OpenScholar`；PaperQA2 以 `Future-House/paper-qa` 为准，不使用搜索结果中的第三方 fork。

## 6. 真实公共接口抽查

时间：2026-09-07 19:22:37（Asia/Shanghai）。临时探测词为 retrieval augmented generation，仅用于返回结构与连通性抽查，不替用户选择研究题目。

四个目标都按本地规则分类为国外服务，经 127.0.0.1:7897；代理监听已确认。每来源一条无密钥小请求，没有下载论文或调用模型。

| 来源 | 首次结果 | 用时 | 结论 |
|---|---|---|---|
| OpenAlex | HTTP 200，JSON，1 条记录 | 2.22 秒 | 当前匿名小查询可用 |
| Crossref | HTTP 200，JSON，1 条记录 | 1.73 秒 | 当前小查询可用 |
| Semantic Scholar | HTTP 200，JSON，1 条记录 | 5.07 秒 | 当前匿名小查询可用 |
| arXiv | 读取超时，无 HTTP 状态 | 18.31 秒 | 当前链路未验证成功 |

arXiv 经同一路由用 Windows curl 重试时，TLS 客户端在获得 HTTP 响应前报 `SEC_E_NO_CREDENTIALS`。这说明重试客户端的本地环境有问题，不能推断 arXiv 服务宕机；没有关闭证书验证或绕过既定路由。

[机器可读摘要](D:/code_project/academic_tools/docs/academic-agent/2026-09-07-source-probes.json)保存 URL、状态和限制，未保存完整原响应。成功的三条请求不能证明召回率、相关性、全文覆盖率、批量配额、稳定性或未来可用性。

## 7. 第一轮查验：需求与来源可行性

**已完成。** 主审和本地独立审查者完成目录/源码抽查、官方资料核对及四个接口的小请求抽查。

| 发现的问题 | 对方案的实际修订 |
|---|---|
| 本地部分“运行器”仅生成计划 | 用独立研究内核执行，技能库只提供方法与约束 |
| tech_information 的流水线和调度未实现 | 限定为采集接口参考，不作为已完成应用复用 |
| Windows 文件锁差异 | 使用唯一提交者和 SQLite 事务，加入中断/并发验收 |
| DOI 和摘录匹配易被当成语义正确 | 四层引用检查，原文支持关系单列 |
| 来源与 Codex 官方能力已变化 | 优先验证官方 Python SDK；适配器读取当前来源规则 |
| 单来源会超时 | 明确 partial/unavailable 状态和独立降级 |
| 非商业许可与 MIT 混杂 | 建立资产登记，核心与可选内容分离 |

## 8. 第二轮查验：独立架构与学术可靠性审查

**已完成并复核修订。** 第二位独立审查者只读检查架构稿与计划，提出 7 项实质问题；主审修订后，该审查者针对原问题再次检查，结论为 7 项已在设计层处理。这不是运行或产品测试结论。

| 级别 | 问题 | 0.2 版采取的修订 | 建设落点 |
|---|---|---|---|
| P1 | 撤稿/更正的首次检查缺少 MVP 落点 | 首次入库及导出前检查，状态未知/过期单列；定期监测仍在后期 | M1、M2 |
| P1 | 取消或旧输入的迟到结果可能提交，文件与事务间存在窗口 | 验证 attempt/租约/项目修订号；不可变文件先落盘核哈希，再登记数据库；增加两类崩溃用例 | M2 |
| P1 | 费用未知时可能错误释放预留并再次花费 | 四种账本状态、待对账继续占用、重试另预留；Codex 不透明用量采用次数/耗时控制 | M0、M2 |
| P1 | 只评主张正确率会奖励过度删减 | 加人工必覆盖发现/局限、真实综述的分歧覆盖、错误拒答指标，支持率与覆盖率同时验收 | M2 |
| P2 | 审查者仍被生成者所选摘录限制 | 提供确定文档版本和独立只读取文能力，记录实际读取范围 | M2 |
| P2 | 文献去重不代表研究统计独立，快照冻结不足 | 默认去重文献组计数；冻结纳排、版本、家族裁决、日期和标签；疑似重复做敏感性分析 | M3 |
| P2 | 允许工具列表可能只有提示词约束 | 记录权限执行位置，构造负向测试；能力不能强制限制时降级或停止 | M0 |

独立审查还明确保留：本机权限、来源批量限额、PDF 质量、语义审查能力、标注投入及多 agent 增益均需实际验证。没有把设计修订等同于消除现实风险。

## 9. 第三轮查验：用户流程、覆盖与交付一致性

**主审自查已完成。** 按“一个关键词项目首次运行、部分全文缺失、来源失败、处理途中退出、人工改动结果、改题目、重新计量”逐项检查设计和计划是否有相应入口、状态、产物及验收例。

自查额外发现：趋势等派生主张不能全部硬套单一原文摘录。已加入 `ComputationRecord`，连接冻结语料、公式/代码、参数、结果表及哈希，使统计结论可复算且能回到输入来源。

| 用户需求 | 方案位置 | 计划位置 | 交付时状态 |
|---|---|---|---|
| 多渠道关键词汇总论文 | 架构 §3–4、§6 | M1 | 有来源分工、去重、分页、失败与导入设计 |
| 论文总结、比较和问答 | 架构 §3、§8.1 | M2 | 有原文位置、读取范围和语义支持规则 |
| 趋势分析 | 架构 §8.2 | M3 | 有语料口径、计算记录、样本限制及敏感性检查 |
| 创新点提取及研究空白 | 架构 §8.3 | M3 | 作者贡献与候选分开，近邻、反证及验证方案明确 |
| 搜索到写作闭环 | 架构 §2、§9 | M1–M2；深入写作 M4 | 第一版到有引用综述，真实结果写作有输入前提 |
| Codex 或其他 API | 架构 §5 | M0、M2 | 可替换后端及能力探测，未宣称已运行 |
| 多 agent 协作 | 架构 §7、§10 | M2 | 有独立输入、统一提交、停止/恢复和基线比较 |
| 扩展性与题目特化 | 架构 §11 | M4 | 数据源、分析能力、领域包有独立边界 |
| 本地与开源参考、多轮查验 | 本记录 §2–8 | M0 与各阶段验收 | 已做本地只读审计、官方资料核对、真实小请求及独立审查 |

交付核对包含：三份 Markdown 的本地链接目标存在、代码围栏配对、临时审查占位段清除、探测 JSON 可解析；接口摘要确为 4 条探测、3 条 HTTP 200、未使用凭据。正文中的质量百分比和恢复用例都明确是未来验收目标，没有写成已经达到的结果。

文档级检查不能证明外链长期可达，也不能代替 M0–M4 的运行测试。若用户选择其他第一版优先级，保留本轮来源与审查记录，并据此调整里程碑顺序。

## 10. 尚未证明的事情

- Codex SDK 或任一模型 API 在本项目中的实际运行、登录继承、结构化输出和取消效果。
- 各文献源的批量分页、持久限额、中文召回和本机长期可用性。
- PDF 表格/公式/OCR 的真实准确率及证据位置质量。
- 多 agent 相对简单单 agent 的质量增益、运行时间和费用。
- 趋势、创新候选和写作在用户具体领域是否足够有用。

这些都已安排进 M0–M4 的验收；本轮不把计划审查表述为产品通过测试。
