# 创新候选生成与依据审查

本模块从当前已纳入、可读且至少有一条已支持陈述的研究快照生成至多三个研究候选。候选始终是
`statement_type="system_hypothesis"` 的系统假设；它不是论文的 `result` 陈述，也不表示全球首次、
已证明新颖或实验已经成功。作者贡献原文在模型上下文中保留为 `author_contribution`，不会被宿主
改写成系统已经验证的创新判断。

## 公共接口

```python
batch = generate_candidates(
    snapshot, backend, timeout_seconds=300, max_input_chars=100_000
)
review = review_candidates(
    snapshot, batch, backend, timeout_seconds=300, max_input_chars=100_000
)
result = create_innovations(runner, work_ids=None)
```

`InnovationCandidate` 包含宿主生成的稳定 `id`、固定陈述类型、研究问题、最接近工作与陈述 ID、
具体差异、作为依据的陈述与证据 ID、可能反例、可证伪假设、最低验证、数据和算力条件、风险、
一至两条定向查询及材料截止。所有文字和引用列表非空，引用 ID 必须组成当前已支持的
工作→陈述→证据链。`CandidateBatch` 固定使用 `innovation-generation-v1`，保存研究输入指纹和一至
三个候选。候选 ID 由输入指纹与规范化候选内容产生，不使用捕获时间、后端随机 ID 或运行预算。

`GroundingReview` 固定使用 `innovation-grounding-review-v1`，每个候选必须恰有一个独立审查项。
审查结论为 `supported`、`partial`、`contradicted` 或 `insufficient`，并保存理由及经过宿主校验的
陈述和证据引用。这里的 `supported` 只表示候选关于所给论文的依据能在本次材料中找到支持，绝不
证明候选新颖或未执行实验的结果。缺项、重复项、外来候选或外来引用会使整次审查验证失败，系统
不会自动把它们修补为成功输出。

## 模型输入和恢复

生成和审查是两个独立的 `ModelJob(kind="synthesize")`，分别经过独立的
`MeteredBackend.validated()` 边界。生成结果完成领域校验后先缓存；如果审查因预算或 Exchange
等待暂停，以相同文献选择再次调用 `create_innovations` 会复用生成结果，只继续审查。

两个任务都收到全部所选文档页面、可引用的所选元数据以及精确的陈述—证据—决定链。检索记录只
投影查询、时间、来源、状态、计数、有效上限、截断、错误、保留 ID 覆盖和完整原事件哈希。原始
服务商响应及未选文献正文/摘要不会发送给模型，但仍保存在不可变研究快照中；原事件内容改变会
通过哈希改变稳定任务身份。原始响应体再大也不会重复占用模型输入，所选论文的完整正文如果超过
`max_input_chars` 则在调用后端前失败，不会截断。

如果生成输入或包含候选批次的较大审查输入超限，结果仍使用五个固定键并返回
`status="failed"`；`gaps` 中的安全类别为 `model_input_too_large`，同时明确提示通过已有
`work_ids` 参数选择更少文献后重试。它不会返回原始提示词或后端诊断。仅审查超限时，已经通过领域
校验的生成批次保持缓存，重试不会重复生成。

`create_innovations` 精确返回 `status`、`input_hash`、`output`、`candidate_ids` 和 `gaps`。状态含义：

| 状态 | 含义 |
|---|---|
| `complete` | 生成与独立审查均通过领域校验，研究输入仍然有效并已发布 |
| `needs_input` | Exchange 等后端需要外部结构化响应；以相同选择重试即可恢复 |
| `budget_paused` | 共享模型调用预算已用完；不会自动提高上限 |
| `stale` | 模型处理期间文档、复核或其他研究依赖改变，结果未提交为成功缓存 |
| `failed` | 当前研究输入、结构化输出、引用链或发布完整性验证失败 |

只有当前生成与审查配对完成后才更新 `analyses/innovation/latest.json`。一个完整版本固定含
`candidates.json`、`grounding-review.json`、`report.md`、`references.bib` 和 `evidence.json`。
参考文献、引用键、URL 和证据附表全部由宿主从真实所选记录构建，不采用模型返回的引用键或链接。
证据附表对所有导出的 Claim 和 Decision 引用求闭包，因此最接近工作的贡献陈述及仅由独立审查
引用的其他所选工作也都有可解析的 Work→Claim→Decision→Evidence 链。中文报告明确标注本次材料
范围和人工判断要求，非支持结论会原样保留。

Demo 后端只接受随包提供的两份原创调度 A/B 文档、对应原创元数据、每份文档中的独立标识行和
经过精确跨度验证的证据。A 的 40 个合成作业条件从 10 降至 8，B 的 60 个突发负载条件从 10
升至 11；该固定协议仅演示候选与审查机制，不代表真实论文语义准确率。任何真实、混合或被修改的
输入都会被拒绝。Codex 与 Exchange 继续使用通用的闭合 Schema 传输。

## 有界反例检索与当前线索

```python
from academic_agent.challenges import (
    candidate_currentness, challenge_candidate, load_candidate, read_challenges,
)

result = challenge_candidate(runner, candidate_id, round_number=1)
historical = load_candidate(root, candidate_id)
coverage = read_challenges(root, candidate_id)
currentness = candidate_currentness(root, candidate_id)
```

每个候选最多两轮，每轮执行候选自带的一至两条查询，每来源每查询请求最多十条，项目容量较小时
采用更小的实际上限。重复相同轮次表示恢复，成功查询不会重复发送；新第二轮会重新检索，不会被
第一轮或其他旧检索的健康缓存替代。失败查询按查询组重试，已成功的另一条查询保留原收据。
没有配置来源时状态为 `not_searched`，不会把空来源集合当作检索成功。

`Runner.search(query, cached=False, limit=None, request_id=None)` 的默认缓存键和旧 `summary` 完全保留。显式
`limit` 只接受 1–100 的整数（不接受布尔、字符串或小数），采用独立缓存身份。实际有效上限、
项目容量及保留工作 ID 保存在搜索事件的独立字段；模型输入可读取这些真实值。历史事件没有保存的
ID 或上限保持未知，返回空 ID 列表和 `retained_ids_available=False`，绝不通过标题相似度补造。
去重与容量控制继续使用既有保守规则，新增记录仍待筛选；出版状态检查仅针对实际保留的工作 ID。

宿主先写入 `challenge_start` 持久边界，再为每次派发写入 `challenge_dispatch` 与独立请求 ID，
随后执行查询。请求 ID 保存在实际搜索事件的独立字段，不能补写到旧缓存。轮次语义身份包含规范化候选内容与原批次
输入指纹、轮次、查询、来源、实际有效上限，不包含版本目录名、候选产物清单哈希或捕获时间。
清单哈希作为独立来源信息固定保存。缓存候选重新发布到另一个不可变版本不会重置轮次；重复候选
ID 的有效相同版本按宿主目录名字典序选择，但恢复继续核验该轮原来固定的候选清单。内容不一致或
已被改写的历史文件会被拒绝。

每次查询之后，以当时新的研究快照发布 `analyses/challenge/v-*/round.json` 和宿主清单，保留此前
版本。`ChallengeRound` 的版本为 `schema_version=1`、`task_version="counterexample-search-v1"`。
`receipts` 保存每条查询的当前成功或最新失败收据；`search_attempts` 保留该轮所有真实搜索尝试。
收据包含精确查询键、请求 ID、事件 ID、时间、完整事件哈希、实际上限、项目容量、逐来源状态/数量/截断/
错误、真实保留 ID、ID 可用性、容量丢弃数和完成标志。历史部分失败尝试的命中仍参与保留 ID 并集；
重试成功后，该尝试的失败不再冒充当前来源故障，历史失败仍可审计。进程在搜索记录落盘后、收据
发布前中断时，只能从该轮持久边界及派发记录之后的精确请求 ID 与查询键恢复，发布完整恢复收据后
才返回输出目录。第一轮来源失败时仍可显式开始新的第二轮，之后恢复第一轮不会误用第二轮记录。
来源配置可以在新轮次更换；恢复既有轮次时须匹配该轮固定的来源和有效上限。若失败查询尚待重试，
还须恢复原项目容量，确保精确查询键不变；更换配置不会静默创建第三轮或重置同一轮。

操作结果包含 `status`、`candidate_id`、`identity`、`round`、`queries`（当前查询收据）、
`retained_work_ids`、`unresolved_tasks` 和项目内相对路径 `output`。状态优先级如下：

| 状态 | 含义 |
|---|---|
| `source_failed` | 当前查询存在部分失败、不可用、限流或保留 ID 不可用 |
| `not_searched` | 尚有查询未实际查询任何来源 |
| `needs_reading` | 线索待筛选、待阅读、待复核、已移除或排除理由缺失 |
| `needs_assessment` | 查询操作已完成，可以在披露覆盖缺口的前提下进行独立评估 |

`unresolved_tasks` 是结构化覆盖事项，分别保留 `empty`、`partial`、`unavailable`、`rate_limited`、
`truncated`、`dropped_at_cap`、`pending_screen`、`excluded_with_reason`、`excluded_without_reason`、
`unread`、`unreviewed`、`missing_work`、`retracted`、`publication_unknown`、`abstract_only` 和
`retained_ids_unavailable`。空结果只描述本次查询范围，排除须披露理由，摘要阅读仍披露访问层级；
`ready` 只表示线索已完成阅读与复核，不表示它是已确认反例，更不表示没有其他反例。

检索覆盖按每个已开始轮次独立计算后汇总，不会因为上一轮相同查询已成功而补齐新一轮尚未检索的
查询。检索事项的 `round` 为对应的 1 或 2；全局线索事项和尚未开轮的事项为 `null`。旧版本没有
此可选字段时仍可读取。每轮检索事项保留轮次上下文，当前线索事项在顶层只汇总一次。

`load_candidate` 精确返回经过纯验证器核验的 `candidate`、`batch`、`review`、`snapshot`、宿主
`version_path` 和 `manifest_hash`。候选与审查的输入哈希、宿主 ID 及引用链均绑定原清单快照。
`validate_candidate_batch` 与 `validate_grounding_review` 同时服务于模型输出和历史读取，读取不会
运行模型或创建可写 Store。历史设想允许查看；检索前另外检查原来所选文献仍纳入、未撤稿，原文、
证据、阅读卡、复核、核心来源元数据及研究问题/项目标题/项目查询没有变。配置的检索来源由各轮
独立固定，来源配置调整不会改写原论文依据。新线索、追加搜索日志、出版检查记录
或新线索完成阅读不会强迫重新生成旧候选；原始依据真正改变则拒绝检索，要求核对过期依据。

`read_challenges` 返回类型化 `rounds`、项目内相对路径 `outputs`、类型化当前 `hits`、
`unresolved_tasks` 和操作 `status`。每个 Hit 包含 `work_id`、`statuses`、`screening_reason`、
`access_level`、`publication_status`。它通过 URI 编码的 SQLite `mode=ro` 连接和研究快照共同
读取当前已提交状态，支持文件名中的 `#`/`%` 与 WAL；不会从旧收据推断当前筛选状态。
访问层级独立读取并校验当前活动文档及其原始文件，不受研究快照只选择纳入可读资料的限制：已附
全文的待筛选或排除线索仍显示 `full_text`，无文字层且需 OCR 的活动文档显示 `needs_ocr`。筛选
与阅读事项仍分别显示待筛选、排除理由或未读，不会因有附件就变为已阅读。读取核验文档哈希、
文档/工作身份、项目内安全路径和前后文档及工作记录的一致性。
因此待筛选→有理由排除、纳入→待阅读→待复核→就绪都会在状态查询中更新。旧版本仍然保存当时
覆盖状态；相同轮次再次运行时如当前覆盖已变，会发布新的覆盖版本，而不重复已成功的网络查询。

`candidate_currentness(root, candidate_id)` 精确返回 `candidate_id`、`current` 和 `gaps`，与检索
调用共享同一原始依据有效性检查。它独立于反例覆盖读取，不创建 Store，也不泄露底层文件或模型
诊断；安全缺口类别为 `candidate_artifact_invalid`、`candidate_evidence_invalid`、
`candidate_scope_changed`、`candidate_evidence_changed`、`candidate_source_changed`。历史候选可以
读取不等于当前依据仍有效，评估与状态调用应分别读取候选当前性和反例覆盖。

## 材料范围内的候选评估

Python 服务 `assess_candidate(runner, candidate_id)` 复用上述历史候选、前提有效性与逐轮覆盖接口，
并冻结当前完整纳入范围（`require_supported=False`）。新增文献、补读或复核后仍评估原候选 ID；
原候选不改写，失效的旧证据只作为历史依赖说明，不能通过旧 ID 冒充当前支持。纳入但没有正文、
阅读卡或支持决定的工作保留缺口，完整可读正文进入模型上下文，不因缺少复核而隐藏正文。

`InnovationAssessment` 使用 `schema_version=1`、`task_version="innovation-assessment-v1"`、
`statement_type="scope_limited_assessment"`。四个固定维度为 `novelty_clues`（新颖性线索）、
`research_value`（研究价值）、`feasibility`（可行性）和 `evidence_strength`（证据强度）。每维有
文本 `judgment`、当前 `claim_ids`/`evidence_ids` 及逐条有引用的 `reasons`。无引用意味着尚无
当前依据，不能据此认证结论。模型另提供 `gaps` 和最低验证的 `proposal`、
`falsification_condition`；后者是可证伪的拟议工作，不能写成已经执行的实验结果。没有综合分数。

宿主固定 `material_cutoff`、`material_scope`、`host_gaps` 和 `evidence_sufficiency`。
证据可用性只取 `insufficient` 或 `available_within_scope`：前者表示记录的覆盖/前提/当前链仍有
缺口，后者仅表示本范围内有可用链且没有记录的未解决缺口，均不认证新颖性或学术真实性。
未检索、空结果、失败、限流、截断、容量丢弃、未读、待复核、排除理由、撤稿等由共享读取接口
产生，模型不能自行删除。旧失败收据保留在历史尝试里；失败修复和命中补读后的当前状态重新读取，
不会把旧收据的未读状态永久贴在文献上。范围内的空结果也不证明全球不存在反例。
不属于候选挑战收据的普通搜索限制另以 `general_search_coverage` 保留原事件的查询、来源、
失败/截断/容量信息；候选查询成功不能替这些历史普通搜索抹掉覆盖限制。它不改变挑战行动状态。

生成与独立审查分别经过预算计量及领域验证后缓存。审查任务为 `innovation-assessment-review-v1`，
产出 `AssessmentReview`（schema 1），精确覆盖每维判断、每条理由、每条模型缺口及最低验证两个
文本字段，单元 ID 如 `novelty_clues.judgment`、`research_value.reasons.0`、`gaps.0` 和
`minimum_validation.falsification_condition`。审查收到完整当前文档、原候选、逐轮收据与当前命中，
不继承生成会话；保留 `supported`、`partial`、`contradicted`、`insufficient`，缺项/重复/未知引用拒绝。
宿主只能验证结构、引用链和覆盖，并不能代替对单元内每项事实及验证设计的独立语义判断。

结果沿用 `complete`、`budget_paused`、`needs_input`、`stale`、`failed` 操作状态，另带
`input_hash`、`output`、`candidate_ids`、`gaps`、`coverage_status`。`complete` 只表示生成、
独立审查和本地产物完成，仍待人工学术判断；可能同时存在 `not_searched` 或 `needs_reading`。
只增加预算能继续已缓存生成后的审查；研究文档、人工决定、筛选、问题、检索配置或实际查询改变
使原评估过期，评估自己的日志和计量记录不影响有效性。Exchange 等待不会重做已成功阶段。

完整输入超限时不派发该阶段、不截断，返回独立的 `model_input_too_large` 缺口。评估使用全部当前
纳入文献；应按真实研究范围、明确理由筛选不相关资料并保留候选原依据和相关反例。不能为了得到
结论排除反例，也不能自动改筛选。原依据不能保留时先处理当前性缺口；范围无法缩小时保留失败，
或由用户明确选择兼容的后端输入上限。该服务没有额外 `work_ids` 参数。

每个 `analyses/assessment/v-*` 版本经现有宿主清单保存七个文件：`candidate.json`、
`assessment.json`、`assessment-review.json`、`context.json`、`report.md`、`references.bib`、
`evidence.json`。`context.json` 固定原候选清单哈希、前提状态、所有适用轮次和当前命中。引用附表
闭合每个已导出 Claim/Decision 的证据依赖，BibTeX 来自宿主工作记录；Markdown 文本转义，引用表
来源 URL 使用既有安全 URL 规则过滤。没有新增 CSV 文件。生成文本/元数据在 JSON 中仍是数据。

`innovation_status(root)` 仅用只读研究和校验产物接口，不初始化 Store、不调用模型或来源，支持
路径中的 `#`/`%`。顶层为 `status`、`candidates`、`gaps`；每个候选分开给出
`historical_available`、`premises_current`/`premise_gaps` 和最新 `assessment`。
评估状态为 `missing`、`current`、`stale` 或 `invalid`，含 `available`、`current`、输出及缺口。
历史报告可以人工编辑并保留，但编辑正式版本会使完整性检查报错；系统不会覆盖或自动修复文件。
建议修改报告副本。最新评估即使是有效的缺口报告，其 `current` 也只表示输入未变，不能替代
`premises_current` 或证据可用性检查。

固定 Demo 评估只接受原始 A/B 正文及元数据：A 为 40 个模拟作业、等待时间 10→8（20%下降）；
B 为 60 个突发到达作业、10→11。它展示协议，不是通用理解或学术准确率测试。新增原创 C 的
“80 个模拟任务、10→10、未观察到改善、未测试切换规则”在测试中使用显式受控阅读/复核/评估
响应；C 补读后同一候选可引用它的当前链，固定 Demo 仍拒绝未知 C。真实模型调用与人工学术准确率
验收在此任务中未执行。
