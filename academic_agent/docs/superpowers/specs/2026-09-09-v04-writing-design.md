# 0.4 写作与领域扩展：设计补充

2026-09-09。独立设计审查提出的五项接口冲突已合并修正并通过限定复审，可进入实现；这不代表功能、测试或学术质量已通过。绑定 docs/spec-v0.4.md 及 docs/writing-contract-notes.md 全部范围，接口按 f85cb4c 核对。0.3 已获用户确认、原子推送及远端核验，正式 main 617d557 的六环境 CI 34353904196 全部通过。本阶段从已发布 develop 46d3630 的隔离 feature/writing-v04 开发，仍须完成自身所有实现、验收和发行门槛。

对应计划为 academic_agent/docs/superpowers/plans/2026-09-09-v04-writing.md。research-v04 工作树及离线环境已准备；本设计入库时尚未实施 0.4 源码。SDD 记录位于仓库 .superpowers/sdd/2026-09-09-v04-writing/，每任务一次合并审查覆盖规格与质量。

## 选择与代价

采用现有 ResearchSnapshot + 不可变 research_artifacts + MeteredBackend.validated 的顺序阶段。新增数据型领域包、领域提取、写作文本单元、独立复核与修订，不增加依赖、数据库表或另一套图引擎。阶段自身有输入身份和成果，暂停后重跑命令恢复。

备选一是 Markdown 标记块直接编辑，较亲切但标题、空行和段落边界转义更难无损解释；备选二是编辑器/图存储，超出本地应用范围。选择显式 JSON 段落副本：编辑操作更笨重，但新增、删除、调序、原文保存和逐字符覆盖规则可以精确定义。最终正文仍必须输出 Markdown、BibTeX 和证据审计。保守地使变化整段待审；大纲或所选资料变化时整稿过期，代价是多复核一些仍正确的文字。

## 一组明确的类型

所有持久化模型与 wire 模型使用 `ConfigDict(extra="forbid", strict=True)`，必需键不能省略，未知枚举/引用/字段拒绝。下列对象的花括号列出完整键集合；`?` 表示允许 null，不表示可省略。ID 均由宿主赋值；模型只用本任务给定 ID 或数组索引。

- `DomainPack={schema_version:1, contract:"writing-domain-1", name, version, terminology:[{term,meaning}], search_hints:[str], screening_hints:[str], fields:[{id,label,description,value_type:"text"|"number",unit:str?}], comparison_columns:[field_id], evaluation_criteria:[{id,text}], writing_structure:[{id,title,purpose}]}`。所有 ID 匹配 `[a-z][a-z0-9_]{0,63}`、唯一且交叉引用有效；非空字符串有上限，最多 24 字段、12 列、12 评价项、12 节；文件最多 256 KiB；禁止重复 JSON 键、NaN、BOM。包只有这组数据键，没有脚本、工具、权限、预算或系统指令字段。提示全部置于低权限任务资料区，不能覆盖宿主规则。
- `WritingSelection={document_id, question, work_ids, domain_path, trend_directory:str?, candidate_directory:str?, candidate_id:str?, assessment_directory:str?}`。`document_id` 为宿主 `doc_<32hex>`；创建时 question 非空，work_ids 显式非空且唯一。候选目录必须与 candidate_id 成对，assessment 必须属于该候选。趋势/候选/评估只接受具体版本目录；不接受 latest。领域入口支持 `builtin:scheduling` 或绝对 JSON 文件路径。
- `ArtifactPin={kind,version_path,manifest_hash,semantic_hash}`；`PackPin={source_path,sha256,pack}`；`WritingContext={schema_version:1,selection,research,pack,trend:TrendPin?,candidate:CandidatePin?,assessment:AssessmentPin?,challenge_context:AssessmentContext?,input_hash}`。三个 pin 包含 ArtifactPin 和从其已验证 bytes 解析的对象；来源、复核、所有轮次的缺口均保留。正文永远不从 pin 的目录重新随意读 JSON。
- 已加载 `DomainArtifact={pin:ArtifactPin,context:WritingContext,fields:dict[work_id,list[DomainField]],review:dict[work_id,list[DomainReviewItem]]}`。load_domain_version 从同一次 load_research 的 manifest+已校验 bytes 构建 pin 和对象；domain semantic_hash 对 context 的语义投影+fields+review 计算，排除自己的 pin。提取发布前直接从 context/fields/review 渲染，不制造未来 pin；写作只有在版本发布并严格加载后取得非空 pin。
- `SourceRef={type:"source",work_id,document_id,claim_id,evidence_ids:[str]}`，必须属于同一篇、同一有效文档，关联有效复核；`SynthesisRef={type:"synthesis",sources:[SourceRef]}` 至少两个不同 work；`StatisticRef={type:"frozen_statistic",pin_hash,computation_path}` 路径从宿主提供的数值目录选择，目录项含确定值、口径、分母、时间、算法与 input_hash；`HypothesisRef={type:"hypothesis",candidate_id,candidate_semantic_hash}`。统计值由宿主计算目录，模型不计算或改写对象值；句子是否准确表达仍由独立语义审查判断。
- `TextUnit={id,text,kind:"source"|"synthesis"|"frozen_statistic"|"hypothesis"|"connector",references:[Ref],purpose:str?}`。kind 对应同型引用；connector 要求非空 purpose，不能豁免语义检查，可用引用说明连接依据。标题也使用 TextUnit，不允许藏在未经审查的 title 字符串中。
- 仅 pending 持久化态另允许宿主创建 `kind="unclassified"`、references=[]、purpose="pending segmentation" 的单元，其 text 恰为提交块全部原文；模型生成/划分 schema 不包含此 kind，完成态禁止它。这样 raw-edit.bin 与结构化 pending 正文都保存真实错句，不能靠空 units 隐藏未划分文字。
- `TextBlock={id,role:"heading"|"paragraph",level:int?,units:[TextUnit]}`；heading level 为 1..3，paragraph 为 null。宿主正文是按顺序拼接 units.text；只有 Markdown 标题前缀、空行和审计脚注标记由宿主插入，不增加事实文字。主张就是非 connector 单元；允许多句单元但审查必须判断其中每一实质句，不把单一引用存在当作整单元支持。
- `Outline={blocks:[TextBlock]}`：包括标题和说明段；`Draft={blocks:[TextBlock]}`：包含真实跨文献段落，且在多文献范围中至少一个 synthesis 单元；不是阅读卡列表。写作结构是建议结构输入，可修改；导入大纲后新的完整大纲参与每个正文生成/复核身份。
- `ReviewItem={unit_id,decision:"supported"|"partial"|"contradicted"|"insufficient",reason}`；`WritingReview={items:[ReviewItem]}` 精确一项覆盖每个单元，包括标题和 connector，拒绝漏项/多项/重复。来源一律 model；报告标明独立模型复核而非人工验证。
- `OutlineReviewState={status:"pending"|"complete",review:WritingReview}`：pending 必须空 items；complete 必须精确覆盖完整大纲全部单元，允许 partial/contradicted/insufficient。正文生成显式接收此对象，将状态、每条决定和理由纳入任务输入及身份；pending 时先返回 needs_input 和 writing-review 动作，不调用正文模型。complete 表示复核执行完成，不表示所有大纲论断获支持。

## 稳定文本身份与内容范围

创建大纲先在既有 writing_progress 保存稳定 document_id，成功生成经结构/引用校验后、调用任何 review 之前分配文本 ID。新增 `assign_generated_ids(document_id:str,task:str,input_hash:str,wire_blocks:list[dict])->list[TextBlock]`：canonical 为 UTF-8 JSON、sort_keys=True、separators=(",",":"),ensure_ascii=False,allow_nan=False；H 为 SHA256。seed=H(canonical(["writing-ids-1",document_id,task,input_hash,wire_blocks]))；第 i 块 ID=`b_`+H(canonical([seed,i]))[:32]，第 j 单元 ID=`u_`+H(canonical([seed,i,j]))[:32]，索引从0开始，宿主拒绝任何碰撞。input_hash 是本阶段完整语义输入（不是未来产物版本），不含创建时间。缓存重放相同 wire_blocks 后重算必得相同 ID；无需待到发布才赋号。

人工现有块 ID 永远保留；新增块 ID=`b_`+H(canonical(["writing-edit-ids-1",document_id,parent_manifest_hash,input_sha256,new_marker]))[:32]，写入 id_map。pending 原文占位单元 ID=`u_`+H(canonical(["unclassified-1",block_id,input_sha256]))[:32]。新增 `assign_segment_ids(block_id:str,segmentation_input_hash:str,spans:list[dict],raw_text:str)->list[TextUnit]`：先验证连续全覆盖，再按 j 产生 `u_`+H(canonical(["writing-segment-ids-1",block_id,segmentation_input_hash,j,span]))[:32]，text 仅从原文截取。segmentation_input_hash 包含准确 raw_text、上下文/领域语义和大纲；不含新版本目录。未变块的有效划分保留原 ID。生成、划分缓存命中、审查后发布失败重试，均不重新随机分配 ID，确保 review job 指纹稳定。

Revision 分开两个维度：`extent:"outline"|"manuscript"` 与 `stage:"pending"|"reviewed"`。extent=outline 时 outline 非空、draft.blocks 必须为空，pending/reviewed 均合法；extent=manuscript 时两者非空，同样允许两个 stage。stage=pending 当前 review.items 必须为空，stage=reviewed 必须精确覆盖全部现有单元且无 unclassified。生成大纲是 outline/reviewed，实际正文是 manuscript/reviewed；人工导入保留父版 extent，不通过编辑 JSON 转换内容范围。`create_draft` 只接受依赖当前、extent=outline、stage=reviewed 的完整大纲版本（含已复核人工大纲），随后生成新的 manuscript/reviewed 版本。只改 JSON 排版的大纲仍是 outline/reviewed。

## 已校验字节和当前性

`trends.load_verified_trends(directory:Path)->VerifiedTrends` 新增只读入口，`VerifiedTrends` 冻结 dataclass 包含 `manifest:Manifest, files:dict[str,bytes], snapshot:CorpusSnapshot, computation:ComputationRecord, summary:dict`。files 恰为旧五产物及 manifest.json。把原 verify_trends 的读取/校验/重算逻辑移入此入口；只读一批 bytes、哈希校验这批 bytes、从它们解析、对它们确定性重算。`verify_trends(directory)` 仅返回 `.summary`，完整键集及旧六文件 bytes 不变。读取继续用 filesystem.io_path 支持普通 Windows 长路径，不混用扩展路径和普通路径比较；保持原路径安全拒绝行为。调用方保留 bytes；测试在读取结束后换文件，返回对象仍必须来自已验证 bytes，后续 fence 必须发现外部变化。

`challenges.load_candidate_version(directory:Path,candidate_id:str)->dict` 提取 load_candidate 的单版本严格校验，返回现有六键 `{candidate,batch,review,snapshot,version_path,manifest_hash}`；load_candidate 继续原历史选择语义。新增 `candidate_version_currentness(root:Path,loaded:dict)->dict` 复用 `_assert_premises`，不再按 ID 重新选择历史版本。

`innovations.load_assessment_version(root:Path,directory:Path)->dict` 提取现有 `_assessment_history` 的单版本校验和 host payload 闭包，返回 `{candidate,context,assessment,review,snapshot,version_path,manifest_hash}`。`assessment_version_currentness(root,loaded)->dict` 用所固定的候选来源、read_challenges、当前 research 和 `assessment_input_hash` 比较此版本，不读取 assessment/latest。旧 innovation_status 输出保持不变；不能拿其“最新评估 current”替代选定版本判断。候选 premises_current 和 assessment current 分别记录；当前缺口报告不表示创新性已证实。

`writing_context.load_writing_context(root,selection)->WritingContext` 捕获当前 ResearchSnapshot（完整所选原文）、包 bytes、具体统计/候选/评估 bytes。验证引用身份和领域字段证据链。当前候选与评估前提失效可作为明确历史设想/缺口展示，但不得展示为当前有效依据；新生成要求选定依赖当前或明确缩小选择。已冻结统计描述历史语料，只验证选定版本完整性、身份和显式选择，不将 CorpusSnapshot.source_snapshot 与当前 Store 的计量元组比较。它不是当前总体的新统计。

基础 WritingContext.input_hash 包括问题、范围及筛选、research_fingerprint、原 document.source_path、有效复核、包原始 bytes 哈希、选定统计 input/manifest、候选与评估内容及当前轮次缺口；不包含领域提取的未来输出。领域提取任务只依赖基础上下文。写作阶段身份在此基础上加入已验证 DomainArtifact 的完整字段+复核语义；大纲生成使用这两者，正文生成再加完整大纲及 OutlineReviewState，整稿审查再加完整有序正文及单元引用。排除自己的产物版本 UUID、目录、创建时间、状态日志和预算；稳定文稿/文本 ID 保持审查目标关联，不因新输出版本改变。候选/评估 republish 的 UUID/manifest 不单独改变语义身份，但原版本 pin 永远保留用于审计和完整性。不要从 semantic projection 中删除原文 source_path。包哪怕只改格式也过期，代价是保守但清晰。

缓存 job.payload 使用上述语义 projection，绝不把整份含 UUID/计量的持久化 envelope 放进去；审计 envelope 保存全部 pin。划分复用键为任务协议/指令/schema/model、context semantic hash、完整大纲 hash、精确 block text/role/level。独立审查身份还必须包含完整有序大纲和正文、所有单元划分/引用；正文任一内容或顺序变化后重做一次整稿审查，旧决定仅作为历史保留。复用未变提取/大纲生成/精确划分可减少调用，但不声称不同全文上下文中旧段落的语义决定仍然有效；代价是多复核未改段落。完全相同的全文重试命中缓存。单独新增预算或运行日志不新增成功调用。

## 外部文件变化窗口

基础入口 `assert_writing_current(root,context)->None` 和 `writing_currentness(root,context)->{current,gaps}` 用于没有领域输出的领域提取阶段，校验研究/原文/包/统计/候选/评估 pins。正文阶段增加必需参数入口 `assert_writing_dependencies_current(root:Path,context:WritingContext,domain_pin:ArtifactPin)->None` 和 `writing_dependencies_current(root:Path,context:WritingContext,domain_pin:ArtifactPin)->dict`（位于 writing.py）：先做基础检查，再用 load_domain_version(domain_pin.version_path) 校验指定 domain 版本的完整文件/manifest/语义哈希，检查其提取上下文与正文的 question、work_ids、research_fingerprint、pack.source_path/hash 一致。必须比较保存的 pin，不能换 latest；写作端任何模型前、validated 退出前（含缓存命中）、publish 每个 callback 及发布后/状态都使用后一入口，不仅检查 context。领域提取不引用自己的未来产物，因此无循环依赖。领域文件在初次加载后被修改，也必须在缓存提交或发布前被发现。

`load_domain_version` 与 `load_writing_version` 只做历史 bytes 完整性、严格对象和宿主产物闭包校验，不以“如今是否当前”拒绝完整历史；export_edit 可以导出完整但 stale 的人类稿件。写作状态分别报告 available 与 current；current 由上述完整领域依赖 fence 计算。捕获完整领域依赖后也做前后 fence；资料反复变化直接 stale，不无限重试或静默重算。领域源资料变化但该版本文件完整是 stale；领域文件损坏是 invalid，不能当成完整历史。

给 `publish_research(..., *, dependency_check:Callable[[],None]|None=None)` 增加可选宿主回调：开始、staging 完成、SQLite 锁内 rename 前、验证 destination 后且 latest 更新前、latest 更新后调用。旧调用不变。外部文件不受 SQLite 锁保护；检查点之间仍存在竞态，不能承诺跨文件原子性。若 rename 后发现变化，保留完整历史目录；latest 前失败不移动指针，latest 后失败不回滚覆盖别人指针，返回 stale 和 output，所有读取重新算当前性。产物 manifest 仅表示已验证捕获输入，不能静态承诺“永远 current”。发生无法读取的依赖则 invalid；历史 bytes 完整但当前身份变化则 stale。

`research_artifacts.KINDS` 与 `_Manifest.kind` 显式增加 `domain`、`writing`，不得伪装 assessment。新 kind 的 payload 集合在 publish 和 load 均严格验证；旧三 kind 不收紧或改变格式。各新版本都有宿主 manifest，正文、引用及审查在同一版本内闭包校验。所有生成字段都由宿主渲染，直接改正式文件视为哈希损坏，保留原件并提示 export-edit/import-revision。

## 领域任务与正文任务

领域提取逐篇调用 `domain-extract-v1`，完整这篇文档及对应有效 Evidence/Claim、包字段定义进入模型；提取结果 `{fields:[{field_id,status:"reported"|"not_reported",value:str|number|null,evidence_ids:[str]}]}` 精确覆盖字段。reported 有值且至少一个本篇有效 evidence；not_reported 为 null+空 evidence，显示“文中未报告”。独立 `domain-review-v1` 接收完整原文、全部字段值/缺失项、全部本篇有效证据与字段定义；输出每一字段 `{field_id,decision,reason}`，包括未报告项，检查遗漏，不仅阅读字段引文片段。宿主比较表只将 supported 值作有效值，其他值带待审/冲突，不作依据；列由 pack.comparison_columns 生成，字段类型或任意外来 evidence 拒绝。

内置 scheduling 包字段 `simulated_jobs,arrival_pattern,waiting_time,baseline`：A 为40，B为60，A到达分布、B独立基线细节为未报告。正文保留 A 10→8 与 B 10→11、负载差异与反例，不把40/60称为真实参与者或外推真实系统。字段值本身不能自动授予正文支持。

另一个包由使用旅程写在安装源码外 `inputs/external-domain.json`，字段 `observation_unit,transfer_limit`、列顺序不同、评价项 `scope_transfer`、结构 `boundary_argument`。使用相同原创资料，由显式 Exchange 响应提取“simulated jobs”和两工作负载无法外推真实系统的限制，均绑本篇有效 evidence；不是在核心代码识别字段名。安装前后核心文件哈希一致。Demo 仅识别内置包哈希+原创协议；未知包/task 返回明确 failed，不能凭 field 名编造结果。

`writing-outline-v1` 生成完整大纲；`writing-draft-v1` 生成真实综述/相关工作/拟研究论证，其必需输入是 `{task,context,domain,outline,outline_review_state}`，正文不得把传入的大纲冲突/不足决定改述成已支持；这些决定和状态改变必须改变生成身份。`writing-review-v1` 是独立调用，接收完整限定上下文、完整大纲及稿件、每个单元和引用，不继承写作者对话。target_ids 必须恰为完整大纲+正文所有单元 ID，outline-only 时正文为空；任意正文/大纲变化后完整覆盖审查，不以局部决定代替整稿语义检查。每阶段都在 validated 中完成结构/引用/覆盖校验后才缓存。完整 prompt 超 max_input_chars 直接 input_too_large，并提示显式减少 work_ids；不截断反例、局限、OCR/摘要限定。

## 精确人工修订协议

`export_edit(root,version,destination)->Path` 只读已校验具体 writing 版本，以 exclusive-create 写 UTF-8 JSON 副本。完整键：`{schema_version:1,document_id,parent_version,parent_manifest_hash,outline:[EditableBlock],paragraphs:[EditableBlock]}`；EditableBlock 精确 `{id,role,level,text}`，outline 允许 heading/paragraph，paragraphs 也含正文 heading。text 是宿主拼接 units.text 原字串，不含宿主插入的审计脚注；副本中不提供可编辑支持标签。

副本允许 UTF-8 BOM（只剥文件起始一个，raw bytes 原封保存），允许 JSON 外部 CRLF/LF；JSON text 中的 `\r\n`、空格、组合字符及非 BMP 字符不归一化。拒绝重复 JSON 键、NaN、孤立 surrogate、NUL、空白整块、空文稿和非法字段。父版本/manifest/document 必须三者精确匹配且完整，允许从历史父版分支，不猜 latest。已有 ID 必须属于父版对应数组且唯一；新增 ID 使用唯一 `new:<slug>`，宿主替换为新的稳定 ID 并保存映射；删除必须在独立顶层 import CLI `--delete-id` 列表明确列出，列表恰等于父版缺失 ID（防止误删）；顺序就是 JSON 数组顺序；禁止把原 ID跨 outline/paragraphs 搬迁。作者与理由通过必需 CLI 参数记录，非空。

`import_revision(runner,parent_version,edit_file,*,author,reason,delete_ids,domain_version:Path|None=None)->dict` 无模型调用。先完整校验父版和输入；没有替代版本时要求父 domain pin 当前，不能捕获新资料却保留旧 domain 结果。若父领域依赖 stale，保留输入不动，返回 stale 与已知 question/work_ids/domain_path 对应的 `domain-extract` argv，及带相同父版/文件/作者/理由的 `writing-import ... --domain-version` 恢复动作（新路径由 extract 的实际 output 填入，不猜 UUID）。

显式 domain_version 只接受具体已验证/current 的 domain 目录，question 和 work_ids 必须与父写作选择相同；pack.source_path、pack.name、contract 必须同源，当前包 bytes/hash/version/字段可以更新但必须与本次实际 loader 一致。重新捕获 WritingContext，验证新 domain 的 research_fingerprint 与当前所选原文/有效复核相符；保持原 statistic/candidate/assessment 的显式选择且仍执行其既有检查，不静默换这些 pins。记录 `domain_rebind={old:ArtifactPin,new:ArtifactPin}`，没有显式替换为 null。父版和原领域产物不改，revision 的 author/reason 同时解释重新绑定。

成功导入保存 raw bytes、SHA256、父引用、输入块原 text、变化/删除/新增/调序列表、作者/原因/时间，先发布 pending 修订。text/role/level 精确变化的块、或来源/大纲/领域语义依赖变化的块进入 pending_ids；重新绑定后语义不同则全部大纲+正文清空旧引用，以 exact-text unclassified 单元保留并待重划分/整稿复核。语义完全相同的新版领域 republish 仅更新审计 pin，不无故废掉缓存。未变且上下文相同的块可保留精确单元划分与来源 pin。正文/顺序/大纲/依赖语义任一变化使当前 review.items 为空；旧决定仅沿父版作为历史查看。格式变化仅记文件层修订，内容与依赖完全相同时保留整份已验证 review 且 stage=reviewed、extent 保持原值。没有模型在重新绑定时改写人工 text。

`segment_revision` 使用 `writing-segment-v1` 返回 `{blocks:[{block_id,spans:[{start,end,kind,references,purpose}]}]}`，不允许 text 字段。只对 pending 块划分；start/end 是 Python Unicode code-point 半开区间，严格 int 且排除 bool，每块必须 `start0=0, end_last=len(text), end_i=start_next, end_i>start_i`。宿主用原文 `text[start:end]` 创建单元；完整 raw edit 留存，不允许模型改写、删除或美化错误。随后独立 review 必须看到原始 text 与划分，包括藏在 connector 里的数字错误。

`review_writing(runner,version)->dict` 把 pending 划分、review 结果发布为相同 extent、stage=reviewed 的新版本，旧 pending 版本不变。重跑成功版本复用成果，无需重复模型调用。extent=outline 的空正文修订可正常重划分/完整审查，之后 create_draft 消费完整 OutlineReviewState；pending 大纲提示先复核，complete 但有冲突/不足的大纲继续携带这些决定供正文生成，不被默认为全支持。

## 产物、状态和验收

domain 版本 payload 恰为 `context.json,domain.json,domain-review.json,comparison.md,references.bib,evidence.json`；writing 版本恰为 `context.json,outline.json,draft.json,review.json,revision.json,raw-edit.bin,report.md,references.bib,evidence.json`。初生模型稿 raw-edit.bin 为空 bytes，revision.json origin=model；人工稿 origin=human，raw bytes 必须 hash 匹配。stage=pending 时 review 恰为 `{items:[]}`，pending_ids 仅指待重新划分的块；即使 pending_ids 为空（纯调序），当前整稿审查仍未完成。stage=reviewed 时 review 精确覆盖全部现有大纲+正文单元，不得混入旧上下文决定；是否允许空正文仅由 extent 决定，与 pending/reviewed 无关。

revision.json 完整键为 `{schema_version:1,document_id,extent:"outline"|"manuscript",stage:"pending"|"reviewed",parent_version:str?,parent_manifest_hash:str?,origin:"model"|"human",author,reason,created_at,input_sha256:str?,changed_ids,added_ids,deleted_ids,reordered:bool,id_map:dict,pending_ids,reused_from:dict[block_id,ArtifactPin],domain_rebind:{old:ArtifactPin,new:ArtifactPin}?}`；生成 author 为 backend/model，domain_rebind=null。每块作者/模型、创建时间、父版由本版 revision 或 reused_from 链确定，evidence.json 逐块展开来源，不把重新审查者改成原文作者。人工稿的复核子版本继续保存原 raw-edit.bin/hash、人类作者和原 domain_rebind 记录，独立复核者/模型/job/时间另写 evidence.json。reused_from 表示内容或划分来源而不是当前支持状态。无额外自由模型文本放入 report；报告来自全部单元和结构化决定，含明确局限和修复命令。

`writing_status(root,document_id=None)->dict` 不创建 Store、不模型/搜索，只返回 `{status,documents,gaps}`；每文稿 `{document_id,versions}`；每版本 `{version,available,current,extent,stage,review_complete,gaps,output}`。当前性使用完整 context+domain_pin 检查，与内容范围/复核完成独立；反例/insufficient 不等于执行失败。预算/失败进度沿现有 Store logs 记录 `writing_progress`/`domain_progress`，每项 `{document_id,input_hash,stage,status,output,error}`，status 为 complete/paused_budget/needs_input/input_too_large/stale/failed；这里 progress.stage 是任务阶段名称，不是 Revision.stage。guide 给出补原文、review、domain-extract、writing-draft、writing-import --domain-version、writing-review 或重新选择/生成依赖的具体 argv。

验收分开报告：宿主结构拒绝、受控 Exchange/Demo 语义协议、有限真实 Backend 观察、人工学术判断。固定负例覆盖改错数字、遗漏否定、藏负载条件、无依据普遍化、模拟任务冒充参与者、假设冒充已观察结果、标题和 connector 藏事实。真实后端最多4次新增调用：单篇外部包提取+独立领域复核2次，跨论文实际 draft+独立 writing review 2次；沿已有授权/预算，不能自动调高或因失败自动重试。分别记录完整结果及人工核对观察，不把预设响应报告为准确率。完整套件只在发行节点执行一次；按失败补定向测试；安装包闭环、独立最终审查、六环境 CI 和既有发布确认全部完成后才判发行完成。
